package encoder

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// Item 9b (docs/internal/task-memory-campaign.md): the word-embedding table in the checkpoint's own precision, read
// from the mapping. F16/BF16 tables are mapped 16-bit views widened one row per token.

// stEntry is one tensor of a test checkpoint: dtype and shape, and its little-endian bytes.
type stEntry struct {
	dtype string
	shape []int
	raw   []byte
}

// writeStFile writes tensors as a safetensors file. The header is space-padded so the data starts at a multiple of 8
// — or, with odd, at an odd offset, which puts every 2-byte value on an odd address in the mapping.
func writeStFile(t *testing.T, path string, tensors map[string]stEntry, odd bool) {
	t.Helper()
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	header := map[string]any{}
	var payload []byte
	for _, n := range names {
		e := tensors[n]
		header[n] = map[string]any{"dtype": e.dtype, "shape": e.shape, "data_offsets": []int{len(payload), len(payload) + len(e.raw)}}
		payload = append(payload, e.raw...)
	}
	hb, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	for (8+len(hb))%8 != 0 {
		hb = append(hb, ' ')
	}
	if odd {
		hb = append(hb, ' ')
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(hb)))
	out = append(append(out, hb...), payload...)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}
}

// halfBytes encodes f32 values as dtype ("F16" or "BF16") little-endian bytes.
func halfBytes(dtype string, v []float32) []byte {
	out := make([]byte, 2*len(v))
	for i, f := range v {
		h := linalg.F32ToF16(f)
		if dtype == "BF16" {
			h = linalg.F32ToBF16(f)
		}
		binary.LittleEndian.PutUint16(out[2*i:], h)
	}
	return out
}

// The row decode matches embed's widening for every 16-bit pattern — NaNs, infinities and subnormals included — so a
// mapped table and a widened one agree bit for bit by construction.
func TestWordTable_decodeMatchesEmbedExhaustively(t *testing.T) {
	all := make([]byte, 2*65536)
	for i := range 65536 {
		binary.LittleEndian.PutUint16(all[2*i:], uint16(i))
	}
	p := filepath.Join(t.TempDir(), "all.safetensors")
	writeStFile(t, p, map[string]stEntry{
		"f16":  {"F16", []int{256, 256}, all},
		"bf16": {"BF16", []int{256, 256}, all},
	}, false)
	st, err := embed.OpenSafetensorsMmap(p)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	for _, name := range []string{"f16", "bf16"} {
		q := mappedWordTable(st, name, 256, 256)
		if q == nil {
			t.Fatalf("%s: no mapped view of an aligned table", name)
		}
		tn, _ := st.Tensor(name)
		var want []float32
		if name == "f16" {
			want, err = tn.Float16sToF32()
		} else {
			want, err = tn.BFloat16sToF32()
		}
		if err != nil {
			t.Fatal(err)
		}
		row := make([]float32, 256)
		for id := range 256 {
			q.rowInto(id, row)
			for j, v := range row {
				if math.Float32bits(v) != math.Float32bits(want[id*256+j]) {
					t.Fatalf("%s pattern %#04x: row decode %#08x, embed %#08x", name, id*256+j, math.Float32bits(v), math.Float32bits(want[id*256+j]))
				}
			}
		}
	}
}

// tinyCheckpoint writes a seeded BERT or GTE checkpoint with every tensor in dtype into a fresh dir: vocab 20,000 ×
// hidden 64 (a 4.9 MiB f32 table, so the heap gate can see it), one layer.
func tinyCheckpoint(t *testing.T, arch, dtype string, odd bool) string {
	t.Helper()
	const V, D, H, I, P = 20000, 64, 4, 128, 64
	rng := rand.New(rand.NewSource(int64(len(arch) + len(dtype))))
	ts := map[string]stEntry{}
	add := func(name string, shape ...int) {
		n := 1
		for _, s := range shape {
			n *= s
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * 0.1
		}
		if strings.HasSuffix(name, "LayerNorm.weight") || strings.HasSuffix(name, "ln.weight") {
			for i := range v {
				v[i] += 1
			}
		}
		ts[name] = stEntry{dtype, shape, halfBytes(dtype, v)}
	}
	cfg := map[string]any{"vocab_size": V, "hidden_size": D, "num_hidden_layers": 1, "num_attention_heads": H,
		"intermediate_size": I, "max_position_embeddings": P, "type_vocab_size": 2, "layer_norm_eps": 1e-12, "hidden_act": "gelu"}
	add("embeddings.word_embeddings.weight", V, D)
	add("embeddings.token_type_embeddings.weight", 2, D)
	add("embeddings.LayerNorm.weight", D)
	add("embeddings.LayerNorm.bias", D)
	switch arch {
	case "bert":
		add("embeddings.position_embeddings.weight", P, D)
		p := "encoder.layer.0."
		for _, n := range []string{"attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"} {
			add(p+n+".weight", D, D)
			add(p+n+".bias", D)
		}
		add(p+"attention.output.LayerNorm.weight", D)
		add(p+"attention.output.LayerNorm.bias", D)
		add(p+"intermediate.dense.weight", I, D)
		add(p+"intermediate.dense.bias", I)
		add(p+"output.dense.weight", D, I)
		add(p+"output.dense.bias", D)
		add(p+"output.LayerNorm.weight", D)
		add(p+"output.LayerNorm.bias", D)
	case "gte":
		cfg["position_embedding_type"], cfg["rope_theta"] = "rope", 10000.0
		p := "encoder.layer.0."
		add(p+"attention.qkv_proj.weight", 3*D, D)
		add(p+"attention.qkv_proj.bias", 3*D)
		add(p+"attention.o_proj.weight", D, D)
		add(p+"attention.o_proj.bias", D)
		add(p+"attn_ln.weight", D)
		add(p+"attn_ln.bias", D)
		add(p+"mlp.up_gate_proj.weight", 2*I, D)
		add(p+"mlp.down_proj.weight", D, I)
		add(p+"mlp.down_proj.bias", D)
		add(p+"mlp_ln.weight", D)
		add(p+"mlp_ln.bias", D)
	}
	dir := t.TempDir()
	cb, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cb, 0o644); err != nil {
		t.Fatal(err)
	}
	writeStFile(t, filepath.Join(dir, "model.safetensors"), ts, odd)
	return dir
}

type embedModel interface {
	Embed([]int32) []float32
	Close() error
	wordEmbHeapBytes() int
}

func loadTiny(t *testing.T, arch, dir string, native bool) embedModel {
	t.Helper()
	var m embedModel
	var err error
	if arch == "bert" {
		m, err = loadBERT(dir, native)
	} else {
		m, err = loadGTE(dir, native)
	}
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// tinyIDs is 40 seeded id sequences over the whole vocab (every row region is gathered).
func tinyIDs() [][]int32 {
	rng := rand.New(rand.NewSource(3))
	out := make([][]int32, 40)
	for i := range out {
		ids := make([]int32, 1+rng.Intn(60))
		for j := range ids {
			ids[j] = int32(rng.Intn(20000))
		}
		out[i] = ids
	}
	return out
}

func sameEmbeds(t *testing.T, what string, a, b embedModel) {
	t.Helper()
	for i, ids := range tinyIDs() {
		x, y := a.Embed(ids), b.Embed(ids)
		for j := range x {
			if math.Float32bits(x[j]) != math.Float32bits(y[j]) {
				t.Fatalf("%s: sequence %d component %d: %v vs %v", what, i, j, x[j], y[j])
			}
		}
	}
}

// heapAfter is HeapAlloc after load and two GCs, minus before, with the model kept alive (signed).
func heapAfter(t *testing.T, load func() embedModel) (int64, embedModel) {
	runtime.GC()
	runtime.GC()
	var a, b runtime.MemStats
	runtime.ReadMemStats(&a)
	m := load()
	runtime.GC()
	runtime.GC()
	runtime.ReadMemStats(&b)
	return int64(b.HeapAlloc) - int64(a.HeapAlloc), m
}

// Gates (a) and (b), model-free: on F16 and BF16 checkpoints of a tiny BERT and GTE, the default load's table is a
// mapped 16-bit view, its outputs are bit-identical to the widened table's, and it adds no heap for the table: the
// native load's heap ≤ the widened load's − the f32 table + 1 MiB.
func TestNativeWordEmb_halfCheckpoints(t *testing.T) {
	const tableF32 = 20000 * 64 * 4
	for _, arch := range []string{"bert", "gte"} {
		for _, dtype := range []string{"F16", "BF16"} {
			t.Run(arch+"/"+dtype, func(t *testing.T) {
				dir := tinyCheckpoint(t, arch, dtype, false)
				wHeap, widened := heapAfter(t, func() embedModel { return loadTiny(t, arch, dir, false) })
				nHeap, native := heapAfter(t, func() embedModel { return loadTiny(t, arch, dir, true) })
				defer widened.Close()
				defer native.Close()
				if native.wordEmbHeapBytes() != 0 {
					t.Fatalf("the default load did not map the %s table (%d heap bytes)", dtype, native.wordEmbHeapBytes())
				}
				sameEmbeds(t, "native vs widened", native, widened)
				t.Logf("heap after load: widened %.2f MiB, native %.2f MiB (table f32 %.2f MiB)", float64(wHeap)/(1<<20), float64(nHeap)/(1<<20), float64(tableF32)/(1<<20))
				if nHeap > wHeap-tableF32+1<<20 {
					t.Fatalf("native load heap %d B, want ≤ widened %d − table %d + 1 MiB", nHeap, wHeap, tableF32)
				}
			})
		}
	}
}

// Gate (c): a checkpoint whose data starts at an odd offset cannot be viewed as 16-bit values, so the default load
// falls back to the widened copy — and its outputs equal those of the aligned copy of the same weights.
func TestNativeWordEmb_misalignedFallsBackToCopy(t *testing.T) {
	for _, arch := range []string{"bert", "gte"} {
		for _, dtype := range []string{"F16", "BF16"} {
			t.Run(arch+"/"+dtype, func(t *testing.T) {
				aligned := loadTiny(t, arch, tinyCheckpoint(t, arch, dtype, false), true)
				defer aligned.Close()
				odd := loadTiny(t, arch, tinyCheckpoint(t, arch, dtype, true), true)
				defer odd.Close()
				if odd.wordEmbHeapBytes() != 20000*64*4 {
					t.Fatalf("misaligned %s table: %d heap bytes, want the widened f32 copy (%d)", dtype, odd.wordEmbHeapBytes(), 20000*64*4)
				}
				sameEmbeds(t, "misaligned (copy) vs aligned (view)", odd, aligned)
			})
		}
	}
}

// The opt-ins still convert from a mapped 16-bit table: WordEmbF16 keeps an F16 checkpoint's view as is, and makes a
// heap copy from BF16; WordEmbInt8 quantizes either.
func TestNativeWordEmb_optInsFromHalfTables(t *testing.T) {
	for _, dtype := range []string{"F16", "BF16"} {
		dir := tinyCheckpoint(t, "bert", dtype, false)
		for _, p := range []WordEmbPrecision{WordEmbF16, WordEmbInt8} {
			m, err := LoadBERTWithOptions(dir, LoadOptions{WordEmb: p})
			if err != nil {
				t.Fatal(err)
			}
			want := 0 // F16 opt-in on an F16 checkpoint: the mapped view
			if !(p == WordEmbF16 && dtype == "F16") {
				want = map[WordEmbPrecision]int{WordEmbF16: 20000 * 64 * 2, WordEmbInt8: 20000*64 + 20000*4}[p]
			}
			if got := m.wordEmbHeapBytes(); got != want {
				t.Errorf("%s checkpoint, %v: table holds %d heap bytes, want %d", dtype, p, got, want)
			}
			_ = m.Close()
		}
	}
}

// Gates (a) and (b) on a real model: minilm converted to F16 and to BF16. Model-gated.
func TestNativeWordEmb_minilmConverted(t *testing.T) {
	src := filepath.Join("..", "testdata", "minilm-model")
	if _, err := os.Stat(filepath.Join(src, "model.safetensors")); err != nil {
		t.Skip("testdata/minilm-model/ not present")
	}
	st, err := embed.OpenSafetensorsMmap(filepath.Join(src, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	for _, dtype := range []string{"F16", "BF16"} {
		dir := t.TempDir()
		ts := map[string]stEntry{}
		for _, name := range st.Names() {
			tn, _ := st.Tensor(name)
			if tn.DType == "I64" { // minilm's position_ids buffer, copied as is
				v, err := tn.Int64s()
				must(t, err)
				raw := make([]byte, 0, 8*len(v))
				for _, x := range v {
					raw = binary.LittleEndian.AppendUint64(raw, uint64(x))
				}
				ts[name] = stEntry{"I64", tn.Shape, raw}
				continue
			}
			v, _ := tn.Float32s()
			ts[name] = stEntry{dtype, tn.Shape, halfBytes(dtype, v)}
		}
		writeStFile(t, filepath.Join(dir, "model.safetensors"), ts, false)
		for _, f := range []string{"config.json", "tokenizer.json", "sentence_bert_config.json"} {
			if b, err := os.ReadFile(filepath.Join(src, f)); err == nil {
				must(t, os.WriteFile(filepath.Join(dir, f), b, 0o644))
			}
		}
		wHeap, wm := heapAfter(t, func() embedModel { b, err := loadBERT(dir, false); must(t, err); return b })
		nHeap, nm := heapAfter(t, func() embedModel { b, err := LoadBERT(dir); must(t, err); return b })
		table := wordEmbF32Bytes(t, dir)
		w, n := wm.(*BERT), nm.(*BERT)
		for _, s := range wordEmbFixedTexts {
			x, _ := w.Encode(s)
			y, _ := n.Encode(s)
			for i := range x {
				if math.Float32bits(x[i]) != math.Float32bits(y[i]) {
					t.Fatalf("minilm %s %q: native differs from widened", dtype, s)
				}
			}
		}
		t.Logf("minilm as %s: heap after load widened %.1f MiB, native %.1f MiB (table f32 %.1f MiB); %d texts bit-identical",
			dtype, float64(wHeap)/(1<<20), float64(nHeap)/(1<<20), float64(table)/(1<<20), len(wordEmbFixedTexts))
		if nHeap > wHeap-int64(table)+1<<20 {
			t.Errorf("minilm %s: native heap %d, want ≤ %d", dtype, nHeap, wHeap-int64(table)+1<<20)
		}
		_ = w.Close()
		_ = n.Close()
	}
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
