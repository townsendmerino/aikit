package encoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"maps"
	"math"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"testing/fstest"
)

// tinyBERTTokenizer is a minimal WordPiece tokenizer.json over ids the tiny checkpoint's 20,000-row table covers.
const tinyBERTTokenizer = `{"version":"1.0","added_tokens":[
{"id":0,"content":"[PAD]","special":true},{"id":1,"content":"[UNK]","special":true},
{"id":2,"content":"[CLS]","special":true},{"id":3,"content":"[SEP]","special":true}],
"normalizer":{"type":"BertNormalizer","lowercase":true},
"pre_tokenizer":{"type":"BertPreTokenizer"},
"model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,
"vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"hello":4,"world":5,"##s":6,"model":7,"files":8}}}`

// bertModelFiles is every file LoadBERTFromFS reads, by its name under the model's root.
var bertModelFiles = []string{"config.json", "model.safetensors", "sentence_bert_config.json", "1_Pooling/config.json", "tokenizer.json"}

// tinyBERTModel writes a complete seeded BERT model folder (the tinyCheckpoint weights in BF16, plus a sentence_bert
// config, a CLS pooling declaration and a WordPiece tokenizer) and returns the folder with each file's bytes by name.
func tinyBERTModel(t *testing.T) (string, map[string][]byte) {
	t.Helper()
	dir := tinyCheckpoint(t, "bert", "BF16", false)
	extra := map[string]string{
		"sentence_bert_config.json": `{"max_seq_length":40,"do_lower_case":true}`,
		"1_Pooling/config.json":     `{"word_embedding_dimension":64,"pooling_mode_cls_token":true,"pooling_mode_mean_tokens":false}`,
		"tokenizer.json":            tinyBERTTokenizer,
	}
	for name, body := range extra {
		p := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	files := map[string][]byte{}
	for _, name := range bertModelFiles {
		b, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(name)))
		if err != nil {
			t.Fatal(err)
		}
		files[name] = b
	}
	return dir, files
}

// mapFSAt serves files under root, a folder that need not exist on disk.
func mapFSAt(root string, files map[string][]byte) fstest.MapFS {
	m := fstest.MapFS{}
	for name, b := range files {
		m[path.Join(root, name)] = &fstest.MapFile{Data: b, Mode: 0o644}
	}
	return m
}

// sameBERT asserts a and b loaded the same model: configuration, pooling, sequence cap, position offset, and Embed
// bit for bit on ids.
func sameBERT(t *testing.T, a, b *BERT, ids [][]int32) {
	t.Helper()
	if a.cfg != b.cfg || a.pool != b.pool || a.maxSeq != b.maxSeq || a.posOff != b.posOff {
		t.Fatalf("loads differ: cfg %+v pool %q maxSeq %d posOff %d, against cfg %+v pool %q maxSeq %d posOff %d",
			a.cfg, a.pool, a.maxSeq, a.posOff, b.cfg, b.pool, b.maxSeq, b.posOff)
	}
	if n := firstEmbedDiff(a, b, ids); n >= 0 {
		t.Fatalf("Embed differs on sequence %d", n)
	}
}

// firstEmbedDiff is the index of the first sequence whose embeddings are not bit-identical, or -1.
func firstEmbedDiff(a, b *BERT, ids [][]int32) int {
	for i, s := range ids {
		x, y := a.Embed(s), b.Embed(s)
		if len(x) != len(y) {
			return i
		}
		for j := range x {
			if math.Float32bits(x[j]) != math.Float32bits(y[j]) {
				return i
			}
		}
	}
	return -1
}

// LoadBERTFromFS over a folder that exists only in memory loads exactly what LoadBERT loads from disk.
func TestLoadBERTFromFS_matchesLoadBERT(t *testing.T) {
	dir, files := tinyBERTModel(t)
	want, err := LoadBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer want.Close()
	got, err := LoadBERTFromFS(mapFSAt("models/tiny", files), "models/tiny")
	if err != nil {
		t.Fatal(err)
	}
	defer got.Close()

	if got.pool != poolCLS || got.maxSeq != 40 {
		t.Fatalf("pool %q maxSeq %d, want cls and 40 from the model's own files", got.pool, got.maxSeq)
	}
	sameBERT(t, want, got, tinyIDs())
	for _, text := range []string{"hello world", "Hello models, files!", "unknown words here"} {
		x, err := want.Encode(text)
		if err != nil {
			t.Fatal(err)
		}
		y, err := got.Encode(text)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(x, y) {
			t.Fatalf("%q: Encode differs", text)
		}
	}

	// Break it first: one changed weight must show, or the comparison above proves nothing.
	bad := maps.Clone(files)
	bad["model.safetensors"] = slices.Clone(files["model.safetensors"])
	bad["model.safetensors"][len(bad["model.safetensors"])-1] ^= 0x3f
	broken, err := LoadBERTFromFS(mapFSAt("models/tiny", bad), "models/tiny")
	if err != nil {
		t.Fatal(err)
	}
	defer broken.Close()
	if firstEmbedDiff(want, broken, tinyIDs()) < 0 {
		t.Fatal("a changed weight left every embedding bit-identical: the parity check cannot fail")
	}
}

// countingFS counts every Open per name. Wrapped in countingReadFileFS it is an fs.ReadFileFS that counts each
// ReadFile too and hands out the map's own slices, the shape of a caller serving bytes it has already checked.
type countingFS struct {
	m  fstest.MapFS
	mu sync.Mutex
	n  map[string]int
}

func (c *countingFS) count(name string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.n[name]++
}

func (c *countingFS) Open(name string) (fs.File, error) { c.count(name); return c.m.Open(name) }

type countingReadFileFS struct{ *countingFS }

func (c countingReadFileFS) ReadFile(name string) ([]byte, error) {
	c.count(name)
	f, ok := c.m[name]
	if !ok {
		return nil, &fs.PathError{Op: "read", Path: name, Err: fs.ErrNotExist}
	}
	return f.Data, nil
}

// LoadBERTFromFS reads each of the model's five files exactly once and touches nothing else, through Open and through
// ReadFile; the tokenizer it parsed is the one Tokenizer returns.
func TestLoadBERTFromFS_readsEachFileOnce(t *testing.T) {
	_, files := tinyBERTModel(t)
	for _, viaReadFile := range []bool{false, true} {
		c := &countingFS{m: mapFSAt("m", files), n: map[string]int{}}
		var fsys fs.FS = c
		if viaReadFile {
			fsys = countingReadFileFS{c}
		}
		b, err := LoadBERTFromFS(fsys, "m")
		if err != nil {
			t.Fatal(err)
		}
		want := map[string]int{}
		for _, name := range bertModelFiles {
			want["m/"+name] = 1
		}
		if !maps.Equal(c.n, want) {
			t.Errorf("ReadFile=%v: reads %v, want each model file once: %v", viaReadFile, c.n, want)
		}
		tok := b.Tokenizer()
		if tok == nil {
			t.Fatalf("ReadFile=%v: no tokenizer from a WordPiece tokenizer.json", viaReadFile)
		}
		ids, err := tok.EncodeWithSpecials("hello worlds", b.maxSeq)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(ids, []int32{2, 4, 5, 6, 3}) {
			t.Errorf("ReadFile=%v: Tokenizer().EncodeWithSpecials = %v, want [2 4 5 6 3]", viaReadFile, ids)
		}
		b.Close()
	}
}

// Optional files fall back as LoadBERT's do; required ones fail the load and name what is missing.
func TestLoadBERTFromFS_missingFiles(t *testing.T) {
	_, files := tinyBERTModel(t)
	without := func(names ...string) fstest.MapFS {
		m := maps.Clone(files)
		for _, n := range names {
			delete(m, n)
		}
		return mapFSAt("m", m)
	}
	for _, c := range []struct{ name, wantErr string }{
		{"config.json", "read BERT config"},
		{"model.safetensors", "open BERT safetensors"},
	} {
		_, err := LoadBERTFromFS(without(c.name), "m")
		if err == nil || !strings.Contains(err.Error(), c.wantErr) || !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("without %s: err %v, want %q wrapping fs.ErrNotExist", c.name, err, c.wantErr)
		}
	}

	b, err := LoadBERTFromFS(without("sentence_bert_config.json", "1_Pooling/config.json", "tokenizer.json"), "m")
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	if b.pool != poolMean || b.maxSeq != 64 || b.Tokenizer() != nil {
		t.Errorf("pool %q maxSeq %d tokenizer %v, want the mean default, the position cap 64 and none", b.pool, b.maxSeq, b.Tokenizer())
	}
	if _, err := b.Encode("hello"); err == nil {
		t.Error("Encode without a tokenizer succeeded")
	}
}

// On the real bge-small files LoadBERTFromFS gives LoadBERT's vectors bit for bit and the reference's to cosine 0.9999,
// opening each file once; the tokenizer Tokenizer returns reproduces the reference's ids.
func TestBGE_loadBERTFromFSMatchesLoadBERT(t *testing.T) {
	const dir = "../testdata/bge-small"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no bge-small model at %s — fetch + run scripts/oracle/pin_bge.py", dir)
	}
	raw, err := os.ReadFile("../testdata/bge_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		Cases []struct {
			Text     string    `json:"text"`
			InputIDs []int32   `json:"input_ids"`
			Emb      []float32 `json:"embedding"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}

	want, err := LoadBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer want.Close()
	c := &countingFS{m: fstest.MapFS{}, n: map[string]int{}}
	for _, name := range bertModelFiles {
		b, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(name)))
		if err != nil {
			t.Fatal(err)
		}
		c.m[name] = &fstest.MapFile{Data: b, Mode: 0o644}
	}
	got, err := LoadBERTFromFS(countingReadFileFS{c}, ".")
	if err != nil {
		t.Fatal(err)
	}
	defer got.Close()
	for _, name := range bertModelFiles {
		if c.n[name] != 1 {
			t.Errorf("%s read %d times, want once", name, c.n[name])
		}
	}
	if len(c.n) != len(bertModelFiles) {
		t.Errorf("reads %v, want only the model's five files", c.n)
	}

	ids := make([][]int32, len(g.Cases))
	for i, cs := range g.Cases {
		ids[i] = cs.InputIDs
		tok, err := got.Tokenizer().EncodeWithSpecials(cs.Text, got.maxSeq)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(tok, cs.InputIDs) {
			t.Errorf("%q: Tokenizer ids %v, reference %v", cs.Text, tok, cs.InputIDs)
		}
		if cos := cos32(got.Embed(cs.InputIDs), cs.Emb); cos < 0.9999 {
			t.Errorf("%q: cosine %.6f to the reference < 0.9999", cs.Text, cos)
		}
	}
	sameBERT(t, want, got, ids)
}
