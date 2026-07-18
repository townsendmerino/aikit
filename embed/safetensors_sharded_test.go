package embed

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"
)

// buildF32Safetensors writes a minimal valid safetensors blob holding the given
// F32 tensors (sorted by name for a deterministic payload layout).
func buildF32Safetensors(tensors map[string][]float32) []byte {
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	// simple insertion sort (no import churn)
	for i := 1; i < len(names); i++ {
		for j := i; j > 0 && names[j] < names[j-1]; j-- {
			names[j], names[j-1] = names[j-1], names[j]
		}
	}
	header := make(map[string]any)
	var payload []byte
	off := 0
	for _, n := range names {
		d := tensors[n]
		raw := make([]byte, 4*len(d))
		for i, v := range d {
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
		}
		header[n] = map[string]any{"dtype": "F32", "shape": []int{len(d)}, "data_offsets": []int{off, off + len(raw)}}
		payload = append(payload, raw...)
		off += len(raw)
	}
	hb, err := json.Marshal(header)
	if err != nil {
		panic(err)
	}
	out := make([]byte, 8, 8+len(hb)+len(payload))
	binary.LittleEndian.PutUint64(out, uint64(len(hb)))
	out = append(out, hb...)
	out = append(out, payload...)
	return out
}

// two tensors split across two shards, named by an index.
func twoShardFixture() (index []byte, shard1, shard2 []byte, a, b []float32) {
	a = []float32{1, 2, 3, 4}
	b = []float32{-5, 6.5, 7}
	shard1 = buildF32Safetensors(map[string][]float32{"a.weight": a})
	shard2 = buildF32Safetensors(map[string][]float32{"b.weight": b})
	index = []byte(`{"metadata":{"total_size":1},"weight_map":{` +
		`"a.weight":"model-00001-of-00002.safetensors",` +
		`"b.weight":"model-00002-of-00002.safetensors"}}`)
	return
}

func checkTensor(t *testing.T, sf *SafetensorsFile, name string, want []float32) {
	t.Helper()
	tensor, err := sf.Tensor(name)
	if err != nil {
		t.Fatalf("Tensor(%q): %v", name, err)
	}
	got, err := tensor.Float32s()
	if err != nil {
		t.Fatalf("Float32s(%q): %v", name, err)
	}
	if len(got) != len(want) {
		t.Fatalf("%q len %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("%q[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

func TestShardedFromFS(t *testing.T) {
	index, shard1, shard2, a, b := twoShardFixture()
	fsys := fstest.MapFS{
		"m/model.safetensors.index.json":     {Data: index},
		"m/model-00001-of-00002.safetensors": {Data: shard1},
		"m/model-00002-of-00002.safetensors": {Data: shard2},
	}
	sf, err := OpenSafetensorsShardedFromFS(fsys, "m/model.safetensors.index.json")
	if err != nil {
		t.Fatalf("OpenSafetensorsShardedFromFS: %v", err)
	}
	checkTensor(t, sf, "a.weight", a) // shard 1
	checkTensor(t, sf, "b.weight", b) // shard 2
	if len(sf.Names()) != 2 {
		t.Errorf("Names() = %v, want 2 tensors across shards", sf.Names())
	}
	if _, err := sf.Tensor("missing"); err == nil {
		t.Error("expected error for missing tensor")
	}
}

func TestShardedMmap(t *testing.T) {
	index, shard1, shard2, a, b := twoShardFixture()
	dir := t.TempDir()
	for name, data := range map[string][]byte{
		"model.safetensors.index.json":     index,
		"model-00001-of-00002.safetensors": shard1,
		"model-00002-of-00002.safetensors": shard2,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	sf, err := OpenSafetensorsShardedMmap(filepath.Join(dir, "model.safetensors.index.json"))
	if err != nil {
		t.Fatalf("OpenSafetensorsShardedMmap: %v", err)
	}
	checkTensor(t, sf, "a.weight", a)
	checkTensor(t, sf, "b.weight", b)
	if err := sf.Close(); err != nil { // munmaps both shard regions
		t.Errorf("Close: %v", err)
	}
	if err := sf.Close(); err != nil { // idempotent
		t.Errorf("second Close: %v", err)
	}
}

func TestShardedIndexErrors(t *testing.T) {
	// empty weight_map
	if _, _, err := parseShardIndex([]byte(`{"weight_map":{}}`)); err == nil {
		t.Error("expected error for empty weight_map")
	}
	// H1: shard names that escape the model directory must be rejected as a
	// malformed format, not resolved against the filesystem (path traversal).
	for _, bad := range []string{
		`{"weight_map":{"w":"../../../etc/passwd"}}`,
		`{"weight_map":{"w":"/etc/passwd"}}`,
		`{"weight_map":{"w":"sub/dir/shard.safetensors"}}`,
		`{"weight_map":{"w":".."}}`,
	} {
		_, _, err := parseShardIndex([]byte(bad))
		if err == nil {
			t.Errorf("parseShardIndex(%s): expected rejection, got nil", bad)
			continue
		}
		if !errors.Is(err, ErrFormat) {
			t.Errorf("parseShardIndex(%s): error %v, want ErrFormat", bad, err)
		}
	}
	// a named tensor missing from its shard
	_, shard1, _, _, _ := twoShardFixture()
	fsys := fstest.MapFS{
		"m/model.safetensors.index.json": {Data: []byte(
			`{"weight_map":{"a.weight":"s1.safetensors","ghost.weight":"s1.safetensors"}}`)},
		"m/s1.safetensors": {Data: shard1},
	}
	if _, err := OpenSafetensorsShardedFromFS(fsys, "m/model.safetensors.index.json"); err == nil {
		t.Error("expected error when a weight_map tensor is absent from its shard")
	}
}
