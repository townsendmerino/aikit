package encoder

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"runtime"
	"testing"
)

// q8WordEmbTexts are the inputs of the item-2 gate: short and long, code and prose, as queries and documents.
var q8WordEmbTexts = []string{
	"how do I parse json in go",
	"func ParseJSON(b []byte) (map[string]any, error) { var m map[string]any; return m, json.Unmarshal(b, &m) }",
	"binary search over a sorted slice of integers",
	"def quicksort(xs):\n    if len(xs) < 2:\n        return xs\n    p = xs[0]\n    return quicksort([x for x in xs[1:] if x < p]) + [p] + quicksort([x for x in xs[1:] if x >= p])",
	"",
	"the quick brown fox jumps over the lazy dog 0123456789 !@#$%^&*()",
}

// q8EncodeHash hashes ModelQ8.Encode's output bits over q8WordEmbTexts, each as a query and as a document.
func q8EncodeHash(t *testing.T, m *ModelQ8) string {
	h := sha256.New()
	var b [4]byte
	for _, isQuery := range []bool{true, false} {
		for _, s := range q8WordEmbTexts {
			v, err := m.Encode(s, isQuery)
			if err != nil {
				t.Fatal(err)
			}
			for _, x := range v {
				binary.LittleEndian.PutUint32(b[:], math.Float32bits(x))
				h.Write(b[:])
			}
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// q8EncodeHashBefore is q8EncodeHash on testdata/encoder-model (CodeRankEmbed), captured on the tree that still
// cloned WordEmb onto the heap (task-memory-campaign.md item 2).
const q8EncodeHashBefore = "133525a31476c40e1c648dc33865ed68c6ca0b207fd7121f46dfac70ac25118e"

// q8HeapAfterLoadBefore is HeapAlloc after LoadQ8 and two GCs on that tree (min of 3 loads, M1 Pro).
var q8HeapAfterLoadBefore uint64 = 209_662_928

// Item 2: reading WordEmb from the mapping changes no output bit, and the heap after load drops by ≥ 80 MiB
// (the table is 89.4 MiB). Model-gated.
func TestModelQ8_wordEmbFromMapping(t *testing.T) {
	const dir = "../testdata/encoder-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no encoder model at %s — see scripts/README.md", dir)
	}
	best := uint64(math.MaxUint64)
	var m *ModelQ8
	for range 3 {
		if m != nil {
			closeQ8ForTest(m)
		}
		runtime.GC()
		runtime.GC()
		var a runtime.MemStats
		runtime.ReadMemStats(&a)
		var err error
		if m, err = LoadQ8(dir); err != nil {
			t.Skipf("q8 unsupported for this checkpoint: %v", err)
		}
		runtime.GC()
		runtime.GC()
		var b runtime.MemStats
		runtime.ReadMemStats(&b)
		best = min(best, b.HeapAlloc-a.HeapAlloc)
	}
	got := q8EncodeHash(t, m)
	t.Logf("Encode hash %s; heap after load %d B = %.1f MiB (before: %.1f MiB)", got, best, float64(best)/(1<<20), float64(q8HeapAfterLoadBefore)/(1<<20))
	if got != q8EncodeHashBefore {
		t.Errorf("Encode hash %s, before %s: reading WordEmb from the mapping changed an output", got, q8EncodeHashBefore)
	}
	if q8HeapAfterLoadBefore != 0 && best+80<<20 > q8HeapAfterLoadBefore {
		t.Errorf("heap after load %.1f MiB, want ≤ %.1f MiB (before − 80 MiB)", float64(best)/(1<<20), float64(q8HeapAfterLoadBefore-80<<20)/(1<<20))
	}
	closeQ8ForTest(m)
}

func closeQ8ForTest(m *ModelQ8) {
	if err := m.Close(); err != nil {
		panic(err)
	}
}
