package encoder

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"math/rand"
	"os"
	"runtime"
	"testing"
)

// spladeTileLs are the sequence lengths the row-tiling gate checks: one row, either side of a tile edge, a
// few tiles, and the model's 512-token ceiling.
var spladeTileLs = []int{1, 15, 16, 17, 64, 512}

// spladeSeededIDs is a [CLS] … [SEP] id sequence of length L drawn from the vocab, seeded by L.
func spladeSeededIDs(L, vocab int) []int32 {
	rng := rand.New(rand.NewSource(int64(1000 + L)))
	ids := make([]int32, L)
	for i := range ids {
		ids[i] = int32(1000 + rng.Intn(vocab-1000))
	}
	ids[0] = 101 // [CLS]
	if L > 1 {
		ids[L-1] = 102 // [SEP]
	}
	return ids
}

// spladeExpandHash hashes expandIDs over spladeTileLs: every term id and every weight's bits, in order.
func spladeExpandHash(s *SPLADE) string {
	h := sha256.New()
	var b [4]byte
	for _, L := range spladeTileLs {
		v := s.expandIDs(spladeSeededIDs(L, s.vocab))
		binary.LittleEndian.PutUint32(b[:], uint32(len(v.Terms)))
		h.Write(b[:])
		for i, tm := range v.Terms {
			binary.LittleEndian.PutUint32(b[:], tm)
			h.Write(b[:])
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v.Weights[i]))
			h.Write(b[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// spladeExpandHashBeforeTiling is spladeExpandHash on testdata/splade-model, captured on the tree before the row
// tiling (task-memory-campaign.md item 1), where expandIDs materialized the whole L × vocab logit matrix.
const spladeExpandHashBeforeTiling = "86f33233fdd684488ff4e9465b7e34b4e39cff4128642023664215dbeabbe465"

// Item 1, gate (a): tiling the vocabulary projection changes no bit of any expansion. Model-gated.
func TestSPLADE_expandMatchesBeforeTiling(t *testing.T) {
	const dir = "../testdata/splade-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skip("testdata/splade-model/ not present; see scripts/README.md")
	}
	s, err := LoadSPLADE(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	got := spladeExpandHash(s)
	t.Logf("expandIDs hash over L=%v: %s", spladeTileLs, got)
	if got != spladeExpandHashBeforeTiling {
		t.Fatalf("expandIDs hash %s, before tiling %s: the tiled expansion is not bit-identical", got, spladeExpandHashBeforeTiling)
	}
}

// spladeExpandBytesBeforeTiling is TotalAlloc of one 512-token expandIDs on testdata/splade-model, measured on
// the tree before the row tiling (min of 5 runs, M1 Pro).
var spladeExpandBytesBeforeTiling uint64 = 93_780_128

// Item 1, gate (b): one 512-token expand allocates at least 55 MiB less than before (the L × V logits were
// 59.6 MiB). Model-gated.
func TestSPLADE_expandBytes512(t *testing.T) {
	const dir = "../testdata/splade-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skip("testdata/splade-model/ not present; see scripts/README.md")
	}
	s, err := LoadSPLADE(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	ids := spladeSeededIDs(512, s.vocab)
	s.expandIDs(ids) // warm the scratch pool
	best := uint64(math.MaxUint64)
	for range 5 {
		var a, b runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&a)
		s.expandIDs(ids)
		runtime.ReadMemStats(&b)
		best = min(best, b.TotalAlloc-a.TotalAlloc)
	}
	t.Logf("TotalAlloc per 512-token expand: %d B = %.1f MiB (before tiling: %.1f MiB)", best, float64(best)/(1<<20), float64(spladeExpandBytesBeforeTiling)/(1<<20))
	if spladeExpandBytesBeforeTiling == 0 {
		t.Skip("before-figure not recorded yet")
	}
	if bar := spladeExpandBytesBeforeTiling - 55<<20; best > bar {
		t.Fatalf("TotalAlloc %.1f MiB per expand, want ≤ %.1f MiB (before − 55 MiB)", float64(best)/(1<<20), float64(bar)/(1<<20))
	}
}

// spladePoolWhole is the pre-tiling pooling, kept as the oracle: the whole L × V logit matrix, then the max.
func spladePoolWhole(t, decoderW, decoderB []float32, L, D, V int) []float32 {
	logits := make([]float32, L*V)
	matmulBTColsInto(t, decoderW, logits, L, D, V)
	addBias(logits, decoderB, L, V)
	pooled := make([]float32, V)
	for i := range L {
		for v, x := range logits[i*V : (i+1)*V] {
			if x > pooled[v] {
				pooled[v] = x
			}
		}
	}
	return pooled
}

// Item 1, model-free: tiled pooling is bit-identical to the whole matrix on synthetic heads, at every L in
// spladeTileLs, with negative-only columns and NaN logits in play.
func TestSpladePool_tiledMatchesWhole(t *testing.T) {
	const D, V = 48, 1500
	rng := rand.New(rand.NewSource(7))
	dec := make([]float32, V*D)
	for i := range dec {
		dec[i] = float32(rng.NormFloat64())
	}
	bias := make([]float32, V)
	for v := range bias {
		bias[v] = float32(rng.NormFloat64())
		if v%7 == 0 {
			bias[v] = -1e6 // a column no token can lift above zero
		}
	}
	bias[3] = float32(math.NaN()) // a NaN logit in every row of one column
	for _, L := range spladeTileLs {
		x := make([]float32, L*D)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		want := spladePoolWhole(x, dec, bias, L, D, V)
		got := make([]float32, V)
		spladePoolInto(got, x, dec, bias, L, D, V)
		for v := range got {
			if math.Float32bits(got[v]) != math.Float32bits(want[v]) {
				t.Fatalf("L=%d v=%d: tiled %v, whole %v", L, v, got[v], want[v])
			}
		}
	}
}
