package ann

import (
	"math"
	"math/rand/v2"
	"testing"
)

// randUnit makes a random L2-normalized vector (the index invariant).
func randUnit(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	var ss float64
	for i := range v {
		x := rng.NormFloat64()
		v[i] = float32(x)
		ss += x * x
	}
	inv := float32(1.0 / math.Sqrt(ss))
	for i := range v {
		v[i] *= inv
	}
	return v
}

func randUnitSet(rng *rand.Rand, n, dim int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		out[i] = randUnit(rng, dim)
	}
	return out
}

// TestHNSW_recallVsFlat: HNSW must return nearly the same top-k as the
// exact Flat scan. This is the headline correctness/quality bar.
func TestHNSW_recallVsFlat(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 43))
	const (
		n   = 2000
		dim = 64
		k   = 10
	)
	vecs := randUnitSet(rng, n, dim)
	flat := New(vecs)
	h := BuildHNSW(vecs, Config{M: 16, EfConstruction: 200, EfSearch: 64, Seed: 7})

	if h.Len() != n {
		t.Fatalf("Len=%d want %d", h.Len(), n)
	}

	const queries = 200
	var hit, total int
	for range queries {
		query := randUnit(rng, dim)
		exact := flat.Query(query, k)
		approx := h.Query(query, k)
		if len(approx) != k {
			t.Fatalf("approx returned %d hits, want %d", len(approx), k)
		}
		// Verify descending-score, ascending-index-tiebreak contract.
		for i := 1; i < len(approx); i++ {
			if approx[i].Score > approx[i-1].Score+1e-9 {
				t.Fatalf("results not descending: [%d]=%v [%d]=%v",
					i-1, approx[i-1].Score, i, approx[i].Score)
			}
		}
		exactSet := make(map[int]struct{}, k)
		for _, e := range exact {
			exactSet[e.Index] = struct{}{}
		}
		for _, a := range approx {
			if _, ok := exactSet[a.Index]; ok {
				hit++
			}
		}
		total += k
	}
	recall := float64(hit) / float64(total)
	t.Logf("recall@%d = %.4f over %d queries (n=%d, dim=%d)", k, recall, queries, n, dim)
	if recall < 0.90 {
		t.Errorf("recall@%d = %.4f, want ≥ 0.90", k, recall)
	}
}

// TestHNSW_higherEfHigherRecall: ef is the recall knob — more candidates
// should never reduce recall.
func TestHNSW_higherEfHigherRecall(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	const (
		n   = 1500
		dim = 48
		k   = 10
	)
	vecs := randUnitSet(rng, n, dim)
	flat := New(vecs)
	h := BuildHNSW(vecs, Config{M: 12, EfConstruction: 150, Seed: 3})

	recallAt := func(ef int) float64 {
		var hit, total int
		qrng := rand.New(rand.NewPCG(99, 100))
		for range 100 {
			query := randUnit(qrng, dim)
			exact := flat.Query(query, k)
			approx := h.QueryEf(query, k, ef)
			exactSet := make(map[int]struct{}, k)
			for _, e := range exact {
				exactSet[e.Index] = struct{}{}
			}
			for _, a := range approx {
				if _, ok := exactSet[a.Index]; ok {
					hit++
				}
			}
			total += k
		}
		return float64(hit) / float64(total)
	}
	low := recallAt(10)
	high := recallAt(200)
	t.Logf("recall ef=10: %.4f, ef=200: %.4f", low, high)
	if high < low-1e-9 {
		t.Errorf("higher ef reduced recall: ef10=%.4f ef200=%.4f", low, high)
	}
	if high < 0.95 {
		t.Errorf("ef=200 recall %.4f, want ≥0.95", high)
	}
}

// TestHNSW_determinism: same vectors + same Config (seed) ⇒ identical
// results, build to build.
func TestHNSW_determinism(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	vecs := randUnitSet(rng, 500, 32)
	cfg := Config{M: 16, EfConstruction: 100, Seed: 1234}
	h1 := BuildHNSW(vecs, cfg)
	h2 := BuildHNSW(vecs, cfg)

	qrng := rand.New(rand.NewPCG(7, 8))
	for q := range 50 {
		query := randUnit(qrng, 32)
		r1 := h1.Query(query, 10)
		r2 := h2.Query(query, 10)
		if len(r1) != len(r2) {
			t.Fatalf("query %d: len %d vs %d", q, len(r1), len(r2))
		}
		for i := range r1 {
			if r1[i].Index != r2[i].Index || r1[i].Score != r2[i].Score {
				t.Fatalf("query %d pos %d: %+v vs %+v", q, i, r1[i], r2[i])
			}
		}
	}
}

// TestHNSW_exactSmall: on a tiny index, ef large enough should find the
// true nearest exactly (graph is fully connected at this size).
func TestHNSW_exactSmall(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	vecs := randUnitSet(rng, 50, 16)
	flat := New(vecs)
	h := BuildHNSW(vecs, Config{Seed: 1})
	qrng := rand.New(rand.NewPCG(13, 14))
	for q := range 30 {
		query := randUnit(qrng, 16)
		want := flat.Query(query, 1)[0].Index
		got := h.QueryEf(query, 1, 50)[0].Index
		if got != want {
			t.Errorf("query %d: nearest got %d want %d", q, got, want)
		}
	}
}

func TestHNSW_edgeCases(t *testing.T) {
	// Empty index.
	h := NewHNSW(Config{})
	if h.Len() != 0 {
		t.Errorf("empty Len=%d", h.Len())
	}
	if got := h.Query([]float32{1, 0}, 5); got != nil {
		t.Errorf("empty Query should be nil, got %v", got)
	}
	// Single vector.
	h.Add([]float32{1, 0, 0})
	got := h.Query([]float32{1, 0, 0}, 5)
	if len(got) != 1 || got[0].Index != 0 {
		t.Fatalf("single-vec query: %+v", got)
	}
	if math.Abs(got[0].Score-1.0) > 1e-6 {
		t.Errorf("self-similarity score %v, want ~1", got[0].Score)
	}
	// k larger than Len returns all.
	h.Add([]float32{0, 1, 0})
	if got := h.Query([]float32{1, 0, 0}, 100); len(got) != 2 {
		t.Errorf("k>Len should return all (2), got %d", len(got))
	}
	// Dimension-mismatched query → nil.
	if got := h.Query([]float32{1, 0}, 1); got != nil {
		t.Errorf("dim-mismatch query should be nil, got %v", got)
	}
	// k<=0 → nil.
	if got := h.Query([]float32{1, 0, 0}, 0); got != nil {
		t.Errorf("k=0 should be nil, got %v", got)
	}
}

func TestHNSW_addDimMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected panic on dimension mismatch")
		}
	}()
	h := NewHNSW(Config{})
	h.Add([]float32{1, 0, 0})
	h.Add([]float32{1, 0}) // wrong dim
}

func BenchmarkHNSW_Query(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	const (
		n   = 50000
		dim = 64
	)
	vecs := randUnitSet(rng, n, dim)
	h := BuildHNSW(vecs, Config{M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	query := randUnit(rng, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = h.Query(query, 10)
	}
}

// BenchmarkFlat_Query is the O(N) baseline at the same scale, for the
// HNSW-vs-Flat speedup comparison.
func BenchmarkFlat_Query(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	const (
		n   = 50000
		dim = 64
	)
	vecs := randUnitSet(rng, n, dim)
	f := New(vecs)
	query := randUnit(rng, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = f.Query(query, 10)
	}
}
