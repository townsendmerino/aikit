package sparse

import (
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

// randCorpus builds n sparse vectors over a vocab of V terms, each with ~density
// non-zero positive weights — a stand-in for a SPLADE expansion.
func randCorpus(rng *rand.Rand, n, V, density int) []SparseVec {
	docs := make([]SparseVec, n)
	for d := range docs {
		seen := map[uint32]bool{}
		for j := 0; j < density; j++ {
			t := uint32(rng.IntN(V))
			if seen[t] {
				continue
			}
			seen[t] = true
			docs[d].Terms = append(docs[d].Terms, t)
			docs[d].Weights = append(docs[d].Weights, float32(rng.Float64()*2))
		}
	}
	return docs
}

// bruteScores is the independent reference: a dense per-document dot product,
// matching New's contract (skip weight ≤ 0; sum duplicate query terms).
func bruteScores(docs []SparseVec, q SparseVec) []float64 {
	qm := map[uint32]float64{}
	for i := 0; i < min(len(q.Terms), len(q.Weights)); i++ {
		qm[q.Terms[i]] += float64(q.Weights[i])
	}
	out := make([]float64, len(docs))
	for d, v := range docs {
		for i := 0; i < min(len(v.Terms), len(v.Weights)); i++ {
			if v.Weights[i] <= 0 {
				continue
			}
			out[d] += qm[v.Terms[i]] * float64(v.Weights[i])
		}
	}
	return out
}

func TestScores_matchesBruteForce(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	docs := randCorpus(rng, 2000, 5000, 80)
	ix := New(docs)
	for qi := 0; qi < 30; qi++ {
		q := randCorpus(rng, 1, 5000, 20)[0]
		got := ix.Scores(q)
		want := bruteScores(docs, q)
		for d := range want {
			if diff := math.Abs(got[d] - want[d]); diff > 1e-9*(math.Abs(want[d])+1) {
				t.Fatalf("query %d doc %d: score %.12f, brute %.12f (diff %.2e)", qi, d, got[d], want[d], diff)
			}
		}
	}
}

func TestQuery_topKMatchesSortedScores(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	docs := randCorpus(rng, 500, 1000, 40)
	ix := New(docs)
	q := randCorpus(rng, 1, 1000, 30)[0]

	scores := ix.Scores(q)
	// Reference top-k: every positive doc, sorted by (−score, +id).
	type sc struct {
		d int
		s float64
	}
	var ref []sc
	for d, s := range scores {
		if s > 0 {
			ref = append(ref, sc{d, s})
		}
	}
	sort.Slice(ref, func(i, j int) bool {
		if ref[i].s != ref[j].s {
			return ref[i].s > ref[j].s
		}
		return ref[i].d < ref[j].d
	})

	const k = 10
	got := ix.Query(q, k)
	if len(got) != k {
		t.Fatalf("Query returned %d hits, want %d", len(got), k)
	}
	for i, h := range got {
		if h.Index != ref[i].d || h.Score != ref[i].s {
			t.Errorf("rank %d: got {doc %d, %.6f}, want {doc %d, %.6f}", i, h.Index, h.Score, ref[i].d, ref[i].s)
		}
	}
}

func TestQuery_tieBreakByID(t *testing.T) {
	// Two docs with identical single-term weight → identical score → must come
	// back in ascending-id order.
	docs := []SparseVec{
		{Terms: []uint32{3}, Weights: []float32{1}}, // doc 0
		{Terms: []uint32{9}, Weights: []float32{0}}, // doc 1: zero weight, never scores
		{Terms: []uint32{3}, Weights: []float32{1}}, // doc 2: ties doc 0
	}
	ix := New(docs)
	got := ix.Query(SparseVec{Terms: []uint32{3}, Weights: []float32{1}}, 10)
	if len(got) != 2 || got[0].Index != 0 || got[1].Index != 2 {
		t.Fatalf("tie order: got %+v, want docs [0 2] ascending", got)
	}
}

func TestQuery_kSemantics(t *testing.T) {
	docs := []SparseVec{
		{Terms: []uint32{1}, Weights: []float32{3}},
		{Terms: []uint32{1}, Weights: []float32{2}},
		{Terms: []uint32{1}, Weights: []float32{1}},
		{Terms: []uint32{2}, Weights: []float32{5}}, // term 2: not in query → score 0
	}
	ix := New(docs)
	q := SparseVec{Terms: []uint32{1}, Weights: []float32{1}}

	if got := ix.Query(q, 0); len(got) != 0 {
		t.Errorf("k=0: got %d hits, want 0", len(got))
	}
	if got := ix.Query(q, -1); len(got) != 3 { // every positive doc
		t.Errorf("k<0: got %d hits, want 3 (positives only, doc 3 excluded)", len(got))
	}
	if got := ix.Query(q, 100); len(got) != 3 { // k>positives clamps
		t.Errorf("k>positives: got %d hits, want 3", len(got))
	}
	got := ix.Query(q, 2)
	if len(got) != 2 || got[0].Index != 0 || got[1].Index != 1 {
		t.Errorf("k=2: got %+v, want docs [0 1]", got)
	}
}

func TestNew_dupQuerySumsAndLenMismatchSafe(t *testing.T) {
	ix := New([]SparseVec{{Terms: []uint32{4}, Weights: []float32{1}}})
	// Duplicate query term 4 (0.5 + 0.5) and a trailing term with no weight (the
	// shorter slice bounds the walk) must not panic and must sum to 1.0·1.0.
	q := SparseVec{Terms: []uint32{4, 4, 7}, Weights: []float32{0.5, 0.5}}
	got := ix.Query(q, 1)
	if len(got) != 1 || math.Abs(got[0].Score-1.0) > 1e-6 {
		t.Fatalf("dup/len-mismatch: got %+v, want doc 0 score ~1.0", got)
	}
}
