package ann

import (
	"math"
	"math/rand/v2"
	"testing"
)

// clusteredCorpus builds nClusters tight clusters of perCluster near-duplicate
// unit vectors — the structure that breaks plain M-nearest (Alg-3) HNSW, since a
// node links only to its near-clones and never reaches other clusters.
func clusteredCorpus(rng *rand.Rand, nClusters, perCluster, dim int) (vecs [][]float32, centers [][]float32) {
	centers = randUnitSet(rng, nClusters, dim)
	for _, c := range centers {
		for j := 0; j < perCluster; j++ {
			v := make([]float32, dim)
			var ss float64
			for d := range v {
				x := float64(c[d]) + 0.04*rng.NormFloat64()
				v[d] = float32(x)
				ss += x * x
			}
			inv := float32(1 / math.Sqrt(ss))
			for d := range v {
				v[d] *= inv
			}
			vecs = append(vecs, v)
		}
	}
	return vecs, centers
}

// TestHNSW_heuristicBeatsSimpleOnClustered is the §4.1 regression test: on
// clustered data the default diversity heuristic (Alg-4) must keep high recall,
// and must beat the plain M-nearest selection (Alg-3, Config.SimpleNeighbors) —
// the failure the bench harness exposed (Alg-3 capped ~0.68 on real code
// embeddings). Model-free, so it runs in CI.
func TestHNSW_heuristicBeatsSimpleOnClustered(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 9))
	const dim = 128
	vecs, centers := clusteredCorpus(rng, 40, 50, dim)
	f := New(vecs)

	qrng := rand.New(rand.NewPCG(5, 5))
	recall := func(simple bool) float64 {
		h := BuildHNSW(vecs, Config{M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1, SimpleNeighbors: simple})
		const Q, k = 80, 10
		var sum float64
		for i := 0; i < Q; i++ {
			// Query near a random cluster center so the true top-k is a tight
			// cluster — the intra/inter-cluster navigation Alg-3 fails at.
			c := centers[i%len(centers)]
			q := make([]float32, dim)
			var ss float64
			for d := range q {
				x := float64(c[d]) + 0.04*qrng.NormFloat64()
				q[d] = float32(x)
				ss += x * x
			}
			inv := float32(1 / math.Sqrt(ss))
			for d := range q {
				q[d] *= inv
			}
			truth := make(map[int]bool, k)
			for _, h := range f.Query(q, k) {
				truth[h.Index] = true
			}
			got := 0
			for _, h := range h.Query(q, k) {
				if truth[h.Index] {
					got++
				}
			}
			sum += float64(got) / float64(k)
		}
		return sum / Q
	}

	diverse := recall(false) // Alg-4, the default
	simple := recall(true)   // Alg-3, opted in via SimpleNeighbors
	t.Logf("clustered recall@10: Alg-4 (default) = %.4f, Alg-3 (SimpleNeighbors) = %.4f", diverse, simple)

	if diverse < 0.90 {
		t.Errorf("Alg-4 clustered recall@10 = %.4f, want ≥ 0.90 (heuristic regression)", diverse)
	}
	if diverse < simple {
		t.Errorf("Alg-4 (%.4f) should not be worse than Alg-3 (%.4f) on clustered data", diverse, simple)
	}
}
