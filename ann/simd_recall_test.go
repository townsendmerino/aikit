package ann

import (
	"math/rand/v2"
	"sort"
	"testing"
)

// TestFlat_recallVsFloat64 validates the float64-scalar → linalg.Dot (float32
// SIMD) swap: recall@k must be unchanged. The new top-k must contain every result
// an exact float64 brute-force scan returns, EXCEPT items that tie within float32
// precision at the k-th boundary — tie-order may flip, recall may not. This is
// the guarantee the Flat doc now advertises.
func TestFlat_recallVsFloat64(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 99))
	const N, d, k, queries = 4000, 768, 10, 40
	corpus := make([][]float32, N)
	for i := range corpus {
		corpus[i] = randUnit(rng, d)
	}
	f := New(corpus)

	const tieEps = 1e-5 // > float32 dot accumulation error over d=768 unit terms
	flips := 0
	for qi := range queries {
		q := randUnit(rng, d)
		got := f.Query(q, k) // new float32 SIMD path

		// Reference: exact float64 brute-force ranking.
		type sc struct {
			idx int
			s   float64
		}
		ref := make([]sc, N)
		for i, v := range corpus {
			var dot float64
			for j := range v {
				dot += float64(v[j]) * float64(q[j])
			}
			ref[i] = sc{i, dot}
		}
		sort.Slice(ref, func(a, b int) bool {
			if ref[a].s != ref[b].s {
				return ref[a].s > ref[b].s
			}
			return ref[a].idx < ref[b].idx
		})

		inNew := make(map[int]bool, k)
		for _, h := range got {
			inNew[h.Index] = true
		}
		kthNew := got[len(got)-1].Score // smallest score retained by the new path

		for r := range k {
			if inNew[ref[r].idx] {
				continue
			}
			// A ref top-k item is missing from the new top-k. Acceptable ONLY if
			// it ties the new k-th score within float32 precision (a flip, not a
			// recall loss). A clear gap means a real bug.
			if ref[r].s-kthNew > tieEps {
				t.Errorf("query %d: ref rank %d item %d (f64 score %.8f) absent from new top-k (kth=%.8f) — recall loss, not a tie",
					qi, r, ref[r].idx, ref[r].s, kthNew)
			}
			flips++
		}
	}
	t.Logf("%d queries × top-%d on N=%d d=%d: %d boundary tie-flips, all within float32 ε (recall unchanged)", queries, k, N, d, flips)
}
