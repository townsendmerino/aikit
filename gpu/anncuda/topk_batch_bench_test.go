//go:build linux

package anncuda

import (
	"fmt"
	"math/rand"
	"testing"
)

// BenchmarkTopKBatch measures the device top-k path at the small batch sizes
// audit M-17 is about: at M <= 8 the per-call executor round-trips were roughly
// half of the recorded single-query time, so the fixed cost dominated the work.
//
// anncuda had no benchmark at all, which is why that was counted from the call
// sequence rather than measured.
func BenchmarkTopKBatch(b *testing.B) {
	const dim, k = 256, 10
	rng := rand.New(rand.NewSource(77))
	for _, n := range []int{200_000, 500_000} {
		vecs := make([][]float32, n)
		for i := range vecs {
			v := make([]float32, dim)
			for j := range v {
				v[j] = float32(rng.NormFloat64())
			}
			vecs[i] = v
		}
		idx := newTestIndex(b, vecs)
		for _, M := range []int{1, 4, 8} {
			queries := make([][]float32, M)
			for m := range queries {
				queries[m] = vecs[m]
			}
			b.Run(fmt.Sprintf("N%d/M%d", n, M), func(b *testing.B) {
				for b.Loop() {
					if _, err := idx.TopKBatch(queries, k); err != nil {
						b.Fatalf("TopKBatch: %v", err)
					}
				}
			})
		}
	}
}
