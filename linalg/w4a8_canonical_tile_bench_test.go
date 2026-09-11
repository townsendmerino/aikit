package linalg

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// BenchmarkMatmulBTW4A8Canonical exercises the CANONICAL-layout W4A8 span —
// the path taken by paged MoE experts, any tensor that declined the row4
// repack (K%32, N%4), MatmulBTW4A8Batch at M>1 and the uniform
// WeightMat.MatmulBTInto. Audit M-02: arm64 had no tile here, so each output
// was one M=1 GEMV per (row, column) pair.
//
// M=1 is the control — the tile declines below M=4, so it must not move.
func BenchmarkMatmulBTW4A8Canonical(b *testing.B) {
	const K, N = 1536, 2048
	rng := rand.New(rand.NewPCG(31, 37))
	w := make([]float32, N*K)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	packed, wScales := QuantizeGroupsInt4(w, N, K, 32)
	for _, M := range []int{1, 4, 8, 32, 128} {
		a := make([]float32, M*K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		dst := make([]float32, M*N)
		b.Run(fmt.Sprintf("M%d", M), func(b *testing.B) {
			var ws Workspace
			b.ResetTimer()
			for b.Loop() {
				MatmulBTW4A8Into(&ws, a, packed, wScales, dst, M, K, N, 32)
			}
			macs := int64(M) * int64(K) * int64(N)
			b.ReportMetric(float64(macs)/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "GMAC/s")
		})
	}
}
