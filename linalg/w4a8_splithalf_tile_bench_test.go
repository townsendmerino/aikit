//go:build amd64

package linalg

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// BenchmarkMatmulBTW4A8SplitHalfTile is BenchmarkMatmulBTW4A8Canonical's
// split-half-only twin (audit M-22 follow-up: split-half M>1) — same K, N,
// and M sweep, so the two are directly comparable rows in a perfgate run.
// Routed through WeightMat.MatmulBTW4A8Into (the real dispatch entry, which
// branches M==1 -> the decode kernel, M>1 -> the new tile) rather than
// calling either kernel-level function directly, so this measures what a
// caller actually gets.
//
// M=1 is the control, same reason BenchmarkMatmulBTW4A8Canonical's is: the
// tile declines below M=4 and must not move.
func BenchmarkMatmulBTW4A8SplitHalfTile(b *testing.B) {
	if !hasAVX2 || hasAVX512VNNIVL {
		b.Skip("split-half tile requires AVX2 without AVX512VNNIVL on this core")
	}
	const K, N = 1536, 2048
	rng := rand.New(rand.NewPCG(31, 38))
	w := make([]float32, N*K)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	q4, q4s := QuantizeGroupsInt4(w, N, K, 32)
	wm, ok := RepackInt4SplitHalfInPlace(q4, q4s, N, K, 32)
	if !ok {
		b.Fatal("RepackInt4SplitHalfInPlace: ok=false despite hasAVX2 && !hasAVX512VNNIVL")
	}
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
				wm.MatmulBTW4A8Into(&ws, a, dst, M)
			}
			macs := int64(M) * int64(K) * int64(N)
			b.ReportMetric(float64(macs)/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "GMAC/s")
		})
	}
}
