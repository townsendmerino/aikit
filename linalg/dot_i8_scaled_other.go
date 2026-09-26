//go:build !amd64

package linalg

// dotI8Scaled32 is sum_g aS[g] * sum_{k in group g} a[k]*b[k] over groups of 32; len(a) must be a
// multiple of 32. The W8A8 dot under per-32 activation scales (portable form; no SIMD kernel on this
// architecture yet).
func dotI8Scaled32(a, b []int8, aS []float32) float32 { return dotI8Scaled32Go(a, b, aS) }

// w8a8GroupedSpan computes output columns [n0, n1) across M rows with per-32 activation scales.
func w8a8GroupedSpan(aq []int8, aS []float32, bQ []int8, bScales, dst []float32, M, K, N, nG, n0, n1 int) {
	for n := n0; n < n1; n++ {
		w := bQ[n*K : n*K+K]
		for m := range M {
			dst[m*N+n] = dotI8Scaled32(aq[m*K:m*K+K], w, aS[m*nG:m*nG+nG]) * bScales[n]
		}
	}
}
