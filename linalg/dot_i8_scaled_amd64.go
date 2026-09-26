package linalg

//go:noescape
func dotI8Scaled32AVX2(a, b *int8, aS *float32, nGroups int) float32

//go:noescape
func dotI8Scaled32x2AVX2(a, b0, b1 *int8, aS *float32, nGroups int) (float32, float32)

// dotI8Scaled32 is sum_g aS[g] * sum_{k in group g} a[k]*b[k] over groups of 32; len(a) must be a
// multiple of 32. The W8A8 dot under per-32 activation scales.
func dotI8Scaled32(a, b []int8, aS []float32) float32 {
	nG := len(a) / 32
	if nG == 0 {
		return 0
	}
	if hasAVX2 {
		return dotI8Scaled32AVX2(&a[0], &b[0], &aS[0], nG)
	}
	return dotI8Scaled32Go(a, b, aS)
}

// w8a8GroupedSpan computes output columns [n0, n1) across M rows with per-32 activation scales.
func w8a8GroupedSpan(aq []int8, aS []float32, bQ []int8, bScales, dst []float32, M, K, N, nG, n0, n1 int) {
	if hasAVX2 && nG > 0 {
		for m := range M {
			aPtr := &aq[m*K]
			asPtr := &aS[m*nG]
			dstRow := dst[m*N:]
			n := n0
			for ; n+1 < n1; n += 2 {
				b0 := &bQ[n*K]
				b1 := &bQ[(n+1)*K]
				d0, d1 := dotI8Scaled32x2AVX2(aPtr, b0, b1, asPtr, nG)
				dstRow[n] = d0 * bScales[n]
				dstRow[n+1] = d1 * bScales[n+1]
			}
			if n < n1 {
				w := bQ[n*K : (n+1)*K]
				dstRow[n] = dotI8Scaled32(aq[m*K:m*K+K], w, aS[m*nG:m*nG+nG]) * bScales[n]
			}
		}
		return
	}
	for n := n0; n < n1; n++ {
		w := bQ[n*K : n*K+K]
		for m := range M {
			dst[m*N+n] = dotI8Scaled32(aq[m*K:m*K+K], w, aS[m*nG:m*nG+nG]) * bScales[n]
		}
	}
}
