package linalg

//go:noescape
func dotI8Scaled32AVX2(a, b *int8, aS *float32, nGroups int) float32

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
