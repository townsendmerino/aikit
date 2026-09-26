//go:build !amd64

package linalg

// dotI8Scaled32 is sum_g aS[g] * sum_{k in group g} a[k]*b[k] over groups of 32; len(a) must be a
// multiple of 32. The W8A8 dot under per-32 activation scales (portable form; no SIMD kernel on this
// architecture yet).
func dotI8Scaled32(a, b []int8, aS []float32) float32 { return dotI8Scaled32Go(a, b, aS) }
