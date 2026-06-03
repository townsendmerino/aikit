//go:build amd64

package linalg

// dotI8 is the SIMD-dispatched int8×int8→int32 inner product used by the W8A8
// matmul. On an AVX2 CPU the 16-multiple bulk runs in dotI8AVX2 (dot_amd64.s)
// and the remainder falls to the scalar reference; without AVX2 it is all
// scalar. Output is identical to dotI8Scalar (integer arithmetic — exact, no
// reassociation).
func dotI8(a, b []int8) int32 {
	n := len(a)
	if !hasAVX2 || n < 16 {
		return dotI8Scalar(a, b)
	}
	n16 := n &^ 15
	s := dotI8AVX2(&a[0], &b[0], n16)
	for k := n16; k < n; k++ {
		s += int32(a[k]) * int32(b[k])
	}
	return s
}
