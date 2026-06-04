//go:build arm64

package linalg

// dotI8NEON returns Σ a[i]*b[i] as int32 over the first n int8 elements (n a
// multiple of 16), via NEON SMULL/SADALP (base ARMv8 — no DotProd dependency).
// Implemented in dot_i8_arm64.s.
//
//go:noescape
func dotI8NEON(a, b *int8, n int) int32

// dotI8 is the int8×int8→int32 inner product for the W8A8 matmul: the 16-multiple
// bulk runs in the NEON kernel, the remainder falls to the scalar reference.
// Output is identical to dotI8Scalar (integer arithmetic — exact).
func dotI8(a, b []int8) int32 {
	n := len(a)
	if n < 16 {
		return dotI8Scalar(a, b)
	}
	n16 := n &^ 15
	s := dotI8NEON(&a[0], &b[0], n16)
	for k := n16; k < n; k++ {
		s += int32(a[k]) * int32(b[k])
	}
	return s
}
