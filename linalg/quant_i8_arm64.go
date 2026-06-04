//go:build arm64

package linalg

// dotI8NEON returns Σ a[i]*b[i] as int32 over the first n int8 elements (n a
// multiple of 16), via NEON SMULL/SADALP (base ARMv8 — no DotProd dependency).
// Implemented in dot_i8_arm64.s.
//
//go:noescape
func dotI8NEON(a, b *int8, n int) int32

// dotI8SDOT is the same dot, via the ARMv8.2 DotProd SDOT instruction (~4× fewer
// vector ops than the SMULL/SADALP path). Only safe to call on a DotProd-capable
// core — gated by hasDotProd. Implemented in dot_i8dp_arm64.s.
//
//go:noescape
func dotI8SDOT(a, b *int8, n int) int32

// hasDotProd records, once at init, whether this CPU implements the ARMv8.2
// DotProd extension — so dotI8 picks the SDOT kernel on hardware that has it
// (Apple Silicon, Graviton2+, Neoverse, recent Cortex-A) and falls back to the
// base SMULL/SADALP kernel everywhere else. See detectDotProd (per-OS).
var hasDotProd = detectDotProd()

// dotI8 is the int8×int8→int32 inner product for the W8A8 matmul: the 16-multiple
// bulk runs in the best NEON kernel this CPU supports (SDOT if present, else
// SMULL/SADALP), the remainder falls to the scalar reference. Output is identical
// to dotI8Scalar (integer arithmetic — exact) regardless of the path taken.
func dotI8(a, b []int8) int32 {
	n := len(a)
	if n < 16 {
		return dotI8Scalar(a, b)
	}
	n16 := n &^ 15
	var s int32
	if hasDotProd {
		s = dotI8SDOT(&a[0], &b[0], n16)
	} else {
		s = dotI8NEON(&a[0], &b[0], n16)
	}
	for k := n16; k < n; k++ {
		s += int32(a[k]) * int32(b[k])
	}
	return s
}
