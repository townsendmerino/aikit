//go:build !arm64 && !amd64

package encoder

// Catch-all kernel path: architectures with neither NEON asm (arm64)
// nor AVX2 asm (amd64) alias the arch-neutral interface straight to
// the scalar kernels in dot_generic.go. Same contract as the arm64
// asm: operates on n4 four-element strides, multi-row variants put
// each row's full sum in the first lane of its 4-lane block. dotF32
// in dot.go and the matmul micro-kernel in linalg.go stay arch-
// agnostic against these names.

func dotNEON(a *float32, b *float32, n4 int) float32 { return dotGeneric(a, b, n4) }

func dotNEON4x4(a, b0, b1, b2, b3 *float32, n4 int, sums *[16]float32) {
	dot4x4Generic(a, b0, b1, b2, b3, n4, sums)
}

func dotNEON8x4(a, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n4 int, sums *[32]float32) {
	dot8x4Generic(a, b0, b1, b2, b3, b4, b5, b6, b7, n4, sums)
}
