//go:build !amd64

package linalg

// dotI8x8 has no multi-row int8 kernel on this architecture; dotI8x8Generic
// still gets each row's own SIMD (SDOT on arm64), just not the shared-widening
// win dotI8x8AVX2 buys on amd64. audit M-21.
func dotI8x8(a, b0, b1, b2, b3, b4, b5, b6, b7 []int8, out *[8]int32) {
	dotI8x8Generic(a, b0, b1, b2, b3, b4, b5, b6, b7, out)
}
