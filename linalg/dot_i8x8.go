package linalg

// dotI8x8Generic computes eight int8 dot products of a shared a-row against
// eight b-rows the plain way — one dotI8 call per row, no shared widening.
// Unlike the retired dotI8Cols8Generic (a raw scalar loop), this calls dotI8
// itself, so it is not actually unaccelerated where dotI8 has a kernel of its
// own (SDOT on arm64) — only the WIDENING SHARE across the eight rows that
// dotI8x8AVX2 buys is missing here, not the per-row SIMD.
func dotI8x8Generic(a, b0, b1, b2, b3, b4, b5, b6, b7 []int8, out *[8]int32) {
	bs := [8][]int8{b0, b1, b2, b3, b4, b5, b6, b7}
	for c := range bs {
		out[c] = dotI8(a, bs[c])
	}
}
