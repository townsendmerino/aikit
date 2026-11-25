//go:build amd64

package linalg

import "github.com/townsendmerino/aikit/internal/cpufeat"

// amd64 kernel path. The three arch-neutral kernel names dispatch at
// runtime: AVX2+FMA when the CPU supports it (dotFMA in dot_amd64.s),
// otherwise the scalar kernels in dot_generic.go. Detection is done
// once at package init via hand-rolled CPUID/XGETBV (no x/sys dep —
// matching the repo's minimal-deps, hand-written-asm ethos).
//
// Kernel design: the multi-row 4x4/8x4 variants use register-blocked
// asm (dotFMA4/dotFMA8) that loads each 8-float chunk of the shared `a`
// row ONCE and runs 4 or 8 FMA chains against the b-rows — the a-reuse
// trick arm64's NEON kernel uses (its "M8d" path). dotFMA8/4 write each
// row's full dot product to out[r]; the Go wrappers scatter those into
// the [N*4] lane layout the matmul caller (linalg.go) horizontally sums,
// matching dot_generic.go / arm64 (full sum in the first lane of each
// 4-lane block, rest zero). When the CPU lacks AVX2 the wrappers fall
// back to the scalar kernels in dot_generic.go.

// dotFMA returns Σ a[i]*b[i] over the first n elements using AVX2+FMA
// (single row). Implemented in dot_amd64.s. Used by the single-vector
// dotNEON path.
//
//go:noescape
func dotFMA(a *float32, b *float32, n int) float32

// dotFMA4 / dotFMA8 compute 4 / 8 dot products a·b_r over the first n
// elements (n a multiple of 4), writing each full sum to out[r].
// Implemented in dot_amd64.s.
//
//go:noescape
func dotFMA4(a, b0, b1, b2, b3 *float32, n int, out *[4]float32)

//go:noescape
func dotFMA8(a, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n int, out *[8]float32)

// dotI8AVX2 returns Σ a[i]*b[i] as int32 over the first n int8 elements (n a
// multiple of 16), via VPMOVSXBW+VPMADDWD+VPADDD. Implemented in dot_amd64.s;
// used by dotI8 (quant_i8_amd64.go) for the W8A8 matmul.
//
//go:noescape
func dotI8AVX2(a, b *int8, n int) int32

func dotNEON(a *float32, b *float32, n4 int) float32 {
	if !cpufeat.HasAVX2 {
		return dotGeneric(a, b, n4)
	}
	return dotFMA(a, b, n4*4)
}

func dotNEON4x4(a, b0, b1, b2, b3 *float32, n4 int, sums *[16]float32) {
	if !cpufeat.HasAVX2 {
		dot4x4Generic(a, b0, b1, b2, b3, n4, sums)
		return
	}
	var out [4]float32
	dotFMA4(a, b0, b1, b2, b3, n4*4, &out)
	*sums = [16]float32{}
	sums[0], sums[4], sums[8], sums[12] = out[0], out[1], out[2], out[3]
}

func dotNEON8x4(a, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n4 int, sums *[32]float32) {
	if !cpufeat.HasAVX2 {
		dot8x4Generic(a, b0, b1, b2, b3, b4, b5, b6, b7, n4, sums)
		return
	}
	var out [8]float32
	dotFMA8(a, b0, b1, b2, b3, b4, b5, b6, b7, n4*4, &out)
	*sums = [32]float32{}
	for r := range 8 {
		sums[r*4] = out[r]
	}
}
