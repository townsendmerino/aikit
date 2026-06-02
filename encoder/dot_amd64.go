//go:build amd64

package encoder

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

// cpuid executes CPUID with the given leaf/subleaf. Implemented in
// dot_amd64.s.
//
//go:noescape
func cpuid(eaxIn, ecxIn uint32) (eax, ebx, ecx, edx uint32)

// xgetbv reads XCR0 (the OS-enabled extended-state mask). Implemented
// in dot_amd64.s.
//
//go:noescape
func xgetbv() (eax, edx uint32)

// hasAVX2 is true when the CPU and OS both support AVX2 + FMA + YMM
// state save. Computed once at init; the kernels branch on it.
var hasAVX2 = detectAVX2()

func detectAVX2() bool {
	maxLeaf, _, _, _ := cpuid(0, 0)
	if maxLeaf < 7 {
		return false
	}
	// Leaf 1: ECX feature bits. Need OSXSAVE (27), AVX (28), FMA (12).
	const (
		bitFMA     = 1 << 12
		bitOSXSAVE = 1 << 27
		bitAVX     = 1 << 28
	)
	_, _, ecx1, _ := cpuid(1, 0)
	if ecx1&(bitFMA|bitOSXSAVE|bitAVX) != (bitFMA | bitOSXSAVE | bitAVX) {
		return false
	}
	// OSXSAVE set ⇒ XGETBV is usable. XCR0 bits 1 (SSE/XMM) and 2 (AVX/
	// YMM) must both be set for the OS to preserve YMM across context
	// switches; without that, executing AVX would corrupt state.
	xcr0, _ := xgetbv()
	const xcr0YMM = 1<<1 | 1<<2
	if xcr0&xcr0YMM != xcr0YMM {
		return false
	}
	// Leaf 7, subleaf 0: EBX bit 5 = AVX2.
	_, ebx7, _, _ := cpuid(7, 0)
	const bitAVX2 = 1 << 5
	return ebx7&bitAVX2 != 0
}

func dotNEON(a *float32, b *float32, n4 int) float32 {
	if !hasAVX2 {
		return dotGeneric(a, b, n4)
	}
	return dotFMA(a, b, n4*4)
}

func dotNEON4x4(a, b0, b1, b2, b3 *float32, n4 int, sums *[16]float32) {
	if !hasAVX2 {
		dot4x4Generic(a, b0, b1, b2, b3, n4, sums)
		return
	}
	var out [4]float32
	dotFMA4(a, b0, b1, b2, b3, n4*4, &out)
	*sums = [16]float32{}
	sums[0], sums[4], sums[8], sums[12] = out[0], out[1], out[2], out[3]
}

func dotNEON8x4(a, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n4 int, sums *[32]float32) {
	if !hasAVX2 {
		dot8x4Generic(a, b0, b1, b2, b3, b4, b5, b6, b7, n4, sums)
		return
	}
	var out [8]float32
	dotFMA8(a, b0, b1, b2, b3, b4, b5, b6, b7, n4*4, &out)
	*sums = [32]float32{}
	for r := 0; r < 8; r++ {
		sums[r*4] = out[r]
	}
}
