// Fused int4(weight)×int8(activation) GEMV dot using AVX2 — the amd64 W4A8
// decode kernel's hot loop, counterpart of dot_w4a8_arm64.s. For ONE output it
// streams a whole K-wide weight row and returns the per-group-scaled f32 dot
// Σ_g scale[g]·(act·w)_g (the activation scale is applied by the Go caller).
//
// The nibble-unpack prologue (16 packed bytes → 32 centered int8 weights:
// low nibble = even k, high = odd k, −8) feeds the proven dotI8AVX2 sign-extend
// body (VPMOVSXBW + VPMADDWD). The KEY over the older per-group variant: the
// f32 weight scale is folded IN-REGISTER. Each group's 8 int32 lane-partials are
// converted to f32, multiplied by the group's broadcast scale, and accumulated
// into an 8-lane f32 accumulator — WITHOUT a per-group horizontal reduce. Because
// every lane of a group carries the same scale[g], one final reduce of the
// accumulator yields Σ_g scale[g]·Σ_lane(partials) = Σ_g scale[g]·groupdot[g].
// This removes the 63/64 per-group reductions, the SIMD→GP moves, the int32
// scratch round-trip, and the Go-side fold loop the old kernel paid — the
// overhead that made W4A8 ~2× slower than W8A8 despite reading half the bytes.
//
// group is fixed at 32 (16 packed bytes / 32 activations per iteration); the Go
// caller routes other group sizes and any ragged tail (K % 32) to the scalar
// reference. nGroups = K/32. AVX2 baseline (no VNNI); only called when hasAVX2.

#include "textflag.h"

DATA mask0F<>+0(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA mask0F<>+8(SB)/8, $0x0F0F0F0F0F0F0F0F
GLOBL mask0F<>(SB), RODATA|NOPTR, $16

DATA const8<>+0(SB)/8, $0x0808080808080808
DATA const8<>+8(SB)/8, $0x0808080808080808
GLOBL const8<>(SB), RODATA|NOPTR, $16

// func dotW4A8FoldAVX2(act *int8, packed *byte, scales *float32, nGroups int) float32
TEXT ·dotW4A8FoldAVX2(SB), NOSPLIT, $0-36
	MOVQ act+0(FP), SI     // &act[0]    (int8, 32 per group)
	MOVQ packed+8(FP), DI  // &packed[0] (16 bytes per group)
	MOVQ scales+16(FP), BX // &scales[0] (f32, one per group)
	MOVQ nGroups+24(FP), CX

	LEAQ    mask0F<>(SB), AX
	VMOVDQU (AX), X14       // low-nibble mask (hoisted)
	LEAQ    const8<>(SB), AX
	VMOVDQU (AX), X15       // bias 8 (hoisted)
	VXORPS  Y10, Y10, Y10   // f32 accumulator (8 lanes) = 0

loop:
	// Unpack 16 packed bytes → w0..w15 (X3) and w16..w31 (X4), centered −8.
	VMOVDQU    (DI), X0
	VPAND      X14, X0, X1       // low nibbles  (w0,w2,…,w30)
	VPSRLW     $4, X0, X2
	VPAND      X14, X2, X2       // high nibbles (w1,w3,…,w31)
	VPUNPCKLBW X2, X1, X3        // [w0,w1,…,w15]
	VPUNPCKHBW X2, X1, X4        // [w16,…,w31]
	VPSUBB     X15, X3, X3       // centered: nibble − 8 (signed int8)
	VPSUBB     X15, X4, X4
	VPMOVSXBW  X3, Y3
	VPMOVSXBW  X4, Y4

	// 32 int8 activations, sign-extended.
	VMOVDQU   (SI), X5
	VMOVDQU   16(SI), X6
	VPMOVSXBW X5, Y5
	VPMOVSXBW X6, Y6

	// Pairwise multiply-add → 8 int32 group partials (UNREDUCED).
	VPMADDWD Y5, Y3, Y7
	VPMADDWD Y6, Y4, Y8
	VPADDD   Y8, Y7, Y7

	// In-register f32 fold: convert lanes, broadcast scale[g], FMA into the
	// accumulator. No per-group horizontal reduce — each lane carries scale[g].
	VCVTDQ2PS    Y7, Y9
	VBROADCASTSS (BX), Y11
	VFMADD231PS  Y11, Y9, Y10     // Y10 += Y9 · scale[g]

	ADDQ $16, DI
	ADDQ $32, SI
	ADDQ $4, BX
	SUBQ $1, CX
	JNZ  loop

	// One horizontal f32 reduce of the 8-lane accumulator → return value.
	VEXTRACTF128 $1, Y10, X11
	VADDPS       X11, X10, X10
	VHADDPS      X10, X10, X10
	VHADDPS      X10, X10, X10
	MOVSS        X10, ret+32(FP)
	VZEROUPPER
	RET
