// Fused int4(weight)×int8(activation) per-group dot using AVX2 — the amd64
// W4A8 decode kernel's hot loop, the counterpart of dot_w4a8_arm64.s (SDOT).
// For ONE output it streams a whole K-wide weight row, emitting one int32 dot
// per 32-wide group; the Go caller folds in the per-group f32 scales.
//
// As on arm64, the ONLY new code over the proven int8 dot (dotI8AVX2) is the
// nibble-unpack prologue: 16 packed bytes → 32 centered int8 weights in-register
// (low nibble = even k, high = odd k, −8). The sign-extend (VPMOVSXBW) +
// VPMADDWD + VPADDD reduction below mirror dotI8AVX2 exactly — fully signed, no
// unsigned-offset correction, because the weights are centered to int8 first.
//
// No VNNI is required: this is the AVX2 baseline (validated on a Zen 2 box,
// which has no VPDPBUSD). A VNNI variant (VPDPBUSD) is a future drop-in behind
// the same CPUID gate. group is fixed at 32 (16 packed bytes / 32 activations
// per iteration); the Go caller routes other group sizes and any ragged tail
// (K % 32) to the scalar reference. nGroups = K/32. Only called when hasAVX2.

#include "textflag.h"

DATA mask0F<>+0(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA mask0F<>+8(SB)/8, $0x0F0F0F0F0F0F0F0F
GLOBL mask0F<>(SB), RODATA|NOPTR, $16

DATA const8<>+0(SB)/8, $0x0808080808080808
DATA const8<>+8(SB)/8, $0x0808080808080808
GLOBL const8<>(SB), RODATA|NOPTR, $16

// func dotW4A8GroupsAVX2(act *int8, packed *byte, out *int32, nGroups int)
TEXT ·dotW4A8GroupsAVX2(SB), NOSPLIT, $0-32
	MOVQ act+0(FP), SI     // &act[0]    (int8, 32 per group)
	MOVQ packed+8(FP), DI  // &packed[0] (16 bytes per group)
	MOVQ out+16(FP), DX    // &out[0]    (int32, one per group)
	MOVQ nGroups+24(FP), CX

	LEAQ    mask0F<>(SB), AX
	VMOVDQU (AX), X14        // low-nibble mask (hoisted)
	LEAQ    const8<>(SB), AX
	VMOVDQU (AX), X15        // bias 8 (hoisted)

loop:
	// Unpack 16 packed bytes → w0..w15 (X3) and w16..w31 (X4), centered −8.
	VMOVDQU    (DI), X0          // 16 packed bytes = 32 nibbles
	VPAND      X14, X0, X1       // X1 = low nibbles  (w0,w2,…,w30)
	VPSRLW     $4, X0, X2        // per-16-bit shift…
	VPAND      X14, X2, X2       // …+ mask = high nibbles (w1,w3,…,w31)
	VPUNPCKLBW X2, X1, X3        // interleave → [w0,w1,…,w15]
	VPUNPCKHBW X2, X1, X4        // → [w16,…,w31]
	VPSUBB     X15, X3, X3       // centered: nibble − 8 (signed int8)
	VPSUBB     X15, X4, X4
	VPMOVSXBW  X3, Y3            // w0..w15  → 16 int16
	VPMOVSXBW  X4, Y4            // w16..w31 → 16 int16

	// 32 int8 activations, sign-extended to int16.
	VMOVDQU   (SI), X5
	VMOVDQU   16(SI), X6
	VPMOVSXBW X5, Y5            // act0..15
	VPMOVSXBW X6, Y6            // act16..31

	// Pairwise multiply-add to int32 and combine the two halves.
	VPMADDWD Y5, Y3, Y7        // 8 int32 = Σ pairs (w·act), k=0..15
	VPMADDWD Y6, Y4, Y8        // k=16..31
	VPADDD   Y8, Y7, Y7        // 8 int32 group partials

	// Horizontal int32 sum (same reduce as dotI8AVX2) → out[g].
	VEXTRACTI128 $1, Y7, X8
	VPADDD       X8, X7, X7
	VPHADDD      X7, X7, X7
	VPHADDD      X7, X7, X7
	MOVD         X7, AX
	MOVL         AX, (DX)

	ADDQ $16, DI
	ADDQ $32, SI
	ADDQ $4, DX
	SUBQ $1, CX
	JNZ  loop

	VZEROUPPER
	RET
