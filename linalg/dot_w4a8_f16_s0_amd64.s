// S0 prototype (goinfer docs/tasks/task-cpu-decode-peer-gap-2026-09.md, L1 S0): dotW4A8FoldAVX2 with
// the per-group weight scale stored as IEEE half precision. The ONLY change from dotW4A8FoldAVX2
// (dot_w4a8_amd64.s) is the scale load: VPBROADCASTW + VCVTPH2PS instead of VBROADCASTSS, and the
// scale pointer advancing 2 bytes per group instead of 4. f16→f32 is exact, so given scales that are
// f16-representable this returns bit-for-bit what dotW4A8FoldAVX2 returns on their f32 values.
// Needs AVX2 + F16C.

#include "textflag.h"

DATA s0mask0F<>+0(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA s0mask0F<>+8(SB)/8, $0x0F0F0F0F0F0F0F0F
GLOBL s0mask0F<>(SB), RODATA|NOPTR, $16

DATA s0const8<>+0(SB)/8, $0x0808080808080808
DATA s0const8<>+8(SB)/8, $0x0808080808080808
GLOBL s0const8<>(SB), RODATA|NOPTR, $16

// func dotW4A8FoldF16AVX2(act *int8, packed *byte, scales *uint16, nGroups int) float32
TEXT ·dotW4A8FoldF16AVX2(SB), NOSPLIT, $0-36
	MOVQ act+0(FP), SI
	MOVQ packed+8(FP), DI
	MOVQ scales+16(FP), BX // &scales[0] (f16, one per group)
	MOVQ nGroups+24(FP), CX

	LEAQ    s0mask0F<>(SB), AX
	VMOVDQU (AX), X14
	LEAQ    s0const8<>(SB), AX
	VMOVDQU (AX), X15
	VXORPS  Y10, Y10, Y10

loop:
	VMOVDQU    (DI), X0
	VPAND      X14, X0, X1
	VPSRLW     $4, X0, X2
	VPAND      X14, X2, X2
	VPUNPCKLBW X2, X1, X3
	VPUNPCKHBW X2, X1, X4
	VPSUBB     X15, X3, X3
	VPSUBB     X15, X4, X4
	VPMOVSXBW  X3, Y3
	VPMOVSXBW  X4, Y4

	VMOVDQU   (SI), X5
	VMOVDQU   16(SI), X6
	VPMOVSXBW X5, Y5
	VPMOVSXBW X6, Y6

	VPMADDWD Y5, Y3, Y7
	VPMADDWD Y6, Y4, Y8
	VPADDD   Y8, Y7, Y7

	VCVTDQ2PS    Y7, Y9
	VPBROADCASTW (BX), X11   // scale[g] (f16) in all 8 halves
	VCVTPH2PS    X11, Y11    // → 8 × f32 scale[g] (exact)
	VFMADD231PS  Y11, Y9, Y10

	ADDQ $16, DI
	ADDQ $32, SI
	ADDQ $2, BX
	SUBQ $1, CX
	JNZ  loop

	VEXTRACTF128 $1, Y10, X11
	VADDPS       X11, X10, X10
	VHADDPS      X10, X10, X10
	VHADDPS      X10, X10, X10
	MOVSS        X10, ret+32(FP)
	VZEROUPPER
	RET
