#include "textflag.h"

// func dotI8Scaled32AVX2(a, b *int8, aS *float32, nGroups int) float32
//
// sum over nGroups groups of 32 int8 pairs: aS[g] * sum_{k in g} a[k]*b[k]. Per group it is
// dotI8AVX2's own inner step (sign-extend, VPMADDWD, add) leaving 8 int32 lane sums (each <=
// 4*127*127*2, exact in f32), then VCVTDQ2PS, multiply by the broadcast group scale, and accumulate
// in f32 (fused multiply-add). Four groups per iteration into four independent accumulators (the
// ILP dotI8AVX2 itself has), so the per-group scale does not serialize a compute-bound small matmul;
// a one-group tail loop. Per-group activation scales for W8A8 (actgroup.go).
TEXT ·dotI8Scaled32AVX2(SB), NOSPLIT, $0-36
	MOVQ a+0(FP), SI
	MOVQ b+8(FP), DI
	MOVQ aS+16(FP), DX
	MOVQ nGroups+24(FP), CX
	VXORPS Y0, Y0, Y0 // four independent accumulators, groups 4i+0..3 — dotI8AVX2's own ILP
	VXORPS Y10, Y10, Y10
	VXORPS Y11, Y11, Y11
	VXORPS Y12, Y12, Y12
	CMPQ   CX, $4
	JLT    scaled_tail

scaled_loop4:
	VPMOVSXBW    0(SI), Y1
	VPMOVSXBW    0(DI), Y2
	VPMADDWD     Y2, Y1, Y3
	VPMOVSXBW    16(SI), Y1
	VPMOVSXBW    16(DI), Y2
	VPMADDWD     Y2, Y1, Y4
	VPADDD       Y4, Y3, Y3
	VCVTDQ2PS    Y3, Y3
	VBROADCASTSS 0(DX), Y5
	VFMADD231PS  Y5, Y3, Y0
	VPMOVSXBW    32(SI), Y1
	VPMOVSXBW    32(DI), Y2
	VPMADDWD     Y2, Y1, Y6
	VPMOVSXBW    48(SI), Y1
	VPMOVSXBW    48(DI), Y2
	VPMADDWD     Y2, Y1, Y7
	VPADDD       Y7, Y6, Y6
	VCVTDQ2PS    Y6, Y6
	VBROADCASTSS 4(DX), Y8
	VFMADD231PS  Y8, Y6, Y10
	VPMOVSXBW    64(SI), Y1
	VPMOVSXBW    64(DI), Y2
	VPMADDWD     Y2, Y1, Y3
	VPMOVSXBW    80(SI), Y1
	VPMOVSXBW    80(DI), Y2
	VPMADDWD     Y2, Y1, Y4
	VPADDD       Y4, Y3, Y3
	VCVTDQ2PS    Y3, Y3
	VBROADCASTSS 8(DX), Y5
	VFMADD231PS  Y5, Y3, Y11
	VPMOVSXBW    96(SI), Y1
	VPMOVSXBW    96(DI), Y2
	VPMADDWD     Y2, Y1, Y6
	VPMOVSXBW    112(SI), Y1
	VPMOVSXBW    112(DI), Y2
	VPMADDWD     Y2, Y1, Y7
	VPADDD       Y7, Y6, Y6
	VCVTDQ2PS    Y6, Y6
	VBROADCASTSS 12(DX), Y8
	VFMADD231PS  Y8, Y6, Y12
	ADDQ $128, SI
	ADDQ $128, DI
	ADDQ $16, DX
	SUBQ $4, CX
	CMPQ CX, $4
	JGE  scaled_loop4

scaled_tail:
	TESTQ CX, CX
	JZ    scaled_done
	VPMOVSXBW    0(SI), Y1
	VPMOVSXBW    0(DI), Y2
	VPMADDWD     Y2, Y1, Y3
	VPMOVSXBW    16(SI), Y1
	VPMOVSXBW    16(DI), Y2
	VPMADDWD     Y2, Y1, Y4
	VPADDD       Y4, Y3, Y3
	VCVTDQ2PS    Y3, Y3
	VBROADCASTSS 0(DX), Y5
	VFMADD231PS  Y5, Y3, Y0
	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $4, DX
	DECQ CX
	JMP  scaled_tail

scaled_done:
	VADDPS       Y10, Y0, Y0
	VADDPS       Y12, Y11, Y11
	VADDPS       Y11, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VZEROUPPER
	MOVSS        X0, ret+32(FP)
	RET
