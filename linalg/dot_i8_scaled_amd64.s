#include "textflag.h"

// func dotI8Scaled32AVX2(a, b *int8, aS *float32, nGroups int) float32
//
// sum over nGroups groups of 32 int8 pairs: aS[g] * sum_{k in g} a[k]*b[k]. Per group it is
// dotI8AVX2's own inner step (sign-extend, VPMADDWD, add) leaving 8 int32 lane sums (each <=
// 4*127*127*2, exact in f32), then VCVTDQ2PS, multiply by the broadcast group scale, and accumulate
// in f32 (fused multiply-add). Two groups per iteration into two independent accumulators, so the
// per-group scale does not serialize a compute-bound small matmul; odd tail single. Per-group
// activation scales for W8A8 (actgroup.go).
TEXT ·dotI8Scaled32AVX2(SB), NOSPLIT, $0-36
	MOVQ a+0(FP), SI
	MOVQ b+8(FP), DI
	MOVQ aS+16(FP), DX
	MOVQ nGroups+24(FP), CX
	VXORPS Y0, Y0, Y0 // accumulator, even groups
	VXORPS Y6, Y6, Y6 // accumulator, odd groups — two independent chains
	CMPQ   CX, $2
	JLT    scaled_tail

scaled_loop2:
	VPMOVSXBW    0(SI), Y1
	VPMOVSXBW    0(DI), Y2
	VPMADDWD     Y2, Y1, Y3
	VPMOVSXBW    16(SI), Y1
	VPMOVSXBW    16(DI), Y2
	VPMADDWD     Y2, Y1, Y4
	VPADDD       Y4, Y3, Y3
	VCVTDQ2PS    Y3, Y3
	VBROADCASTSS (DX), Y5
	VFMADD231PS  Y5, Y3, Y0

	VPMOVSXBW    32(SI), Y1
	VPMOVSXBW    32(DI), Y2
	VPMADDWD     Y2, Y1, Y7
	VPMOVSXBW    48(SI), Y1
	VPMOVSXBW    48(DI), Y2
	VPMADDWD     Y2, Y1, Y8
	VPADDD       Y8, Y7, Y7
	VCVTDQ2PS    Y7, Y7
	VBROADCASTSS 4(DX), Y9
	VFMADD231PS  Y9, Y7, Y6

	ADDQ $64, SI
	ADDQ $64, DI
	ADDQ $8, DX
	SUBQ $2, CX
	CMPQ CX, $2
	JGE  scaled_loop2

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
	VBROADCASTSS (DX), Y5
	VFMADD231PS  Y5, Y3, Y0

scaled_done:
	VADDPS       Y6, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VZEROUPPER
	MOVSS        X0, ret+32(FP)
	RET
