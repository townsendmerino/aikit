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
	// Hoist all 4 scale loads to the start of the 4-group block
	// so memory load latency is completely hidden.
	VBROADCASTSS 0(DX), Y8
	VBROADCASTSS 4(DX), Y9
	VBROADCASTSS 8(DX), Y13
	VBROADCASTSS 12(DX), Y14

	// Group 0 integer dot (32 elements)
	VPMOVSXBW 0(SI), Y1
	VPMOVSXBW 0(DI), Y2
	VPMADDWD  Y2, Y1, Y3
	VPMOVSXBW 16(SI), Y1
	VPMOVSXBW 16(DI), Y2
	VPMADDWD  Y2, Y1, Y4

	// Group 1 integer dot (32 elements)
	VPMOVSXBW 32(SI), Y1
	VPMOVSXBW 32(DI), Y2
	VPMADDWD  Y2, Y1, Y5
	VPMOVSXBW 48(SI), Y1
	VPMOVSXBW 48(DI), Y2
	VPMADDWD  Y2, Y1, Y6

	// Group 0 lane-sum and convert to f32 (Y3 ready during group 1 dot)
	VPADDD    Y4, Y3, Y3
	VCVTDQ2PS Y3, Y3

	// Group 2 integer dot (32 elements)
	VPMOVSXBW 64(SI), Y1
	VPMOVSXBW 64(DI), Y2
	VPMADDWD  Y2, Y1, Y7
	VPMOVSXBW 80(SI), Y1
	VPMOVSXBW 80(DI), Y2
	VPMADDWD  Y2, Y1, Y4

	// Group 1 lane-sum and convert to f32 (Y5 ready during group 2 dot)
	VPADDD    Y6, Y5, Y5
	VCVTDQ2PS Y5, Y5

	// Group 0 FMA into acc0 (Y3 convert finished, Y8 preloaded)
	VFMADD231PS Y8, Y3, Y0

	// Group 3 integer dot (32 elements)
	VPMOVSXBW 96(SI), Y1
	VPMOVSXBW 96(DI), Y2
	VPMADDWD  Y2, Y1, Y15
	VPMOVSXBW 112(SI), Y1
	VPMOVSXBW 112(DI), Y2
	VPMADDWD  Y2, Y1, Y6

	// Group 2 lane-sum and convert to f32
	VPADDD    Y4, Y7, Y7
	VCVTDQ2PS Y7, Y7

	// Group 1 FMA into acc1 (Y5 convert finished, Y9 preloaded)
	VFMADD231PS Y9, Y5, Y10

	// Group 3 lane-sum and convert to f32
	VPADDD    Y6, Y15, Y15
	VCVTDQ2PS Y15, Y15

	// Group 2 FMA into acc2 (Y7 convert finished, Y13 preloaded)
	VFMADD231PS Y13, Y7, Y11

	// Group 3 FMA into acc3 (Y15 convert finished, Y14 preloaded)
	VFMADD231PS Y14, Y15, Y12

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

// func dotI8Scaled32x2AVX2(a, b0, b1 *int8, aS *float32, nGroups int) (float32, float32)
//
// 2-column W8A8 kernel: evaluates dot products against two weight rows b0 and b1 simultaneously.
// Reuses activation vector loads and broadcast scales across both columns, cutting memory traffic
// and saturating both FMA execution pipes on Zen 2.
TEXT ·dotI8Scaled32x2AVX2(SB), NOSPLIT, $0-48
	MOVQ a+0(FP), SI
	MOVQ b0+8(FP), DI
	MOVQ b1+16(FP), R8
	MOVQ aS+24(FP), DX
	MOVQ nGroups+32(FP), CX

	VXORPS Y0, Y0, Y0   // col 0 acc 0
	VXORPS Y9, Y9, Y9   // col 0 acc 1
	VXORPS Y8, Y8, Y8   // col 1 acc 0
	VXORPS Y10, Y10, Y10 // col 1 acc 1

	CMPQ CX, $4
	JLT  scaled2_rem

scaled2_loop4:
	// Groups 0 and 1
	VBROADCASTSS 0(DX), Y11
	VBROADCASTSS 4(DX), Y12

	VPMOVSXBW 0(SI), Y1
	VPMOVSXBW 0(DI), Y2
	VPMADDWD  Y2, Y1, Y4
	VPMOVSXBW 0(R8), Y3
	VPMADDWD  Y3, Y1, Y5

	VPMOVSXBW 16(SI), Y1
	VPMOVSXBW 16(DI), Y2
	VPMADDWD  Y2, Y1, Y6
	VPMOVSXBW 16(R8), Y3
	VPMADDWD  Y3, Y1, Y7

	VPADDD      Y6, Y4, Y4
	VPADDD      Y7, Y5, Y5
	VCVTDQ2PS   Y4, Y4
	VCVTDQ2PS   Y5, Y5
	VFMADD231PS Y11, Y4, Y0
	VFMADD231PS Y11, Y5, Y8

	VPMOVSXBW 32(SI), Y1
	VPMOVSXBW 32(DI), Y2
	VPMADDWD  Y2, Y1, Y4
	VPMOVSXBW 32(R8), Y3
	VPMADDWD  Y3, Y1, Y5

	VPMOVSXBW 48(SI), Y1
	VPMOVSXBW 48(DI), Y2
	VPMADDWD  Y2, Y1, Y6
	VPMOVSXBW 48(R8), Y3
	VPMADDWD  Y3, Y1, Y7

	VPADDD      Y6, Y4, Y4
	VPADDD      Y7, Y5, Y5
	VCVTDQ2PS   Y4, Y4
	VCVTDQ2PS   Y5, Y5
	VFMADD231PS Y12, Y4, Y9
	VFMADD231PS Y12, Y5, Y10

	// Groups 2 and 3
	VBROADCASTSS 8(DX), Y11
	VBROADCASTSS 12(DX), Y12

	VPMOVSXBW 64(SI), Y1
	VPMOVSXBW 64(DI), Y2
	VPMADDWD  Y2, Y1, Y4
	VPMOVSXBW 64(R8), Y3
	VPMADDWD  Y3, Y1, Y5

	VPMOVSXBW 80(SI), Y1
	VPMOVSXBW 80(DI), Y2
	VPMADDWD  Y2, Y1, Y6
	VPMOVSXBW 80(R8), Y3
	VPMADDWD  Y3, Y1, Y7

	VPADDD      Y6, Y4, Y4
	VPADDD      Y7, Y5, Y5
	VCVTDQ2PS   Y4, Y4
	VCVTDQ2PS   Y5, Y5
	VFMADD231PS Y11, Y4, Y0
	VFMADD231PS Y11, Y5, Y8

	VPMOVSXBW 96(SI), Y1
	VPMOVSXBW 96(DI), Y2
	VPMADDWD  Y2, Y1, Y4
	VPMOVSXBW 96(R8), Y3
	VPMADDWD  Y3, Y1, Y5

	VPMOVSXBW 112(SI), Y1
	VPMOVSXBW 112(DI), Y2
	VPMADDWD  Y2, Y1, Y6
	VPMOVSXBW 112(R8), Y3
	VPMADDWD  Y3, Y1, Y7

	VPADDD      Y6, Y4, Y4
	VPADDD      Y7, Y5, Y5
	VCVTDQ2PS   Y4, Y4
	VCVTDQ2PS   Y5, Y5
	VFMADD231PS Y12, Y4, Y9
	VFMADD231PS Y12, Y5, Y10

	ADDQ $128, SI
	ADDQ $128, DI
	ADDQ $128, R8
	ADDQ $16, DX
	SUBQ $4, CX
	CMPQ CX, $4
	JGE  scaled2_loop4

scaled2_rem:
	CMPQ CX, $2
	JLT  scaled2_tail

	VBROADCASTSS 0(DX), Y11
	VBROADCASTSS 4(DX), Y12

	VPMOVSXBW 0(SI), Y1
	VPMOVSXBW 0(DI), Y2
	VPMADDWD  Y2, Y1, Y4
	VPMOVSXBW 0(R8), Y3
	VPMADDWD  Y3, Y1, Y5

	VPMOVSXBW 16(SI), Y1
	VPMOVSXBW 16(DI), Y2
	VPMADDWD  Y2, Y1, Y6
	VPMOVSXBW 16(R8), Y3
	VPMADDWD  Y3, Y1, Y7

	VPADDD      Y6, Y4, Y4
	VPADDD      Y7, Y5, Y5
	VCVTDQ2PS   Y4, Y4
	VCVTDQ2PS   Y5, Y5
	VFMADD231PS Y11, Y4, Y0
	VFMADD231PS Y11, Y5, Y8

	VPMOVSXBW 32(SI), Y1
	VPMOVSXBW 32(DI), Y2
	VPMADDWD  Y2, Y1, Y4
	VPMOVSXBW 32(R8), Y3
	VPMADDWD  Y3, Y1, Y5

	VPMOVSXBW 48(SI), Y1
	VPMOVSXBW 48(DI), Y2
	VPMADDWD  Y2, Y1, Y6
	VPMOVSXBW 48(R8), Y3
	VPMADDWD  Y3, Y1, Y7

	VPADDD      Y6, Y4, Y4
	VPADDD      Y7, Y5, Y5
	VCVTDQ2PS   Y4, Y4
	VCVTDQ2PS   Y5, Y5
	VFMADD231PS Y12, Y4, Y9
	VFMADD231PS Y12, Y5, Y10

	ADDQ $64, SI
	ADDQ $64, DI
	ADDQ $64, R8
	ADDQ $8, DX
	SUBQ $2, CX

scaled2_tail:
	TESTQ CX, CX
	JZ    scaled2_done
	VBROADCASTSS 0(DX), Y11
	VPMOVSXBW    0(SI), Y1
	VPMOVSXBW    0(DI), Y2
	VPMADDWD     Y2, Y1, Y4
	VPMOVSXBW    0(R8), Y3
	VPMADDWD     Y3, Y1, Y5
	VPMOVSXBW    16(SI), Y1
	VPMOVSXBW    16(DI), Y2
	VPMADDWD     Y2, Y1, Y6
	VPMOVSXBW    16(R8), Y3
	VPMADDWD     Y3, Y1, Y7
	VPADDD       Y6, Y4, Y4
	VPADDD       Y7, Y5, Y5
	VCVTDQ2PS    Y4, Y4
	VCVTDQ2PS    Y5, Y5
	VFMADD231PS  Y11, Y4, Y0
	VFMADD231PS  Y11, Y5, Y8

scaled2_done:
	VADDPS       Y9, Y0, Y0
	VADDPS       Y10, Y8, Y8

	// Reduce Col 0:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	MOVSS        X0, ret0+40(FP)

	// Reduce Col 1:
	VEXTRACTF128 $1, Y8, X1
	VADDPS       X1, X8, X2
	VHADDPS      X2, X2, X2
	VHADDPS      X2, X2, X2
	MOVSS        X2, ret1+44(FP)

	VZEROUPPER
	RET

