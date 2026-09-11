#include "textflag.h"

// func dotI8x8AVX2(a, b0, b1, b2, b3, b4, b5, b6, b7 *int8, n int, out *[8]int32)
//
// Eight int8 dot products of one shared a-row against eight b-rows, n a multiple of 16.
//
// WHY. The W8A8 matmul calls dotI8 one (row, column) pair at a time (quant.go:250,
// :319), so it re-widens the SAME a-row for every column it scores. Each 16-element
// step of the 1×1 kernel is 2 VPMOVSXBW + 1 VPMADDWD + 1 VPADDD = 4 ops per 16 MACs;
// eight columns cost 32. Sharing the a-widening makes it 1 + 8×3 = 25 for the same
// 128 MACs — the a-side widening is paid once instead of eight times.
//
// The ceiling this is walking toward is VPMADDWD throughput, measured on this box at
// 69.0 GMAC/s (1.03 VPMADDWD/cycle — it is a single-pipe instruction). dotI8AVX2 sits
// at 43.7, or 63% of that; the 37% is the widening competing with the multiplies for
// the same pipes.
//
// NO BIT-IDENTITY RISK, UNLIKE THE f32 KERNELS. Integer addition is associative and
// int8×int8→int32 cannot overflow across any K this library sees (|Σ| ≤ K·127², and
// K would have to exceed 133k), so blocking and reordering are exactly equal to the
// scalar reference. TestDotI8x8_matchesScalar asserts that directly rather than
// against another SIMD kernel.
//
// Registers: Y0..Y7 the eight accumulators, Y8 the widened a, Y9 the widened b, Y10
// the product. SI holds a; R8..R15 the eight b pointers.
TEXT ·dotI8x8AVX2(SB), NOSPLIT, $0-88
	MOVQ a+0(FP), SI
	MOVQ b0+8(FP), R8
	MOVQ b1+16(FP), R9
	MOVQ b2+24(FP), R10
	MOVQ b3+32(FP), R11
	MOVQ b4+40(FP), R12
	MOVQ b5+48(FP), R13
	MOVQ b6+56(FP), R14
	MOVQ b7+64(FP), R15
	MOVQ n+72(FP), CX

	VPXOR Y0, Y0, Y0
	VPXOR Y1, Y1, Y1
	VPXOR Y2, Y2, Y2
	VPXOR Y3, Y3, Y3
	VPXOR Y4, Y4, Y4
	VPXOR Y5, Y5, Y5
	VPXOR Y6, Y6, Y6
	VPXOR Y7, Y7, Y7

loop16:
	CMPQ CX, $16
	JL   done

	VPMOVSXBW 0(SI), Y8 // a[k..k+16] widened once, reused by all eight columns

	VPMOVSXBW 0(R8), Y9
	VPMADDWD  Y9, Y8, Y10
	VPADDD    Y10, Y0, Y0

	VPMOVSXBW 0(R9), Y9
	VPMADDWD  Y9, Y8, Y10
	VPADDD    Y10, Y1, Y1

	VPMOVSXBW 0(R10), Y9
	VPMADDWD  Y9, Y8, Y10
	VPADDD    Y10, Y2, Y2

	VPMOVSXBW 0(R11), Y9
	VPMADDWD  Y9, Y8, Y10
	VPADDD    Y10, Y3, Y3

	VPMOVSXBW 0(R12), Y9
	VPMADDWD  Y9, Y8, Y10
	VPADDD    Y10, Y4, Y4

	VPMOVSXBW 0(R13), Y9
	VPMADDWD  Y9, Y8, Y10
	VPADDD    Y10, Y5, Y5

	VPMOVSXBW 0(R14), Y9
	VPMADDWD  Y9, Y8, Y10
	VPADDD    Y10, Y6, Y6

	VPMOVSXBW 0(R15), Y9
	VPMADDWD  Y9, Y8, Y10
	VPADDD    Y10, Y7, Y7

	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	ADDQ $16, R11
	ADDQ $16, R12
	ADDQ $16, R13
	ADDQ $16, R14
	ADDQ $16, R15
	SUBQ $16, CX
	JMP  loop16

done:
	MOVQ out+80(FP), DI

	VEXTRACTI128 $1, Y0, X9
	VPADDD  X9, X0, X0
	VPHADDD X0, X0, X0
	VPHADDD X0, X0, X0
	MOVL    X0, 0(DI)

	VEXTRACTI128 $1, Y1, X9
	VPADDD  X9, X1, X1
	VPHADDD X1, X1, X1
	VPHADDD X1, X1, X1
	MOVL    X1, 4(DI)

	VEXTRACTI128 $1, Y2, X9
	VPADDD  X9, X2, X2
	VPHADDD X2, X2, X2
	VPHADDD X2, X2, X2
	MOVL    X2, 8(DI)

	VEXTRACTI128 $1, Y3, X9
	VPADDD  X9, X3, X3
	VPHADDD X3, X3, X3
	VPHADDD X3, X3, X3
	MOVL    X3, 12(DI)

	VEXTRACTI128 $1, Y4, X9
	VPADDD  X9, X4, X4
	VPHADDD X4, X4, X4
	VPHADDD X4, X4, X4
	MOVL    X4, 16(DI)

	VEXTRACTI128 $1, Y5, X9
	VPADDD  X9, X5, X5
	VPHADDD X5, X5, X5
	VPHADDD X5, X5, X5
	MOVL    X5, 20(DI)

	VEXTRACTI128 $1, Y6, X9
	VPADDD  X9, X6, X6
	VPHADDD X6, X6, X6
	VPHADDD X6, X6, X6
	MOVL    X6, 24(DI)

	VEXTRACTI128 $1, Y7, X9
	VPADDD  X9, X7, X7
	VPHADDD X7, X7, X7
	VPHADDD X7, X7, X7
	MOVL    X7, 28(DI)

	VZEROUPPER
	RET
