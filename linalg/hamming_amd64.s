//go:build amd64

#include "textflag.h"

// func hammingPOPCNT(q *uint64, codes *uint64, words int, n int, dst *uint16)
//
// dst[i] = Σ_j popcount(codes[i*words+j] ^ q[j]).
//
// Why assembly for something math/bits already expresses: OnesCount64 is only
// intrinsified to POPCNTQ on amd64 when GOAMD64 >= v2. The default is v1, so a
// library build gets the SWAR fallback — ~12 arithmetic ops per word instead of
// one instruction — and that is the difference between the binary prefilter
// beating an exact float32 scan and losing to it. The feature is detected at
// init (cpufeat.HasPOPCNT) rather than at compile time, so the packaged binary stays
// GOAMD64=v1-portable and still runs the fast kernel wherever the CPU has it.
//
// The inner loop keeps FOUR independent accumulators. POPCNTQ is 1-cycle
// latency and 4/cycle throughput on this class of part, so a single
// accumulator's serial ADDQ chain — not the popcounts — would set the pace:
// 12 words = 12 dependent cycles, versus 3 when the chain is split four ways.
// Using a distinct destination register per POPCNTQ also sidesteps the
// well-known false dependency on the destination operand.
TEXT ·hammingPOPCNT(SB), NOSPLIT, $0-40
	MOVQ q+0(FP), SI
	MOVQ codes+8(FP), DI
	MOVQ words+16(FP), CX
	MOVQ n+24(FP), DX
	MOVQ dst+32(FP), BX

	XORQ R8, R8 // i = 0

row_loop:
	CMPQ R8, DX
	JGE  done

	XORQ R9, R9   // j = 0
	XORQ R10, R10 // four independent accumulators
	XORQ R11, R11
	XORQ R12, R12
	XORQ R13, R13

	MOVQ CX, R15
	ANDQ $-4, R15 // R15 = words &^ 3, the 4-word main-loop bound

word4_loop:
	CMPQ R9, R15
	JGE  word1_pre

	MOVQ    0(DI)(R9*8), AX
	XORQ    0(SI)(R9*8), AX
	POPCNTQ AX, AX
	ADDQ    AX, R10

	MOVQ    8(DI)(R9*8), R14
	XORQ    8(SI)(R9*8), R14
	POPCNTQ R14, R14
	ADDQ    R14, R11

	MOVQ    16(DI)(R9*8), AX
	XORQ    16(SI)(R9*8), AX
	POPCNTQ AX, AX
	ADDQ    AX, R12

	MOVQ    24(DI)(R9*8), R14
	XORQ    24(SI)(R9*8), R14
	POPCNTQ R14, R14
	ADDQ    R14, R13

	ADDQ $4, R9
	JMP  word4_loop

word1_pre:
	ADDQ R11, R10
	ADDQ R13, R12
	ADDQ R12, R10

word1_loop:
	CMPQ R9, CX
	JGE  store

	MOVQ    0(DI)(R9*8), AX
	XORQ    0(SI)(R9*8), AX
	POPCNTQ AX, AX
	ADDQ    AX, R10

	INCQ R9
	JMP  word1_loop

store:
	MOVW R10, 0(BX)(R8*2)
	LEAQ (DI)(CX*8), DI // advance to the next row's code
	INCQ R8
	JMP  row_loop

done:
	RET
