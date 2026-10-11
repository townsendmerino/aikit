// AVX2 body of the Q6_K dequantizer for amd64 (gguf_dequant_q6k_amd64.go).
//
// A super-block is two chunks of 128 weights. A chunk reads 64 bytes of ql (low four bits) and 32 of qh (high two
// bits, four quants to a byte). For l in 0..31, with a = ql[l], b = ql[l+32], h = qh[l], the chunk's four runs of 32
// weights take the quants
//
//	(a & 15) | (h<<4 & 48)      (b & 15) | (h<<2 & 48)      (a >> 4) | (h & 48)      (b >> 4) | (h>>2 & 48)
//
// each less 32. Sixteen l at a time share their four multipliers. Per 8 l: VPMOVZXBD widens a, b and h to dwords; the
// quants are built and recentred there; VCVTDQ2PS makes them float32 (exact, the values are -32..31); VMULPS by the
// multiplier (a quant is never a NaN, so the operand order cannot show); 8 weights go into each run.
//
// Every constant comes from memory: a move from a general register into an XMM register is a legacy-SSE instruction
// in Go's assembler, which costs a state transition with the YMM upper halves in use.

#include "textflag.h"

DATA q6kLow<>(SB)/4, $0x0000000F
GLOBL q6kLow<>(SB), RODATA|NOPTR, $4
DATA q6kHigh<>(SB)/4, $0x00000030
GLOBL q6kHigh<>(SB), RODATA|NOPTR, $4
DATA q6kBias<>(SB)/4, $0x00000020
GLOBL q6kBias<>(SB), RODATA|NOPTR, $4

// func dequantQ6KBodyAVX2(q *byte, out *float32, ds *[16]float32)
TEXT ·dequantQ6KBodyAVX2(SB), NOSPLIT, $0-24
	MOVQ q+0(FP), SI          // ql
	MOVQ out+8(FP), DI
	MOVQ ds+16(FP), R8
	LEAQ 128(SI), R9          // qh
	VPBROADCASTD q6kLow<>(SB), Y15
	VPBROADCASTD q6kHigh<>(SB), Y14
	VPBROADCASTD q6kBias<>(SB), Y13
	MOVQ $2, CX               // two chunks

chunk:
	MOVQ $2, DX               // two steps of 16 l

step:
	VBROADCASTSS (R8), Y8
	VBROADCASTSS 4(R8), Y9
	VBROADCASTSS 8(R8), Y10
	VBROADCASTSS 12(R8), Y11
	ADDQ $16, R8
	MOVQ $2, BX               // two halves of 8 l

half:
	VPMOVZXBD (SI), Y0        // a
	VPMOVZXBD 32(SI), Y1      // b
	VPMOVZXBD (R9), Y2        // h

	VPAND  Y15, Y0, Y3        // a & 15
	VPSLLD $4, Y2, Y7
	VPAND  Y14, Y7, Y7
	VPOR   Y7, Y3, Y3
	VPSUBD Y13, Y3, Y3        // run 0's quants - 32

	VPAND  Y15, Y1, Y4        // b & 15
	VPSLLD $2, Y2, Y7
	VPAND  Y14, Y7, Y7
	VPOR   Y7, Y4, Y4
	VPSUBD Y13, Y4, Y4        // run 1

	VPSRLD $4, Y0, Y5         // a >> 4
	VPAND  Y14, Y2, Y7
	VPOR   Y7, Y5, Y5
	VPSUBD Y13, Y5, Y5        // run 2

	VPSRLD $4, Y1, Y6         // b >> 4
	VPSRLD $2, Y2, Y7
	VPAND  Y14, Y7, Y7
	VPOR   Y7, Y6, Y6
	VPSUBD Y13, Y6, Y6        // run 3

	VCVTDQ2PS Y3, Y3
	VCVTDQ2PS Y4, Y4
	VCVTDQ2PS Y5, Y5
	VCVTDQ2PS Y6, Y6
	VMULPS    Y8, Y3, Y3
	VMULPS    Y9, Y4, Y4
	VMULPS    Y10, Y5, Y5
	VMULPS    Y11, Y6, Y6
	VMOVUPS   Y3, (DI)
	VMOVUPS   Y4, 128(DI)
	VMOVUPS   Y5, 256(DI)
	VMOVUPS   Y6, 384(DI)

	ADDQ $8, SI
	ADDQ $8, R9
	ADDQ $32, DI
	DECQ BX
	JNZ  half

	DECQ DX
	JNZ  step

	ADDQ $32, SI              // past the chunk's second 32 bytes of ql, read as b
	ADDQ $384, DI             // past the chunk's runs 1 to 3
	DECQ CX
	JNZ  chunk

	VZEROUPPER
	RET
