// AVX2 body of the Q4_K dequantizer for amd64 (gguf_dequant_q4k_amd64.go).
//
// A super-block's 128 nibble bytes are four groups of 32. In a group, the low nibbles are one 32-weight sub-block and
// the high nibbles the next. Per 8 bytes: VPMOVZXBD widens them to dwords, VPAND 0x0F and VPSRLD $4 split the nibbles,
// VCVTDQ2PS makes them float32 (exact, the values are 0..15), then VMULPS by the sub-block's multiplier and VSUBPS of
// its offset: two rounded operations, not a fused one, because that is what the scalar is on amd64. The product is
// the first source of the subtract, as in the scalar, so a NaN propagates the same way.
//
// Every constant comes from memory: this runs once per 256 weights, and a move from a general register into an XMM
// register is a legacy-SSE instruction in Go's assembler, which costs a state transition with the YMM upper halves
// in use.

#include "textflag.h"

DATA q4kNibble<>(SB)/4, $0x0000000F
GLOBL q4kNibble<>(SB), RODATA|NOPTR, $4

// func dequantQ4KBodyAVX2(qs *byte, out *float32, mul, off *[8]float32)
TEXT ·dequantQ4KBodyAVX2(SB), NOSPLIT, $0-32
	MOVQ qs+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ mul+16(FP), R8
	MOVQ off+24(FP), R9
	VPBROADCASTD q4kNibble<>(SB), Y15
	MOVQ $4, CX               // four groups of 32 bytes

group:
	VBROADCASTSS (R8), Y12    // mul of the low-nibble sub-block
	VBROADCASTSS 4(R8), Y13   // mul of the high-nibble sub-block
	VBROADCASTSS (R9), Y10    // off, low
	VBROADCASTSS 4(R9), Y11   // off, high
	MOVQ $4, DX               // four chunks of 8 bytes

chunk:
	VPMOVZXBD (SI), Y0
	VPAND     Y15, Y0, Y1     // low nibbles
	VPSRLD    $4, Y0, Y2      // high nibbles
	VCVTDQ2PS Y1, Y1
	VCVTDQ2PS Y2, Y2
	VMULPS    Y12, Y1, Y1
	VMULPS    Y13, Y2, Y2
	VSUBPS    Y10, Y1, Y1     // product - off
	VSUBPS    Y11, Y2, Y2
	VMOVUPS   Y1, (DI)
	VMOVUPS   Y2, 128(DI)     // the high-nibble sub-block's 32 weights follow the low one's
	ADDQ $8, SI
	ADDQ $32, DI
	DECQ DX
	JNZ  chunk

	ADDQ $128, DI             // past the high-nibble sub-block
	ADDQ $8, R8
	ADDQ $8, R9
	DECQ CX
	JNZ  group

	VZEROUPPER
	RET
