// NEON body of the Q4_K dequantizer for arm64 (gguf_dequant_q4k_arm64.go).
//
// A super-block's 128 nibble bytes are four groups of 32. In a group, the low nibbles are one 32-weight sub-block and
// the high nibbles the next. Per 16 bytes: AND 0x0F and USHR #4 split the nibbles; USHLL widens bytes to 16 and then
// 32 bits; UCVTF makes them float32 (exact, the values are 0..15); FMUL by the sub-block's multiplier; FMLS subtracts
// dmin*min, fused. That is the scalar's order on arm64: an FMULS, then an FMSUBS whose product is dmin*min. The
// operand roles are the scalar's too, so a NaN in d or dmin propagates identically.
//
// All base ARMv8-A NEON, no feature detection.

#include "textflag.h"

// func dequantQ4KBodyNEON(qs *byte, out *float32, mul, mins *[8]float32, dmin float32)
TEXT ·dequantQ4KBodyNEON(SB), NOSPLIT, $0-36
	MOVD  qs+0(FP), R0
	MOVD  out+8(FP), R1
	MOVD  mul+16(FP), R2
	MOVD  mins+24(FP), R3
	MOVWU dmin+32(FP), R6
	VDUP  R6, V24.S4          // dmin
	MOVD  $15, R4
	VDUP  R4, V30.B16         // 0x0F in every byte
	MOVD  $4, R5              // four groups of 32 bytes

group:
	MOVWU (R2), R6
	VDUP  R6, V20.S4          // mul of the low-nibble sub-block
	MOVWU 4(R2), R6
	VDUP  R6, V21.S4          // mul of the high-nibble sub-block
	MOVWU (R3), R6
	VDUP  R6, V22.S4          // min, low
	MOVWU 4(R3), R6
	VDUP  R6, V23.S4          // min, high
	ADD   $8, R2
	ADD   $8, R3
	ADD   $128, R1, R7        // the high-nibble sub-block's 32 weights follow the low one's
	MOVD  $2, R8              // two halves of 16 bytes

half:
	VLD1.P 16(R0), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16   // low nibbles
	VUSHR  $4, V0.B16, V2.B16   // high nibbles

	VUSHLL $0, V1.B8, V4.H8
	VUSHLL2 $0, V1.B16, V5.H8
	VUSHLL $0, V4.H4, V6.S4
	VUSHLL2 $0, V4.H8, V7.S4
	VUSHLL $0, V5.H4, V8.S4
	VUSHLL2 $0, V5.H8, V9.S4
	VUCVTF V6.S4, V6.S4
	VUCVTF V7.S4, V7.S4
	VUCVTF V8.S4, V8.S4
	VUCVTF V9.S4, V9.S4
	VFMUL  V20.S4, V6.S4, V6.S4   // q * mul
	VFMUL  V20.S4, V7.S4, V7.S4
	VFMUL  V20.S4, V8.S4, V8.S4
	VFMUL  V20.S4, V9.S4, V9.S4
	VFMLS  V22.S4, V24.S4, V6.S4   // - dmin*min, fused
	VFMLS  V22.S4, V24.S4, V7.S4
	VFMLS  V22.S4, V24.S4, V8.S4
	VFMLS  V22.S4, V24.S4, V9.S4
	VST1.P [V6.S4, V7.S4, V8.S4, V9.S4], 64(R1)

	VUSHLL $0, V2.B8, V4.H8
	VUSHLL2 $0, V2.B16, V5.H8
	VUSHLL $0, V4.H4, V6.S4
	VUSHLL2 $0, V4.H8, V7.S4
	VUSHLL $0, V5.H4, V8.S4
	VUSHLL2 $0, V5.H8, V9.S4
	VUCVTF V6.S4, V6.S4
	VUCVTF V7.S4, V7.S4
	VUCVTF V8.S4, V8.S4
	VUCVTF V9.S4, V9.S4
	VFMUL  V21.S4, V6.S4, V6.S4
	VFMUL  V21.S4, V7.S4, V7.S4
	VFMUL  V21.S4, V8.S4, V8.S4
	VFMUL  V21.S4, V9.S4, V9.S4
	VFMLS  V23.S4, V24.S4, V6.S4
	VFMLS  V23.S4, V24.S4, V7.S4
	VFMLS  V23.S4, V24.S4, V8.S4
	VFMLS  V23.S4, V24.S4, V9.S4
	VST1.P [V6.S4, V7.S4, V8.S4, V9.S4], 64(R7)

	SUBS $1, R8, R8
	BNE  half

	ADD  $128, R1             // past the high-nibble sub-block just written through R7
	SUBS $1, R5, R5
	BNE  group

	RET
