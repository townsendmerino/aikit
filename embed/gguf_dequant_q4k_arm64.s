// NEON body of the Q4_K dequantizer for arm64 (gguf_dequant_q4k_arm64.go).
//
// A super-block's 128 nibble bytes are four groups of 32. In a group, the low nibbles are one 32-weight sub-block and
// the high nibbles the next. Per 16 bytes: AND 0x0F and USHR #4 split the nibbles; USHLL widens bytes to 16 and then
// 32 bits; UCVTF makes them float32 (exact, the values are 0..15); FMUL by the sub-block's multiplier; FMLS subtracts
// dmin*min, fused. That is the scalar's order on arm64: an FMULS, then an FMSUBS whose product is dmin*min. The
// operand roles are the scalar's too, so a NaN in d or dmin propagates identically.
//
// USHR, USHLL, UCVTF, FMUL and FMLS (vector) have no Go arm64 assembler mnemonic: raw WORDs, the convention of
// linalg/quant_act_arm64.s. All base ARMv8-A NEON, no feature detection.
//
// Encodings (Rn<<5 | Rd, Rm<<16 where present):
//   USHR   Vd.16B, Vn.16B, #4     = 0x6F0C0400
//   USHLL  Vd.8H,  Vn.8B,  #0     = 0x2F08A400      USHLL2 Vd.8H, Vn.16B, #0 = 0x6F08A400
//   USHLL  Vd.4S,  Vn.4H,  #0     = 0x2F10A400      USHLL2 Vd.4S, Vn.8H,  #0 = 0x6F10A400
//   UCVTF  Vd.4S,  Vn.4S          = 0x6E21D800
//   FMUL   Vd.4S,  Vn.4S, Vm.4S   = 0x6E20DC00
//   FMLS   Vd.4S,  Vn.4S, Vm.4S   = 0x4EA0CC00      (Vd = Vd - Vn*Vm)

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
	WORD   $0x6F0C0402        // USHR   V2.16B, V0.16B, #4     (high nibbles)

	WORD   $0x2F08A424        // USHLL  V4.8H, V1.8B,  #0
	WORD   $0x6F08A425        // USHLL2 V5.8H, V1.16B, #0
	WORD   $0x2F10A486        // USHLL  V6.4S, V4.4H,  #0
	WORD   $0x6F10A487        // USHLL2 V7.4S, V4.8H,  #0
	WORD   $0x2F10A4A8        // USHLL  V8.4S, V5.4H,  #0
	WORD   $0x6F10A4A9        // USHLL2 V9.4S, V5.8H,  #0
	WORD   $0x6E21D8C6        // UCVTF  V6.4S, V6.4S
	WORD   $0x6E21D8E7        // UCVTF  V7.4S, V7.4S
	WORD   $0x6E21D908        // UCVTF  V8.4S, V8.4S
	WORD   $0x6E21D929        // UCVTF  V9.4S, V9.4S
	WORD   $0x6E34DCC6        // FMUL   V6.4S, V6.4S, V20.4S    (q * mul)
	WORD   $0x6E34DCE7        // FMUL   V7.4S, V7.4S, V20.4S
	WORD   $0x6E34DD08        // FMUL   V8.4S, V8.4S, V20.4S
	WORD   $0x6E34DD29        // FMUL   V9.4S, V9.4S, V20.4S
	WORD   $0x4EB6CF06        // FMLS   V6.4S, V24.4S, V22.4S   (- dmin*min, fused)
	WORD   $0x4EB6CF07        // FMLS   V7.4S, V24.4S, V22.4S
	WORD   $0x4EB6CF08        // FMLS   V8.4S, V24.4S, V22.4S
	WORD   $0x4EB6CF09        // FMLS   V9.4S, V24.4S, V22.4S
	VST1.P [V6.S4, V7.S4, V8.S4, V9.S4], 64(R1)

	WORD   $0x2F08A444        // USHLL  V4.8H, V2.8B,  #0
	WORD   $0x6F08A445        // USHLL2 V5.8H, V2.16B, #0
	WORD   $0x2F10A486        // USHLL  V6.4S, V4.4H,  #0
	WORD   $0x6F10A487        // USHLL2 V7.4S, V4.8H,  #0
	WORD   $0x2F10A4A8        // USHLL  V8.4S, V5.4H,  #0
	WORD   $0x6F10A4A9        // USHLL2 V9.4S, V5.8H,  #0
	WORD   $0x6E21D8C6        // UCVTF  V6.4S, V6.4S
	WORD   $0x6E21D8E7        // UCVTF  V7.4S, V7.4S
	WORD   $0x6E21D908        // UCVTF  V8.4S, V8.4S
	WORD   $0x6E21D929        // UCVTF  V9.4S, V9.4S
	WORD   $0x6E35DCC6        // FMUL   V6.4S, V6.4S, V21.4S
	WORD   $0x6E35DCE7        // FMUL   V7.4S, V7.4S, V21.4S
	WORD   $0x6E35DD08        // FMUL   V8.4S, V8.4S, V21.4S
	WORD   $0x6E35DD29        // FMUL   V9.4S, V9.4S, V21.4S
	WORD   $0x4EB7CF06        // FMLS   V6.4S, V24.4S, V23.4S
	WORD   $0x4EB7CF07        // FMLS   V7.4S, V24.4S, V23.4S
	WORD   $0x4EB7CF08        // FMLS   V8.4S, V24.4S, V23.4S
	WORD   $0x4EB7CF09        // FMLS   V9.4S, V24.4S, V23.4S
	VST1.P [V6.S4, V7.S4, V8.S4, V9.S4], 64(R7)

	SUBS $1, R8, R8
	BNE  half

	ADD  $128, R1             // past the high-nibble sub-block just written through R7
	SUBS $1, R5, R5
	BNE  group

	RET
