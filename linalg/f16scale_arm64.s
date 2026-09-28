//go:build arm64

#include "textflag.h"

// func cvtF16ToF32x8NEON(dst *float32, src *uint16, n8 int)
//
// n8 blocks of eight binary16 → f32: one 16-byte load, FCVTL for the low four halves and FCVTL2 for the high
// four, one 32-byte store. The arm64 twin of cvtF16ToF32x8F16C (dot_w4a8_f16_amd64.s). FCVTL from half is
// base ARMv8.0 AdvSIMD (sz=0: half → single, no FEAT_FP16) and exact for every finite input, subnormals
// included, so the result is f16ToF32's bit for bit; a signalling NaN comes back quieted, and a scale is
// never NaN.
//
// WORDs from the encoding FCVTL{2} Vd.4S, Vn.{4,8}H = 0x0E217800 | Q<<30 | Rn<<5 | Rd, cross-checked against
// clang's assembler (fcvtl v1.4s, v0.4h = 0e217801; fcvtl2 v2.4s, v0.8h = 4e217802; 2026-09-28).
TEXT ·cvtF16ToF32x8NEON(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n8+16(FP), R2
	CBZ  R2, done

cvt:
	VLD1.P 16(R1), [V0.H8]
	WORD   $0x0E217801 // FCVTL  V1.4S, V0.4H
	WORD   $0x4E217802 // FCVTL2 V2.4S, V0.8H
	VST1.P [V1.S4, V2.S4], 32(R0)
	SUBS   $1, R2, R2
	BNE    cvt

done:
	RET
