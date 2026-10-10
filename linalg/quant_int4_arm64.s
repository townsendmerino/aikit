// NEON int4 weight quantizer for arm64: the round-clamp-pack pass of quantizeInt4GroupScalar, 16 weights a turn.
//
// Per lane: FMUL by the broadcast inv (the scalar's single f32 `x * inv`), FCVTAS (f32 to int32, round to nearest,
// ties away from zero; NaN gives 0, and an infinity or a value past int32 saturates toward its sign), SMIN/SMAX
// against +7/-7, ADD 8 (so each lane is a nibble, 1..15). SQXTN narrows the sixteen lanes to sixteen bytes
// n0..n15; the narrowing's saturation never fires. Read as eight 16-bit lanes, lane j is n[2j] | n[2j+1]<<8, and
// USRA #4 adds the lane shifted right by four to itself: the low byte becomes n[2j] | n[2j+1]<<4 with no carry,
// because the two nibbles occupy different bits. XTN keeps the low bytes: the eight packed bytes.
//
// FMUL, FCVTAS, SMIN, SMAX, SQXTN, USRA and XTN (vector) have no Go arm64 assembler mnemonic: raw WORDs, the
// convention of quant_act_arm64.s. All base ARMv8-A NEON, no feature detection.
//
// Encodings (Rn<<5 | Rd, Rm<<16 where present):
//   FMUL    Vd.4S, Vn.4S, Vm.4S   = 0x6E20DC00
//   FCVTAS  Vd.4S, Vn.4S          = 0x4E21C800
//   SMIN    Vd.4S, Vn.4S, Vm.4S   = 0x4EA06C00
//   SMAX    Vd.4S, Vn.4S, Vm.4S   = 0x4EA06400
//   SQXTN   Vd.4H,  Vn.4S         = 0x0E614800
//   SQXTN2  Vd.8H,  Vn.4S         = 0x4E614800
//   SQXTN   Vd.8B,  Vn.8H         = 0x0E214800
//   SQXTN2  Vd.16B, Vn.8H         = 0x4E214800
//   USRA    Vd.8H,  Vn.8H, #4     = 0x6F1C1400
//   XTN     Vd.8B,  Vn.8H         = 0x0E212800

#include "textflag.h"

// func quantizeInt4F32NEON(row *float32, packed *byte, n int, inv float32)
TEXT ·quantizeInt4F32NEON(SB), NOSPLIT, $0-28
	MOVD  row+0(FP), R0
	MOVD  packed+8(FP), R1
	MOVD  n+16(FP), R2
	MOVWU inv+24(FP), R3
	VDUP  R3, V16.S4          // inv, broadcast (f32 bit pattern)
	MOVD  $7, R4
	VDUP  R4, V17.S4          // +7 int32 lanes
	MOVD  $-7, R5
	VDUP  R5, V18.S4          // -7 int32 lanes
	MOVD  $8, R6
	VDUP  R6, V19.S4          // the nibble bias

quant16:
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	WORD   $0x6E30DC00        // FMUL   V0.4S, V0.4S, V16.4S   (x * inv, one f32 multiply)
	WORD   $0x6E30DC21        // FMUL   V1.4S, V1.4S, V16.4S
	WORD   $0x6E30DC42        // FMUL   V2.4S, V2.4S, V16.4S
	WORD   $0x6E30DC63        // FMUL   V3.4S, V3.4S, V16.4S
	WORD   $0x4E21C800        // FCVTAS V0.4S, V0.4S          (round, ties away, to int32)
	WORD   $0x4E21C821        // FCVTAS V1.4S, V1.4S
	WORD   $0x4E21C842        // FCVTAS V2.4S, V2.4S
	WORD   $0x4E21C863        // FCVTAS V3.4S, V3.4S
	WORD   $0x4EB16C00        // SMIN   V0.4S, V0.4S, V17.4S   (min(q, 7))
	WORD   $0x4EB16C21        // SMIN   V1.4S, V1.4S, V17.4S
	WORD   $0x4EB16C42        // SMIN   V2.4S, V2.4S, V17.4S
	WORD   $0x4EB16C63        // SMIN   V3.4S, V3.4S, V17.4S
	WORD   $0x4EB26400        // SMAX   V0.4S, V0.4S, V18.4S   (max(q, -7))
	WORD   $0x4EB26421        // SMAX   V1.4S, V1.4S, V18.4S
	WORD   $0x4EB26442        // SMAX   V2.4S, V2.4S, V18.4S
	WORD   $0x4EB26463        // SMAX   V3.4S, V3.4S, V18.4S
	VADD   V19.S4, V0.S4, V0.S4   // q + 8: a nibble, 1..15
	VADD   V19.S4, V1.S4, V1.S4
	VADD   V19.S4, V2.S4, V2.S4
	VADD   V19.S4, V3.S4, V3.S4
	WORD   $0x0E614804        // SQXTN  V4.4H,  V0.4S          (n[0..3]   to int16)
	WORD   $0x4E614824        // SQXTN2 V4.8H,  V1.4S          (n[4..7])
	WORD   $0x0E614845        // SQXTN  V5.4H,  V2.4S          (n[8..11])
	WORD   $0x4E614865        // SQXTN2 V5.8H,  V3.4S          (n[12..15])
	WORD   $0x0E214886        // SQXTN  V6.8B,  V4.8H          (n[0..7]   to bytes)
	WORD   $0x4E2148A6        // SQXTN2 V6.16B, V5.8H          (n[8..15])
	WORD   $0x6F1C14C6        // USRA   V6.8H, V6.8H, #4       (low byte: n[2j] | n[2j+1]<<4)
	WORD   $0x0E2128C7        // XTN    V7.8B, V6.8H           (the eight packed bytes)
	VMOV   V7.D[0], R7
	MOVD.P R7, 8(R1)
	SUBS   $16, R2, R2
	BNE    quant16

	RET
