// NEON activation quantizer for arm64 (task-simd-audit.md S-03).
//
//   maxAbsF32NEON  — max |row[i]| over n f32 (n a multiple of 16), NaN-skipping.
//   quantizeF32NEON — q[i] = int8(clamp(roundTiesAway(row[i]*inv), -127, 127)).
//
// Both are the scalar passes of quantizeRowInt8CoreScalar, lane-for-lane:
//
//   abs/max:  FABS.4S then FMAXNM.4S into four accumulators (breaks the FP-max
//             latency chain four ways), folded with FMAXNM and FMAXNMV at the end.
//             FMAXNM/FMAXNMV are the IEEE maxNum forms: a quiet NaN operand is
//             treated as missing and the other operand returned — the same as the
//             scalar `v > maxAbs`, which is false for NaN. Max is exact, so the
//             fold order cannot change the value.
//   quantize: FMUL.4S by the broadcast inv (the scalar's single f32 `v * inv`),
//             FCVTAS.4S (f32 → int32, round to nearest, ties AWAY from zero —
//             math.Round's rule; the scalar widens the f32 product to f64 exactly
//             before rounding, so rounding the f32 directly is the same integer;
//             NaN → 0, and |x| ≥ 2^31 saturates, which the clamp then covers),
//             SMIN/SMAX against +127/-127, SQXTN/SQXTN2 int32 → int16 → int8 (the
//             saturation is inert — values are already within ±127 — it is just
//             the narrowing form that packs 16 lanes into one register), one
//             16-byte store.
//
// None of FABS/FMAXNM/FMAXNMV/FMUL/FCVTAS/SMIN/SMAX/SQXTN (vector) has a Go arm64
// assembler mnemonic, so they are raw WORDs, same convention as dequant_i8 and
// dot_w4a8. All base ARMv8-A NEON: no feature detection.
//
// Encodings (Rn<<5 | Rd, Rm<<16 where present):
//   FABS    Vd.4S, Vn.4S          = 0x4EA0F800
//   FMAXNM  Vd.4S, Vn.4S, Vm.4S   = 0x4E20C400
//   FMAXNMV Sd, Vn.4S             = 0x6E30C800
//   FMUL    Vd.4S, Vn.4S, Vm.4S   = 0x6E20DC00
//   FCVTAS  Vd.4S, Vn.4S          = 0x4E21C800
//   SMIN    Vd.4S, Vn.4S, Vm.4S   = 0x4EA06C00
//   SMAX    Vd.4S, Vn.4S, Vm.4S   = 0x4EA06400
//   SQXTN   Vd.4H,  Vn.4S         = 0x0E614800
//   SQXTN2  Vd.8H,  Vn.4S         = 0x4E614800
//   SQXTN   Vd.8B,  Vn.8H         = 0x0E214800
//   SQXTN2  Vd.16B, Vn.8H         = 0x4E214800

#include "textflag.h"

// func maxAbsF32NEON(row *float32, n int) float32
TEXT ·maxAbsF32NEON(SB), NOSPLIT, $0-20
	MOVD row+0(FP), R0
	MOVD n+8(FP), R1
	VEOR V16.B16, V16.B16, V16.B16   // four running maxima, all +0
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16

max16:
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	WORD   $0x4EA0F800        // FABS   V0.4S, V0.4S
	WORD   $0x4EA0F821        // FABS   V1.4S, V1.4S
	WORD   $0x4EA0F842        // FABS   V2.4S, V2.4S
	WORD   $0x4EA0F863        // FABS   V3.4S, V3.4S
	WORD   $0x4E20C610        // FMAXNM V16.4S, V16.4S, V0.4S
	WORD   $0x4E21C631        // FMAXNM V17.4S, V17.4S, V1.4S
	WORD   $0x4E22C652        // FMAXNM V18.4S, V18.4S, V2.4S
	WORD   $0x4E23C673        // FMAXNM V19.4S, V19.4S, V3.4S
	SUBS   $16, R1, R1
	BNE    max16

	WORD   $0x4E31C610        // FMAXNM V16.4S, V16.4S, V17.4S
	WORD   $0x4E33C652        // FMAXNM V18.4S, V18.4S, V19.4S
	WORD   $0x4E32C610        // FMAXNM V16.4S, V16.4S, V18.4S
	WORD   $0x6E30CA00        // FMAXNMV S0, V16.4S
	FMOVS  F0, ret+16(FP)
	RET

// func quantizeF32NEON(row *float32, q *int8, n int, inv float32)
TEXT ·quantizeF32NEON(SB), NOSPLIT, $0-28
	MOVD  row+0(FP), R0
	MOVD  q+8(FP), R1
	MOVD  n+16(FP), R2
	MOVWU inv+24(FP), R3
	VDUP  R3, V16.S4          // inv, broadcast (f32 bit pattern)
	MOVD  $127, R4
	VDUP  R4, V17.S4          // +127 int32 lanes
	MOVD  $-127, R5
	VDUP  R5, V18.S4          // -127 int32 lanes

quant16:
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	WORD   $0x6E30DC00        // FMUL   V0.4S, V0.4S, V16.4S   (v * inv, one f32 multiply)
	WORD   $0x6E30DC21        // FMUL   V1.4S, V1.4S, V16.4S
	WORD   $0x6E30DC42        // FMUL   V2.4S, V2.4S, V16.4S
	WORD   $0x6E30DC63        // FMUL   V3.4S, V3.4S, V16.4S
	WORD   $0x4E21C800        // FCVTAS V0.4S, V0.4S          (round, ties away → int32)
	WORD   $0x4E21C821        // FCVTAS V1.4S, V1.4S
	WORD   $0x4E21C842        // FCVTAS V2.4S, V2.4S
	WORD   $0x4E21C863        // FCVTAS V3.4S, V3.4S
	WORD   $0x4EB16C00        // SMIN   V0.4S, V0.4S, V17.4S   (min(x, 127))
	WORD   $0x4EB16C21        // SMIN   V1.4S, V1.4S, V17.4S
	WORD   $0x4EB16C42        // SMIN   V2.4S, V2.4S, V17.4S
	WORD   $0x4EB16C63        // SMIN   V3.4S, V3.4S, V17.4S
	WORD   $0x4EB26400        // SMAX   V0.4S, V0.4S, V18.4S   (max(x, -127))
	WORD   $0x4EB26421        // SMAX   V1.4S, V1.4S, V18.4S
	WORD   $0x4EB26442        // SMAX   V2.4S, V2.4S, V18.4S
	WORD   $0x4EB26463        // SMAX   V3.4S, V3.4S, V18.4S
	WORD   $0x0E614804        // SQXTN  V4.4H,  V0.4S          (q[0..3]   → int16)
	WORD   $0x4E614824        // SQXTN2 V4.8H,  V1.4S          (q[4..7])
	WORD   $0x0E614845        // SQXTN  V5.4H,  V2.4S          (q[8..11])
	WORD   $0x4E614865        // SQXTN2 V5.8H,  V3.4S          (q[12..15])
	WORD   $0x0E214886        // SQXTN  V6.8B,  V4.8H          (q[0..7]   → int8)
	WORD   $0x4E2148A6        // SQXTN2 V6.16B, V5.8H          (q[8..15])
	VST1.P [V6.B16], 16(R1)
	SUBS   $16, R2, R2
	BNE    quant16

	RET
