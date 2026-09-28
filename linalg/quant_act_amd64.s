// AVX2 activation quantizer for amd64 (task-simd-audit.md S-03's amd64 half — see
// quant_act_amd64.go's doc comment for the two ISA differences from the arm64/NEON twin
// (quant_act_arm64.s) this file mirrors, both handled explicitly below, neither assumed).
//
//   maxAbsF32AVX2  — max |row[i]| over n f32 (n a multiple of 8), NaN-skipping.
//   quantizeF32AVX2 — q[i] = int8(clamp(roundTiesAway(row[i]*inv), -127, 127)).
//
// maxAbsF32AVX2: VANDPS with the abs-mask (0x7FFFFFFF) clears the sign bit exactly — -0.0
// becomes +0.0 with no comparison needed, matching the scalar `if v<0 { v=-v }` followed by
// `if v>maxAbs`, where -0.0 never updates maxAbs (an equal-not-greater comparison). Then VMAXPS
// folds the freshly-abs'd row data into the running accumulator, with the ACCUMULATOR always the
// first Go-listed operand — verified empirically (a throwaway VMAXSS probe on this box, both
// operand orders): the first-listed operand survives when either input is NaN, matching the
// scalar `v > maxAbs` (false for NaN v, so maxAbs is unchanged). Four independent accumulators,
// the same reasoning quant_act_arm64.s gives for its four: break the max-latency chain.
//
// quantizeF32AVX2, per lane: y = row[i]*inv (VMULPS, one f32 multiply, matching the scalar's
// single `v * inv`); round-half-away-from-zero via y + copysign(0.5, y) (VANDPS the sign bit,
// VORPS onto +0.5's bits, VADDPS) — x86 has no single-instruction "round ties away" mode (VROUNDPS
// only offers nearest-even/floor/ceil/truncate), so this is the composite form
// quant_act_other.go's own comment already named as the open design; then the float result is
// clamped into [-127, 127] (VMINPS/VMAXPS against the two constants, CONSTANT operand first per
// the same NaN-order rule as the max pass — so a NaN here is folded to +127 by this step, not yet
// the 0 the scalar path wants) BEFORE truncating (VCVTTPS2DQ) — critical, because x86's truncate
// returns one fixed "integer indefinite" value (INT32_MIN) for ANY invalid conversion, unlike
// ARM's FCVTAS, which saturates toward the input's own sign; clamping the float first guarantees
// every truncation input is in-range, so +Inf reaches this step as exactly +127.0 (from the
// pre-truncate clamp) rather than ever hitting the indefinite-value path. NaN still needs an
// explicit fix: VCMPPS ($7, ordered) gives an all-1s mask on every lane that is NOT NaN, and
// VPAND(mask, result) zeroes exactly the NaN lanes (using the ORDERED predicate + a commutative
// AND, not the unordered predicate + ANDN, so no operand-order assumption is needed for THIS
// step — AND does not care which side is which). Finally VPACKSSDW+VPACKSSWB narrow int32→int8,
// each within a single 128-bit lane (VEXTRACTI128 first splits the YMM in half) so there is no
// AVX2 cross-256-bit-lane interleaving to get wrong; every value is already inside ±127 by this
// point, so the pack instructions' own saturation never fires — it is just the narrowing
// mechanism.
//
// Both kernels are lane-for-lane bit-identical to quantizeRowInt8CoreScalar's two scalar passes
// (TestQuantActAVX2Kernels_matchScalar, this package; TestQuantizeRowInt8_bitIdenticalToScalar and
// TestQuantizeRowInt8_corners, quant_act_test.go, arch-independent — posinf-in-body, both-inf,
// nan-is-largest-slot, negzero-scattered, exact-ties-tail among the pinned corners).

#include "textflag.h"

// func maxAbsF32AVX2(row *float32, n int) float32
TEXT ·maxAbsF32AVX2(SB), NOSPLIT, $0-20
	MOVQ row+0(FP), SI
	MOVQ n+8(FP), CX

	// abs-mask 0x7FFFFFFF in every lane, built in-register (VPCMPEQD self -> all-ones,
	// VPSRLD $1 -> 0x7FFFFFFF): no memory constant, no extra cache line.
	VPCMPEQD Y15, Y15, Y15
	VPSRLD   $1, Y15, Y15

	VXORPS Y8, Y8, Y8 // four running maxima, all +0.0
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	VXORPS Y11, Y11, Y11

max32:
	CMPQ CX, $32
	JL   max8

	VMOVUPS   (SI), Y0
	VMOVUPS   32(SI), Y1
	VMOVUPS   64(SI), Y2
	VMOVUPS   96(SI), Y3
	VANDPS    Y15, Y0, Y0 // abs: clear the sign bit (-0.0 -> +0.0, exact)
	VANDPS    Y15, Y1, Y1
	VANDPS    Y15, Y2, Y2
	VANDPS    Y15, Y3, Y3
	VMAXPS    Y8, Y0, Y8  // accumulator FIRST (verified empirically): NaN in the row data cannot win
	VMAXPS    Y9, Y1, Y9
	VMAXPS    Y10, Y2, Y10
	VMAXPS    Y11, Y3, Y11

	ADDQ $128, SI
	SUBQ $32, CX
	JMP  max32

max8:
	CMPQ CX, $8
	JL   maxfold

max8loop:
	VMOVUPS (SI), Y0
	VANDPS  Y15, Y0, Y0
	VMAXPS  Y8, Y0, Y8
	ADDQ    $32, SI
	SUBQ    $8, CX
	CMPQ    CX, $8
	JGE     max8loop

maxfold:
	VMAXPS Y9, Y8, Y8
	VMAXPS Y11, Y10, Y10
	VMAXPS Y10, Y8, Y8
	// horizontal max of Y8's 8 lanes -> X0
	VEXTRACTF128 $1, Y8, X1
	VMAXPS       X1, X8, X0
	VSHUFPS      $0xEE, X0, X0, X1 // X1 = X0[2,3,2,3]
	VMAXPS       X1, X0, X0
	VSHUFPS      $0x55, X0, X0, X1 // X1 = X0[1,1,1,1]
	VMAXSS       X1, X0, X0
	VMOVSS       X0, ret+16(FP)
	VZEROUPPER
	RET

// func quantizeF32AVX2(row *float32, q *int8, n int, inv float32)
TEXT ·quantizeF32AVX2(SB), NOSPLIT, $0-28
	MOVQ         row+0(FP), SI
	MOVQ         q+8(FP), DI
	MOVQ         n+16(FP), CX
	VBROADCASTSS inv+24(FP), Y14

	VPCMPEQD Y12, Y12, Y12
	VPSLLD   $31, Y12, Y13   // Y13 = 0x80000000 in every lane (sign mask, in-register, no data load)

	MOVL    $0x3F000000, AX // 0.5f
	MOVQ    AX, X12
	VPBROADCASTD X12, Y12    // Y12 = 0.5f in every lane

	MOVL    $0x42FE0000, AX // 127.0f
	MOVQ    AX, X11
	VPBROADCASTD X11, Y11    // Y11 = +127.0 in every lane

	MOVL    $0xC2FE0000, AX // -127.0f
	MOVQ    AX, X10
	VPBROADCASTD X10, Y10    // Y10 = -127.0 in every lane

quant8:
	CMPQ CX, $8
	JL   done

	VMOVUPS (SI), Y0
	VMULPS  Y14, Y0, Y0          // y = row * inv

	VANDPS  Y13, Y0, Y1           // sign bit of y
	VORPS   Y12, Y1, Y1           // ±0.5 with y's own sign
	VADDPS  Y1, Y0, Y2            // r = y + copysign(0.5, y)

	VMINPS  Y2, Y11, Y2            // const FIRST: r = min(127, r)  (NaN -> 127, fixed below)
	VMAXPS  Y2, Y10, Y2            // r = max(-127, r)              (NaN -> stays 127 either way)
	VCVTTPS2DQ Y2, Y3              // i32 = trunc(r); r is always in [-127,127], never overflows

	VCMPPS  $7, Y0, Y0, Y4          // ordered predicate: all-1s where y is NOT NaN
	VPAND   Y4, Y3, Y3              // zero the NaN lanes (AND is commutative: order-safe)

	VEXTRACTI128 $1, Y3, X5
	VPACKSSDW    X5, X3, X3         // 8x int32 -> 8x int16, single 128-bit lane, order preserved
	VPACKSSWB    X3, X3, X3         // 8x int16 -> 8(+8 dup) int8; only the low 8 bytes are stored
	VMOVQ        X3, (DI)

	ADDQ $32, SI
	ADDQ $8, DI
	SUBQ $8, CX
	JMP  quant8

done:
	VZEROUPPER
	RET
