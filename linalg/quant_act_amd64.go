//go:build amd64

package linalg

// AVX2 activation quantizer (task-simd-audit.md S-03's "open" amd64 half — the NEON form shipped
// 2026-09-03; this is its amd64 twin).
//
// quantizeRowInt8Core runs two scalar passes on the calling goroutine before every W8A8/W4A8
// fan-out. Both passes are elementwise (a max, then a multiply-round-clamp), so the vector forms
// are bit-identical to the scalar loops by construction; TestQuantizeRowInt8_bitIdenticalToScalar
// (arch-independent, quant_act_test.go) and TestQuantActAVX2Kernels_matchScalar (this package,
// against the raw kernels directly) hold them to it over random rows and the pinned corners (NaN,
// ±Inf, -0.0, exact .5 ties, saturating magnitudes, all-zero).
//
// TWO ISA DIFFERENCES FROM ARM64, BOTH HANDLED EXPLICITLY, NEITHER GUESSED:
//
//  1. VMAXPS's NaN rule is asymmetric (Intel SDM: "if a value in the second [SRC2] operand is a
//     NaN, that NaN is returned" — i.e. when either operand is NaN, the operand supplied as Go's
//     FIRST listed source register is what the instruction returns), unlike ARM's FMAXNM, which is
//     the IEEE maxNum form and skips a NaN on EITHER side. Empirically verified on this box
//     (a throwaway VMAXSS probe, both operand orders, both argument orders): the register listed
//     FIRST in Go's `VMAXPS srcA, srcB, dst` survives when either operand is NaN. The running
//     accumulator is therefore always srcA; the freshly-loaded row data is always srcB — so a NaN
//     row value can never poison the accumulator, matching the scalar `v > maxAbs` (false for NaN
//     v, so maxAbs is left unchanged).
//  2. x86's VCVTTPS2DQ (truncate-to-int32) returns a single fixed "integer indefinite" value
//     (INT32_MIN) for ANY invalid conversion — NaN or a magnitude that overflows int32 alike —
//     unlike ARM's FCVTAS, which saturates toward the operand's own sign on overflow. Naively
//     porting the NEON pipeline (round, truncate, then SMIN/SMAX-equivalent clamp) would turn
//     +Inf into -127 instead of +127: the clamp receives INT32_MIN both for +Inf and for NaN, and
//     cannot tell them apart. Fixed by clamping the FLOAT into [-127, 127] BEFORE truncating
//     (VMINPS/VMAXPS against the two constants, before VCVTTPS2DQ) — every truncation input is
//     then guaranteed in-range, so the indefinite-value path is never taken for a genuine ±Inf.
//     NaN still needs its own fix (the float clamp turns NaN into 127 as a side effect, using
//     the same operand-order rule as (1) — not the 0 the scalar path gives), handled by masking:
//     an unordered compare on the pre-round product (VCMPPS, predicate 3) marks NaN lanes, and
//     VANDNPS(mask, result) zeroes exactly those lanes — cheaper than a full blend since the
//     "else" value is 0.
//
// quant_act_amd64_test.go's TestQuantActAVX2Kernels_matchScalar exercises both directly, and
// quant_act_test.go's corner cases (posinf-in-body, both-inf, nan-is-largest-slot, …) exercise
// them through the real quantizeRowInt8Core entry point.

// maxAbsF32AVX2 returns max |row[i]| over the first n elements (n a multiple of 8, n > 0),
// skipping quiet NaNs the way the scalar comparisons do. Implemented in quant_act_amd64.s.
//
//go:noescape
func maxAbsF32AVX2(row *float32, n int) float32

// quantizeF32AVX2 writes q[i] = int8(clamp(roundTiesAway(row[i]*inv), -127, 127)) for the first n
// elements (n a multiple of 8, n > 0). Implemented in quant_act_amd64.s.
//
//go:noescape
func quantizeF32AVX2(row *float32, q *int8, n int, inv float32)

// maxAbsF32 is the abs/max pass: the 8-aligned prefix in AVX2, the tail in the scalar loop
// continuing from the vector result (max is exact, so the split is invisible in the value).
func maxAbsF32(row []float32) float32 {
	n := len(row) &^ 7
	var m float32
	if n > 0 {
		m = maxAbsF32AVX2(&row[0], n)
	}
	if n < len(row) {
		m = maxAbsF32Scalar(row[n:], m)
	}
	return m
}

// quantizeRowScaled is the round-and-clamp pass: the 8-aligned prefix in AVX2, the tail scalar.
// Every production row (K = 768…18944) is a multiple of 8, so the tail exists for odd test shapes.
func quantizeRowScaled(row []float32, q []int8, inv float32) {
	if len(row) == 0 {
		return
	}
	_ = q[len(row)-1]
	n := len(row) &^ 7
	if n > 0 {
		quantizeF32AVX2(&row[0], &q[0], n, inv)
	}
	if n < len(row) {
		quantizeRowScaledScalar(row[n:], q[n:], inv)
	}
}
