//go:build arm64

package linalg

// NEON activation quantizer (task-simd-audit.md S-03).
//
// quantizeRowInt8Core ran two scalar passes on the calling goroutine before every
// W8A8/W4A8 fan-out — seven times per layer on the 1.5B, ~509k elements per decode
// token, an estimated 0.6–1.6 ms of a ~25 ms token. Both passes are elementwise
// (a max, then a multiply-round-clamp), so the vector forms are bit-identical to
// the scalar loops by construction; TestQuantizeRowInt8_bitIdenticalToScalar
// holds them to it over random rows and the pinned corners (NaN, ±Inf, -0.0,
// exact .5 ties, saturating magnitudes, all-zero).

// maxAbsF32NEON returns max |row[i]| over the first n elements (n a multiple of
// 16, n > 0), skipping quiet NaNs the way the scalar comparisons do. FABS then
// FMAXNM into four accumulators, FMAXNMV at the end. Implemented in
// quant_act_arm64.s; base ARMv8-A NEON, no feature check.
//
//go:noescape
func maxAbsF32NEON(row *float32, n int) float32

// quantizeF32NEON writes q[i] = int8(clamp(roundTiesAway(row[i]*inv), -127, 127))
// for the first n elements (n a multiple of 16, n > 0): FMUL by the broadcast inv
// (the same single f32 multiply as the scalar `v * inv`), FCVTAS (round to nearest,
// ties away — math.Round's rule, on the f32 value the scalar widens exactly to
// f64), SMIN/SMAX against ±127, then a saturating narrow to int8 and one 16-byte
// store. NaN converts to 0 under FCVTAS, which is what the scalar's int8(NaN)
// yields on arm64. Implemented in quant_act_arm64.s.
//
//go:noescape
func quantizeF32NEON(row *float32, q *int8, n int, inv float32)

// maxAbsF32 is the abs/max pass: the 16-aligned prefix in NEON, the tail in the
// scalar loop continuing from the vector result (max is exact, so the split is
// invisible in the value).
func maxAbsF32(row []float32) float32 {
	n := len(row) &^ 15
	var m float32
	if n > 0 {
		m = maxAbsF32NEON(&row[0], n)
	}
	if n < len(row) {
		m = maxAbsF32Scalar(row[n:], m)
	}
	return m
}

// quantizeRowScaled is the round-and-clamp pass: the 16-aligned prefix in NEON,
// the tail scalar. Every production row (K = 768…8960) is a multiple of 16, so the
// tail exists for odd test shapes. The bounds check up front is the one the scalar
// loop's q[j] would have raised; the kernel takes raw pointers and must not be
// handed a q shorter than row.
func quantizeRowScaled(row []float32, q []int8, inv float32) {
	if len(row) == 0 {
		return
	}
	_ = q[len(row)-1]
	n := len(row) &^ 15
	if n > 0 {
		quantizeF32NEON(&row[0], &q[0], n, inv)
	}
	if n < len(row) {
		quantizeRowScaledScalar(row[n:], q[n:], inv)
	}
}
