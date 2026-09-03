//go:build !arm64

package linalg

// The activation quantizer's two passes on arches without a vector kernel: the
// scalar reference. arm64 has the NEON pair in quant_act_arm64.go; amd64 still
// runs scalar here (the AVX2 form — VMULPS + a round-half-away via
// x+copysign(0.5,x) truncation — is task-simd-audit.md S-03's open amd64 half).

func maxAbsF32(row []float32) float32 { return maxAbsF32Scalar(row, 0) }

func quantizeRowScaled(row []float32, q []int8, inv float32) {
	quantizeRowScaledScalar(row, q, inv)
}
