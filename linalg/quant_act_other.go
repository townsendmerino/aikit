//go:build !arm64 && !amd64

package linalg

// The activation quantizer's two passes on arches without a vector kernel: the
// scalar reference. arm64 has the NEON pair in quant_act_arm64.go; amd64 has the
// AVX2 pair in quant_act_amd64.go (task-simd-audit.md S-03's amd64 half, shipped
// 2026-09-28).

func maxAbsF32(row []float32) float32 { return maxAbsF32Scalar(row, 0) }

func quantizeRowScaled(row []float32, q []int8, inv float32) {
	quantizeRowScaledScalar(row, q, inv)
}
