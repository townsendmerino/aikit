//go:build !amd64 && !arm64

package linalg

import "math"

// widenBF16 widens src into dst[:len(src)], exactly. Scalar here; amd64 has an AVX2 form (widen_bf16_amd64.go) and arm64 a NEON one (widen_bf16_arm64.go).
func widenBF16(dst []float32, src []uint16) {
	dst = dst[:len(src)]
	for i, u := range src {
		dst[i] = math.Float32frombits(uint32(u) << 16)
	}
}
