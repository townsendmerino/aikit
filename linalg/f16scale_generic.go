//go:build !amd64 && !arm64

package linalg

// widenF16 widens src into dst (equal lengths) with the scalar f16ToF32. amd64 (F16C) and arm64 (FCVTL) have
// SIMD twins in f16scale_amd64.go and f16scale_arm64.go.
func widenF16(dst []float32, src []uint16) {
	for i, h := range src {
		dst[i] = f16ToF32(h)
	}
}
