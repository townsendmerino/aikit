//go:build !arm64 && !amd64

package linalg

// dotW4A8 computes one W4A8 output (before the activation scale). arm64 (SDOT)
// and amd64 (AVX2) have fused kernels; every other target routes to the portable
// scalar reference here.
func dotW4A8(act []int8, packed []byte, scales []float32, group, K int) float32 {
	return dotW4A8Scalar(act, packed, scales, group, K)
}
