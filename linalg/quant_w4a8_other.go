//go:build !arm64

package linalg

// dotW4A8 computes one W4A8 output (before the activation scale). No SDOT kernel
// off arm64 yet (the amd64 AVX2/VNNI fused kernel is a follow-up validated on the
// Linux box), so this routes to the portable scalar reference; sums is unused.
func dotW4A8(act []int8, packed []byte, scales []float32, group, K int, sums []int32) float32 {
	return dotW4A8Scalar(act, packed, scales, group, K)
}
