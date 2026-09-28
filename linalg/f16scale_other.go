//go:build !amd64

package linalg

// dotW4A8F16Row is dotW4A8 over f16 scales: widened into buf (at least len(scales) floats), then dotW4A8.
func dotW4A8F16Row(act []int8, packed []byte, scales []uint16, group, K int, buf []float32) float32 {
	nG := len(scales)
	widenF16(buf[:nG], scales)
	return dotW4A8(act, packed, buf[:nG], group, K)
}
