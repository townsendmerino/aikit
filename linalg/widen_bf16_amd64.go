//go:build amd64

package linalg

import "math"

//go:noescape
func cvtBF16ToF32x8AVX2(dst *float32, src *uint16, n8 int)

// widenBF16 widens src into dst[:len(src)], exactly: eight per VPMOVZXWD + VPSLLD, the last block overlapping so no scalar tail is needed
// once there are eight or more.
func widenBF16(dst []float32, src []uint16) {
	n := len(src)
	dst = dst[:n]
	if n < 8 || !hasAVX2 {
		for i, u := range src {
			dst[i] = math.Float32frombits(uint32(u) << 16)
		}
		return
	}
	n8 := n / 8
	cvtBF16ToF32x8AVX2(&dst[0], &src[0], n8)
	if n8*8 < n {
		cvtBF16ToF32x8AVX2(&dst[n-8], &src[n-8], 1)
	}
}
