//go:build arm64

package linalg

import "math"

//go:noescape
func cvtBF16ToF32x8NEON(dst *float32, src *uint16, n8 int)

// widenBF16 widens src into dst[:len(src)], exactly: eight per SHLL / SHLL2 pair (a 16-bit pattern shifted into the high half of a 32-bit lane),
// sixteen per iteration while two blocks remain, the last block overlapping so no scalar tail is needed once there are eight or more. The arm64
// twin of the amd64 AVX2 widen (widen_bf16_amd64.go); before it, arm64 took the scalar loop of widen_bf16_other.go.
func widenBF16(dst []float32, src []uint16) {
	n := len(src)
	dst = dst[:n]
	if n < 8 {
		for i, u := range src {
			dst[i] = math.Float32frombits(uint32(u) << 16)
		}
		return
	}
	n8 := n / 8
	cvtBF16ToF32x8NEON(&dst[0], &src[0], n8)
	if n8*8 < n {
		cvtBF16ToF32x8NEON(&dst[n-8], &src[n-8], 1)
	}
}
