//go:build arm64

package linalg

//go:noescape
func cvtF16ToF32x8NEON(dst *float32, src *uint16, n8 int)

// widenF16 widens src into dst (equal lengths): eight per FCVTL / FCVTL2 pair, the last block overlapping so
// no scalar tail is needed once there are eight or more — the arm64 twin of the amd64 F16C widen.
//
// Until 2026-09-28 arm64 took the portable scalar loop (f16scale_generic.go), which put a branchy per-scale
// conversion on every quad of every M=1 row4 decode matmul. On an M1 Pro that measured 0.544× / 0.554× /
// 0.435× of the f32-scale build's served CPU decode (qwen2.5 0.5B / 1.5B / 7B; goinfer
// docs/tasks/task-cpu-decode-peer-gap-2026-09.md, "L1 build: the Mac half").
func widenF16(dst []float32, src []uint16) {
	n := len(src)
	if n < 8 {
		for i, h := range src {
			dst[i] = f16ToF32(h)
		}
		return
	}
	_ = dst[n-1]
	n8 := n / 8
	cvtF16ToF32x8NEON(&dst[0], &src[0], n8)
	if n8*8 < n {
		cvtF16ToF32x8NEON(&dst[n-8], &src[n-8], 1)
	}
}
