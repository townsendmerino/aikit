//go:build amd64

package linalg

//go:noescape
func cvtF16ToF32x8F16C(dst *float32, src *uint16, n8 int)

//go:noescape
func dotW4A8FoldF16RowAVX2(act *int8, packed *byte, scales *uint16, nGroups int, buf *float32) float32

var hasF16C = detectF16C()

// widenF16 widens src into dst (equal lengths): eight per VCVTPH2PS, the last block overlapping so no
// scalar tail is needed once there are eight or more.
func widenF16(dst []float32, src []uint16) {
	n := len(src)
	if n < 8 || !hasF16C {
		for i, h := range src {
			dst[i] = f16ToF32(h)
		}
		return
	}
	n8 := n / 8
	cvtF16ToF32x8F16C(&dst[0], &src[0], n8)
	if n8*8 < n {
		cvtF16ToF32x8F16C(&dst[n-8], &src[n-8], 1)
	}
}

// dotW4A8F16Row is dotW4A8 over f16 scales. buf is caller scratch of at least K/group floats. On an AVX2 +
// F16C host without VNNI (the host dotW4A8 runs dotW4A8FoldAVX2 on) it is one asm call that widens the
// row's scales and runs that kernel's loop; otherwise the scales are widened into buf and dotW4A8 runs.
// Bit-identical to dotW4A8 on the widened values either way.
func dotW4A8F16Row(act []int8, packed []byte, scales []uint16, group, K int, buf []float32) float32 {
	nG := len(scales)
	if group == 32 && K%32 == 0 && nG == K/32 && nG >= 8 && nG <= len(buf) && hasAVX2 && hasF16C && !hasAVX512VNNIVL {
		return dotW4A8FoldF16RowAVX2(&act[0], &packed[0], &scales[0], nG, &buf[0])
	}
	widenF16(buf[:nG], scales)
	return dotW4A8(act, packed, buf[:nG], group, K)
}
