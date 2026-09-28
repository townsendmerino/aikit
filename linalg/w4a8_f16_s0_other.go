//go:build !amd64

package linalg

func dotW4A8F16(act []int8, packed []byte, scales []uint16, group, K int) float32 {
	return dotW4A8F16Go(act, packed, scales, group, K)
}

func SetInt4F16PerGroupS0(on bool) {}

var f16S0PerGroup = true // the Go fallback converts per call anyway

func widenScalesF16(dst []float32, src []uint16) {
	for i, h := range src {
		dst[i] = f16ToF32(h)
	}
}

func dotW4A8F16Row(act []int8, packed []byte, scales []uint16, group, K int, buf []float32) float32 {
	return dotW4A8F16(act, packed, scales, group, K)
}
