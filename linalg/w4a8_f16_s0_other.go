//go:build !amd64

package linalg

func dotW4A8F16(act []int8, packed []byte, scales []uint16, group, K int) float32 {
	return dotW4A8F16Go(act, packed, scales, group, K)
}
