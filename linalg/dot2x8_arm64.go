//go:build arm64

package linalg

//go:noescape
func dotNEON2x8(a0, a1, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n4 int, sums *[64]float32)
