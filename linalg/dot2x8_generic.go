//go:build !arm64

package linalg

import "unsafe"

// dotNEON2x8 is the portable fallback for the 2×8 microkernel. amd64 keeps its
// AVX2 Dot8x4 path in the encoder GEMM (the 2×8 wiring is gated to arm64), so this
// exists only to keep Dot2x8 callable and tested on every arch.
func dotNEON2x8(a0, a1, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n4 int, sums *[64]float32) {
	bs := [8]*float32{b0, b1, b2, b3, b4, b5, b6, b7}
	as := [2]*float32{a0, a1}
	for i := range sums {
		sums[i] = 0
	}
	n := n4 * 4
	for ai := 0; ai < 2; ai++ {
		arow := unsafe.Slice(as[ai], n)
		for bi := 0; bi < 8; bi++ {
			brow := unsafe.Slice(bs[bi], n)
			var s float32
			for k := 0; k < n; k++ {
				s += arow[k] * brow[k]
			}
			sums[(ai*8+bi)*4] = s // reduced into lane 0, like Dot8x4's contract
		}
	}
}
