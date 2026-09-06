//go:build arm64

package linalg

// arm64 impls: the NEON kernels, with the n%4 tail on the scalar contract.

func expContractImpl(dst, src []float32) {
	n4 := len(src) &^ 3
	// The kernel has no range guards; anything outside them takes the scalar
	// path, which is where ExpF32's ±Inf/NaN/overflow behaviour lives.
	safe := n4 > 0
	for i := 0; i < n4 && safe; i++ {
		if v := src[i]; v != v || v > expOverflowF32 || v < expUnderflowF32 {
			safe = false
		}
	}
	if safe {
		expF32ContractNEON(&dst[0], &src[0], n4)
	} else {
		expContractScalarInto(dst[:n4], src[:n4])
	}
	expContractScalarInto(dst[n4:], src[n4:])
}

func softmaxContractImpl(dst, src []float32) { softmaxRowContractNEON(dst, src) }

func siluContractImpl(dst, src []float32) {
	n4 := len(src) &^ 3
	if n4 > 0 {
		siluF32ContractNEON(&dst[0], &src[0], n4)
	}
	siluContractScalarInto(dst[n4:], src[n4:])
}
