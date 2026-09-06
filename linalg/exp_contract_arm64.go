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

// geluTanhContractImpl routes only the TANH through the kernel. The wrapper
// arithmetic around it — x³, the inner scaling, the final 0.5·x·(1+t) — is a
// handful of multiplies and exactly ONE fused op, so it stays in Go rather than
// doubling the assembly for a small share of the cost.
//
// Chunked through a stack buffer, which does two things at once: no allocation,
// and correctness when dst ALIASES src. Each chunk reads its source values into
// the loop variable before writing the corresponding dst element, and later
// chunks are untouched until their turn, so in-place use is safe.
func geluTanhContractImpl(dst, src []float32) {
	const c = 0.7978845608028654 // √(2/π)
	var buf [256]float32
	for off := 0; off < len(src); off += len(buf) {
		n := min(len(buf), len(src)-off)
		s := src[off : off+n]
		for i, v := range s {
			x3 := mulRound32(mulRound32(v, v), v)
			buf[i] = mulRound32(c, fma32(0.044715, x3, v))
		}
		if n4 := n &^ 3; n4 > 0 {
			tanhF32ContractNEON(&buf[0], &buf[0], n4)
		}
		for i := n &^ 3; i < n; i++ {
			buf[i] = tanhF32Contract(buf[i])
		}
		for i, v := range s {
			dst[off+i] = mulRound32(mulRound32(0.5, v), 1+buf[i])
		}
	}
}

// geluContractImpl routes the ERF through the kernel; the wrapper is one scaling
// multiply in and two out, with nothing to fuse into, so it stays in Go. Chunked
// through a stack buffer for the same two reasons as the tanh form: no
// allocation, and correctness when dst aliases src.
func geluContractImpl(dst, src []float32) {
	const invSqrt2 = 0.7071067811865476
	var buf [256]float32
	for off := 0; off < len(src); off += len(buf) {
		n := min(len(buf), len(src)-off)
		s := src[off : off+n]
		for i, v := range s {
			buf[i] = mulRound32(invSqrt2, v)
		}
		if n4 := n &^ 3; n4 > 0 {
			erfF32ContractNEON(&buf[0], &buf[0], n4)
		}
		for i := n &^ 3; i < n; i++ {
			buf[i] = erfF32Contract(buf[i])
		}
		for i, v := range s {
			dst[off+i] = mulRound32(mulRound32(0.5, v), 1+buf[i])
		}
	}
}
