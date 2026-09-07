//go:build amd64

package linalg

// amd64 impls. The AVX2 kernels require FMA3, which hasAVX2 already checks (see
// detectAVX2), because the contract needs a TRUE fused multiply-add and AVX2
// alone cannot provide one. Without it the scalar contract runs — and since the
// two are bit-identical, that is a speed fallback and never a numeric one.
//
// This is also the answer to S-08's "GOAMD64 is unpinned" finding, at least for
// these kernels: the choice is made at RUNTIME, and a v1 build and a v3 build
// produce identical output. Nothing has to be mandated on consumers.

func expContractImpl(dst, src []float32) {
	if !hasAVX2 {
		expContractScalarInto(dst, src)
		return
	}
	n8 := len(src) &^ 7
	// The kernel has no range guards, so anything outside them — NaN, ±Inf,
	// past overflow or underflow — takes the scalar path, where ExpF32's
	// special-case behaviour lives.
	safe := n8 > 0
	for i := 0; i < n8 && safe; i++ {
		if v := src[i]; v != v || v > expOverflowF32 || v < expUnderflowF32 {
			safe = false
		}
	}
	if safe {
		expF32ContractAVX2(&dst[0], &src[0], n8)
	} else {
		expContractScalarInto(dst[:n8], src[:n8])
	}
	expContractScalarInto(dst[n8:], src[n8:])
}

// siluContractImpl and softmaxContractImpl reuse the AVX2 EXP and keep the
// surrounding arithmetic in Go, pinned to the same order the scalar reference
// uses. That needs no further assembly and is bit-identical by construction: the
// only thing that changed is which code computed the exponentials, and that part
// is already gated bit-for-bit.
//
// The exponential is the expensive term by a wide margin (11.8x over f64
// math.Exp), so this captures most of the available win for a fraction of the
// kernel surface.

func siluContractImpl(dst, src []float32) {
	if !hasAVX2 {
		siluContractScalarInto(dst, src)
		return
	}
	var buf [256]float32
	for off := 0; off < len(src); off += len(buf) {
		n := min(len(buf), len(src)-off)
		s := src[off : off+n]
		for i, v := range s {
			t := -v
			if t < expClampLoF32 {
				t = expClampLoF32
			}
			if t > expClampHiF32 {
				t = expClampHiF32
			}
			buf[i] = t
		}
		if n8 := n &^ 7; n8 > 0 {
			expF32ContractAVX2(&buf[0], &buf[0], n8)
		}
		for i := n &^ 7; i < n; i++ {
			buf[i] = expF32Contract(buf[i])
		}
		for i, v := range s {
			dst[off+i] = v / (1 + buf[i])
		}
	}
}

func softmaxContractImpl(dst, src []float32) {
	if !hasAVX2 || len(src) == 0 {
		softmaxRowContract(dst, src)
		return
	}
	m := src[0]
	for _, v := range src[1:] {
		if v > m {
			m = v
		}
	}
	// dst doubles as scratch. Safe when dst aliases src: each index is read
	// before it is written.
	for i, v := range src {
		d := v - m
		if d < expClampLoF32 {
			d = expClampLoF32
		}
		dst[i] = d
	}
	n8 := len(dst) &^ 7
	if n8 > 0 {
		expF32ContractAVX2(&dst[0], &dst[0], n8)
	}
	for i := n8; i < len(dst); i++ {
		dst[i] = expF32Contract(dst[i])
	}
	// The pinned order, reproduced exactly: element i into partials[i&3], folded
	// by the fixed tree. This is the half that would silently diverge if it were
	// left to a plain sequential sum.
	var p [4]float64
	for i, e := range dst {
		p[i&3] += float64(e)
	}
	sum := (p[0] + p[1]) + (p[2] + p[3])
	if sum == 0 {
		u := float32(1) / float32(len(src))
		for i := range dst {
			dst[i] = u
		}
		return
	}
	inv := float32(1 / sum)
	for i := range dst {
		dst[i] *= inv
	}
}

// geluTanhContractImpl and geluContractImpl route the TANH and the ERF through
// their AVX2 kernels and keep the wrapper arithmetic in Go, exactly as the arm64
// impls do — the wrappers are a handful of multiplies and at most one fused op,
// so putting them in assembly would double the kernel surface for a small share
// of the cost.
//
// Chunked through a stack buffer, which buys two things at once: no allocation,
// and correctness when dst ALIASES src. Each chunk reads its source values before
// writing the corresponding dst elements, and later chunks are untouched until
// their turn, so in-place use is safe.

func geluTanhContractImpl(dst, src []float32) {
	if !hasAVX2 {
		geluTanhScalarInto(dst, src)
		return
	}
	const c = 0.7978845608028654 // √(2/π)
	var buf [256]float32
	for off := 0; off < len(src); off += len(buf) {
		n := min(len(buf), len(src)-off)
		s := src[off : off+n]
		for i, v := range s {
			x3 := mulRound32(mulRound32(v, v), v)
			buf[i] = mulRound32(c, fma32(0.044715, x3, v))
		}
		if n8 := n &^ 7; n8 > 0 {
			tanhF32ContractAVX2(&buf[0], &buf[0], n8)
		}
		for i := n &^ 7; i < n; i++ {
			buf[i] = tanhF32Contract(buf[i])
		}
		for i, v := range s {
			dst[off+i] = mulRound32(mulRound32(0.5, v), 1+buf[i])
		}
	}
}

func geluContractImpl(dst, src []float32) {
	if !hasAVX2 {
		geluScalarInto(dst, src)
		return
	}
	const invSqrt2 = 0.7071067811865476
	var buf [256]float32
	for off := 0; off < len(src); off += len(buf) {
		n := min(len(buf), len(src)-off)
		s := src[off : off+n]
		for i, v := range s {
			buf[i] = mulRound32(invSqrt2, v)
		}
		if n8 := n &^ 7; n8 > 0 {
			erfF32ContractAVX2(&buf[0], &buf[0], n8)
		}
		for i := n &^ 7; i < n; i++ {
			buf[i] = erfF32Contract(buf[i])
		}
		for i, v := range s {
			dst[off+i] = mulRound32(mulRound32(0.5, v), 1+buf[i])
		}
	}
}
