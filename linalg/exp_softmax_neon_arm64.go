//go:build arm64

package linalg

// expBiasSumF32NEON computes dst[i] = exp(src[i]+bias) for n elements (n a
// multiple of 4) and accumulates them into four float64 lane partials: element i
// into partials[i%4]. The caller folds with the fixed tree (p0+p1)+(p2+p3).
//
// The split is the softmax contract: a denominator is a reduction, so its value
// depends on the order of the adds, and pinning that order is what lets a vector
// path and a scalar path be bit-identical rather than merely close.
//
//go:noescape
func expBiasSumF32NEON(dst, src *float32, n int, bias float32, partials *float64)

// softmaxRowContractNEON is softmaxRowContract with the exponential and the
// denominator's lane partials computed by the kernel. Bit-identical to the
// scalar reference, not approximately: same max, same bias, same clamp, same
// four partials in the same order, same fold tree.
//
// The n%4 tail runs the scalar path, and lands in the SAME partial it would have
// landed in — element i goes to partials[i&3] on both sides, and n&^3 is a
// multiple of 4 so the tail's indices continue the pattern rather than restarting
// it. That is the detail a "vector body plus scalar tail" split usually gets
// wrong.
func softmaxRowContractNEON(dst, src []float32) {
	if len(dst) != len(src) {
		panic("linalg: softmaxRowContractNEON length mismatch")
	}
	if len(src) == 0 {
		return
	}
	m := src[0]
	n := len(src)
	_ = src[n-1]
	i := 1
	for ; i+3 < n; i += 4 {
		v0 := src[i+0]
		v1 := src[i+1]
		v2 := src[i+2]
		v3 := src[i+3]
		if v0 > m {
			m = v0
		}
		if v1 > m {
			m = v1
		}
		if v2 > m {
			m = v2
		}
		if v3 > m {
			m = v3
		}
	}
	for ; i < n; i++ {
		if v := src[i]; v > m {
			m = v
		}
	}
	softmaxRowContractNEONWithMax(dst, src, m)
}

func softmaxRowContractNEONWithMax(dst, src []float32, m float32) {
	if len(dst) != len(src) {
		panic("linalg: softmaxRowContractNEON length mismatch")
	}
	if len(src) == 0 {
		return
	}
	var p [4]float64
	n4 := len(src) &^ 3
	if n4 > 0 {
		expBiasSumF32NEON(&dst[0], &src[0], n4, -m, &p[0])
	}
	for i := n4; i < len(src); i++ {
		d := src[i] - m
		if d < expClampLoF32 {
			d = expClampLoF32
		}
		e := expF32Contract(d)
		dst[i] = e
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
	nd := len(dst)
	if nd > 0 {
		_ = dst[nd-1]
		d := 0
		for ; d+7 < nd; d += 8 {
			dst[d+0] *= inv
			dst[d+1] *= inv
			dst[d+2] *= inv
			dst[d+3] *= inv
			dst[d+4] *= inv
			dst[d+5] *= inv
			dst[d+6] *= inv
			dst[d+7] *= inv
		}
		for ; d < nd; d++ {
			dst[d] *= inv
		}
	}
}
