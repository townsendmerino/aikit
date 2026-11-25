//go:build !goexperiment.simd

package elem

// softmaxRowIntoRaw is SoftmaxRowInto (exp.go) with the length/empty checks
// already done. This is the default build — byte-for-byte the body
// SoftmaxRowInto had before exp_simd.go existed; see that file for the
// GOEXPERIMENT=simd alternative.
func softmaxRowIntoRaw(dst, src []float32) {
	maxV := src[0]
	for _, v := range src[1:] {
		if v > maxV {
			maxV = v
		}
	}
	var sum float64
	for i, v := range src {
		// v-maxV <= 0 by construction, so only the underflow end is reachable
		// and expF32Core's guards suffice.
		e := expF32Core(v - maxV)
		dst[i] = e
		sum += float64(e)
	}
	if sum == 0 {
		u := 1.0 / float32(len(dst))
		for i := range dst {
			dst[i] = u
		}
		return
	}
	inv := float32(1.0 / sum)
	for i := range dst {
		dst[i] *= inv
	}
}

// siluIntoRaw is SiLUInto (exp.go) with the length/empty checks already done.
// This is the default build — byte-for-byte the loop SiLUInto had before
// exp_simd.go's siluIntoRaw existed; see that file for the GOEXPERIMENT=simd
// alternative.
func siluIntoRaw(dst, src []float32) {
	for i, v := range src {
		dst[i] = SiLUF32(v)
	}
}

// geluIntoRaw is GELUInto (exp.go) with the length/empty checks already
// done. This is the default build — byte-for-byte the loop GELUInto had
// before exp_simd.go's geluIntoRaw existed; see that file for the
// GOEXPERIMENT=simd alternative.
func geluIntoRaw(dst, src []float32) {
	for i, v := range src {
		dst[i] = GELUF32(v)
	}
}

// tanhIntoRaw is TanhInto (exp.go) with the length/empty checks already
// done. This is the default build — byte-for-byte the loop TanhInto had
// before exp_simd.go's tanhIntoRaw existed; see that file for the
// GOEXPERIMENT=simd alternative.
func tanhIntoRaw(dst, src []float32) {
	for i, v := range src {
		dst[i] = TanhF32(v)
	}
}

// geluTanhIntoRaw is GELUTanhInto (exp.go) with the length/empty checks
// already done. This is the default build — byte-for-byte the loop
// GELUTanhInto had before exp_simd.go's geluTanhIntoRaw existed; see that
// file for the GOEXPERIMENT=simd alternative.
func geluTanhIntoRaw(dst, src []float32) {
	for i, v := range src {
		dst[i] = GELUTanhF32(v)
	}
}

// softmaxRowScaledIntoRaw is SoftmaxRowScaledInto (exp.go) with the
// length/empty checks already done. Default build: still two passes — the
// explicit scale multiply, then the existing softmaxRowIntoRaw — matching
// dead-ends §4.4's own finding that fusing them isn't worth it at scalar
// math.Exp speeds (the scale pass is ~2% of the cost here). This is exactly
// what callers used to do at the call site before this function existed;
// moving it here changes where the two passes live, not what they cost.
func softmaxRowScaledIntoRaw(dst, src []float32, scale float32) {
	for i, v := range src {
		dst[i] = v * scale
	}
	softmaxRowIntoRaw(dst, dst)
}
