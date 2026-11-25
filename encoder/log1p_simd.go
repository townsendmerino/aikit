//go:build goexperiment.simd

package encoder

import "simd"

// log1pPoolInto (SIMD build) — see log1p_scalar.go for the default-build
// body. Applies log1p unconditionally to every element (no `x > 0` mask
// needed): pooled is seeded at 0 and only ever raised by max, so it is never
// negative, and log1pF32Core(0) == 0 exactly by construction (verified by
// TestLog1pF32Core_matchesMathLog1p), matching the default build's `x > 0`
// skip — which was always a call-count optimization, not a correctness
// requirement.
//
// log1pF32Core is a NEW kernel (not a vectorization of an existing aikit
// function, unlike items 7/8) — Go 1.27's simd package ships zero
// transcendentals, so this ports Cephes' verified single-precision logf
// algorithm (single/logf.c, S. Moshier) rather than inventing coefficients:
// small x (< 2^-12) uses log1p(x) ~= x - 0.5x^2 directly (the x^3/3 term is
// below float32 precision there — verified empirically, see the accuracy
// test); larger x computes u = 1+x (safe: x is never tiny in this branch,
// so no cancellation), decomposes u's IEEE-754 bits into mantissa m in
// [0.5, 1) and exponent e (frexp-equivalent), reduces m into [-0.293, 0.414)
// exactly as Cephes does (threshold sqrt(2)/2), evaluates Cephes' verified
// 9-coefficient polynomial, and reconstructs via the same ln2-hi/lo split
// (0.693359375 + -2.12194440e-4 = ln2 to float32 precision) Cephes and Go's
// own src/math/log1p.go both use — independently cross-checked reduction
// bounds (Sqrt2M1/Sqrt2HalfM1 in Go's stdlib match Cephes' SQRTHF exactly).
func log1pPoolInto(pooled []float32) {
	c := newLog1pSIMDConsts()
	var probe simd.Float32s
	L := probe.Len()
	i := 0
	for ; i+L <= len(pooled); i += L {
		log1pF32CoreVec(simd.LoadFloat32s(pooled[i:i+L]), &c).Store(pooled[i : i+L])
	}
	if i < len(pooled) {
		storeF32Part(log1pF32CoreVec(loadF32Part(pooled[i:]), &c), pooled[i:])
	}
}

type log1pSIMDConsts struct {
	one, half, threshold, sqrtHF                     simd.Float32s
	p0, p1, p2, p3, p4, p5, p6, p7, p8, ln2Hi, ln2Lo simd.Float32s
	expMask, expFieldMask, i126U                     simd.Uint32s
	i126, oneI, zeroI32                              simd.Int32s
}

func newLog1pSIMDConsts() log1pSIMDConsts {
	return log1pSIMDConsts{
		one:          simd.BroadcastFloat32s(1.0),
		half:         simd.BroadcastFloat32s(0.5),
		threshold:    simd.BroadcastFloat32s(1.0 / (1 << 12)),
		sqrtHF:       simd.BroadcastFloat32s(0.70710678),
		p0:           simd.BroadcastFloat32s(7.0376836292e-2),
		p1:           simd.BroadcastFloat32s(-1.1514610310e-1),
		p2:           simd.BroadcastFloat32s(1.1676998740e-1),
		p3:           simd.BroadcastFloat32s(-1.2420140846e-1),
		p4:           simd.BroadcastFloat32s(1.4249322787e-1),
		p5:           simd.BroadcastFloat32s(-1.6668057665e-1),
		p6:           simd.BroadcastFloat32s(2.0000714765e-1),
		p7:           simd.BroadcastFloat32s(-2.4999993993e-1),
		p8:           simd.BroadcastFloat32s(3.3333331174e-1),
		ln2Hi:        simd.BroadcastFloat32s(0.693359375),
		ln2Lo:        simd.BroadcastFloat32s(-2.12194440e-4),
		expMask:      simd.BroadcastUint32s(0xFF),
		expFieldMask: simd.BroadcastUint32s(0x7F800000), // exponent field, cleared via AndNot
		i126U:        simd.BroadcastUint32s(126),
		i126:         simd.BroadcastInt32s(126),
		oneI:         simd.BroadcastInt32s(1),
		zeroI32:      simd.BroadcastInt32s(0),
	}
}

// log1pF32CoreVec is log1pF32Core (log1p_scalar.go's algorithm, see that
// file's doc comment), vectorized.
func log1pF32CoreVec(x simd.Float32s, c *log1pSIMDConsts) simd.Float32s {
	// small-x branch: x - 0.5*x*x
	smallResult := x.Sub(c.half.Mul(x).Mul(x))

	// large-x branch: u = x+1, decompose bits (frexp-equivalent).
	u := x.Add(c.one)
	bits := u.ToBits()
	eField := bits.ShiftAllRight(23).And(c.expMask)
	eInit := eField.ConvertToInt32().Sub(c.i126)

	mBits := bits.AndNot(c.expFieldMask).Or(c.i126U.ShiftAllLeft(23))
	m := mBits.BitsToFloat32()

	lowMask := m.Less(c.sqrtHF)
	eLow := eInit.Sub(c.oneI)
	mLow := m.Add(m).Sub(c.one)
	mHigh := m.Sub(c.one)

	e := eLow.IfElse(lowMask, eInit)
	mr := mLow.IfElse(lowMask, mHigh)

	z := mr.Mul(mr)
	p := c.p0.MulAdd(mr, c.p1)
	p = p.MulAdd(mr, c.p2)
	p = p.MulAdd(mr, c.p3)
	p = p.MulAdd(mr, c.p4)
	p = p.MulAdd(mr, c.p5)
	p = p.MulAdd(mr, c.p6)
	p = p.MulAdd(mr, c.p7)
	p = p.MulAdd(mr, c.p8)
	y := p.Mul(mr).Mul(z)

	neZero := e.NotEqual(c.zeroI32)
	fe := e.ConvertToFloat32()
	y = y.Add(c.ln2Lo.Mul(fe).Masked(neZero))
	y = y.Sub(c.half.Mul(z))
	largeResult := mr.Add(y)
	largeResult = largeResult.Add(c.ln2Hi.Mul(fe).Masked(neZero))

	smallMask := x.Less(c.threshold)
	return smallResult.IfElse(smallMask, largeResult)
}
