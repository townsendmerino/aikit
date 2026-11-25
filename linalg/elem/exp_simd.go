//go:build goexperiment.simd

package elem

import (
	"math"

	"simd"
)

// softmaxRowIntoRaw (SIMD build) — perf-campaign item 13, validated as a
// prototype (~/tmcode/go127-simd-audit, 2026-08-20) before landing here: on
// apple-m1pro (128-bit NEON) softmax measured ~2.5-2.6x faster than the
// scalar build; on nvidia-rtx2070s (Ryzen 7 3700X, AVX2) ~3.0-3.2x at the
// default 128-bit width and ~3.7-4.6x with GODEBUG=simd='+256' (AVX2's
// native width — Go 1.27's simd package defaults conservatively to 128-bit
// even on AVX2 hardware; 256 needs the explicit override past a safety
// check). Both are CPU-vs-CPU numbers — scalar Go arithmetic vs vector CPU
// instructions (NEON / AVX2) — nothing to do with the GPU (CUDA/Metal)
// kernels, which are a separate code path entirely.
//
// Vectorizes the independent axes only (max-scan, exp, scale); the f64 sum
// stays a scalar sequential fold in the SAME order the non-SIMD build uses
// (exp_scalar.go) — the one part of softmax that is a genuine reduction, per
// this codebase's standing rule that reductions re-tree and elementwise
// passes don't (dead-ends §8.4 applied the same discipline to layerNorm).
//
// NOT bit-identical to the non-SIMD build: expF32CoreVec's MulAdd is a fused
// op, and Go's scalar compiler already fuses p*r+c on arm64 but not amd64 —
// so this can differ from the scalar softmaxRowIntoRaw by up to 1 ULP per
// exponential (TestExpF32CoreVec_matchesScalarULP gates the bound). That is
// strictly a GOEXPERIMENT=simd-only difference: the default build (this
// file excluded) is untouched.
func softmaxRowIntoRaw(dst, src []float32) {
	n := len(src)
	c := newExpSIMDConsts()
	var probe simd.Float32s
	L := probe.Len()

	// Pass 1: max — same scalar fold the non-SIMD kernel uses over the lane
	// partials, so maxV is bit-identical to it (max is order-invariant absent
	// NaN, which is outside softmax's contract either way).
	maxV := src[0]
	i := 0
	if n >= L {
		m := simd.LoadFloat32s(src[0:L])
		for i = L; i+L <= n; i += L {
			m = m.Max(simd.LoadFloat32s(src[i : i+L]))
		}
		var buf [16]float32 // Len() <= 512/32 = 16
		m.Store(buf[:L])
		maxV = buf[0]
		for _, v := range buf[1:L] {
			if v > maxV {
				maxV = v
			}
		}
	}
	for ; i < n; i++ { // tail (and the whole row when n < L)
		if v := src[i]; v > maxV {
			maxV = v
		}
	}

	// Pass 2: exp(v - max), vectorized.
	bcMax := simd.BroadcastFloat32s(maxV)
	i = 0
	for ; i+L <= n; i += L {
		expF32CoreVec(simd.LoadFloat32s(src[i:i+L]).Sub(bcMax), &c).Store(dst[i : i+L])
	}
	if i < n {
		// The part load fills the tail lanes with zero — in-domain (e^0=1) —
		// and the part store writes back only the real elements.
		v := loadF32Part(src[i:])
		storeF32Part(expF32CoreVec(v.Sub(bcMax), &c), dst[i:])
	}

	// Pass 3: f64 sum — scalar on purpose, same tree as the non-SIMD kernel.
	var sum float64
	for _, e := range dst {
		sum += float64(e)
	}
	if sum == 0 {
		u := 1.0 / float32(n)
		for j := range dst {
			dst[j] = u
		}
		return
	}

	// Pass 4: scale.
	inv := simd.BroadcastFloat32s(float32(1.0 / sum))
	i = 0
	for ; i+L <= n; i += L {
		simd.LoadFloat32s(dst[i : i+L]).Mul(inv).Store(dst[i : i+L])
	}
	if i < n {
		v := loadF32Part(dst[i:])
		storeF32Part(v.Mul(inv), dst[i:])
	}
}

// softmaxRowScaledIntoRaw is SoftmaxRowScaledInto (exp.go) with the
// length/empty checks already done — T2 (dead-ends §4.4, reopened after
// item 7's SIMD exp). The SAME four passes softmaxRowIntoRaw has, minus
// one: the row max is computed on the UNSCALED src (identical max-scan to
// softmaxRowIntoRaw's pass 1), scaled by ONE scalar multiply, and the
// `src[i] *= scale` a caller used to do in its own separate O(L²) pass is
// folded directly into pass 2's per-lane exp computation instead — see
// SoftmaxRowScaledInto's doc comment for why this is bit-identical to that
// two-pass sequence, not just an approximation of it.
func softmaxRowScaledIntoRaw(dst, src []float32, scale float32) {
	n := len(src)
	c := newExpSIMDConsts()
	var probe simd.Float32s
	L := probe.Len()

	// Pass 1: max over UNSCALED src — identical to softmaxRowIntoRaw's own
	// max-scan; nothing about scale enters this pass.
	maxV := src[0]
	i := 0
	if n >= L {
		m := simd.LoadFloat32s(src[0:L])
		for i = L; i+L <= n; i += L {
			m = m.Max(simd.LoadFloat32s(src[i : i+L]))
		}
		var buf [16]float32
		m.Store(buf[:L])
		maxV = buf[0]
		for _, v := range buf[1:L] {
			if v > maxV {
				maxV = v
			}
		}
	}
	for ; i < n; i++ {
		if v := src[i]; v > maxV {
			maxV = v
		}
	}
	scaledMax := maxV * scale

	// Pass 2: exp(src[i]*scale - scaledMax), vectorized. The `*scale` here
	// IS the fused-away separate pass — same single multiply a caller's own
	// loop would have done, just performed inside this one instead of a
	// pass of its own beforehand.
	bcScale := simd.BroadcastFloat32s(scale)
	bcMax := simd.BroadcastFloat32s(scaledMax)
	i = 0
	for ; i+L <= n; i += L {
		v := simd.LoadFloat32s(src[i : i+L]).Mul(bcScale).Sub(bcMax)
		expF32CoreVec(v, &c).Store(dst[i : i+L])
	}
	if i < n {
		v := loadF32Part(src[i:])
		vv := v.Mul(bcScale).Sub(bcMax)
		storeF32Part(expF32CoreVec(vv, &c), dst[i:])
	}

	// Pass 3: f64 sum — unchanged.
	var sum float64
	for _, e := range dst {
		sum += float64(e)
	}
	if sum == 0 {
		u := 1.0 / float32(n)
		for j := range dst {
			dst[j] = u
		}
		return
	}

	// Pass 4: scale (normalize) — unchanged.
	inv := simd.BroadcastFloat32s(float32(1.0 / sum))
	i = 0
	for ; i+L <= n; i += L {
		simd.LoadFloat32s(dst[i : i+L]).Mul(inv).Store(dst[i : i+L])
	}
	if i < n {
		v := loadF32Part(dst[i:])
		storeF32Part(v.Mul(inv), dst[i:])
	}
}

// expSIMDConsts holds the broadcast constants so a hot loop hoists them once.
type expSIMDConsts struct {
	log2e, magic, ln2Hi, ln2Lo             simd.Float32s
	c0, c1, c2, c3, c4, c5, one, underflow simd.Float32s
	two, overflow, posInf                  simd.Float32s
	i127, i1, i255                         simd.Int32s
}

func newExpSIMDConsts() expSIMDConsts {
	return expSIMDConsts{
		log2e:     simd.BroadcastFloat32s(log2eF32),
		magic:     simd.BroadcastFloat32s(roundMagicF32),
		ln2Hi:     simd.BroadcastFloat32s(ln2HiF32),
		ln2Lo:     simd.BroadcastFloat32s(ln2LoF32),
		c0:        simd.BroadcastFloat32s(1.9875691500e-4),
		c1:        simd.BroadcastFloat32s(1.3981999507e-3),
		c2:        simd.BroadcastFloat32s(8.3334519073e-3),
		c3:        simd.BroadcastFloat32s(4.1665795894e-2),
		c4:        simd.BroadcastFloat32s(1.6666665459e-1),
		c5:        simd.BroadcastFloat32s(5.0000001201e-1),
		one:       simd.BroadcastFloat32s(1),
		underflow: simd.BroadcastFloat32s(expUnderflowF32),
		two:       simd.BroadcastFloat32s(2),
		overflow:  simd.BroadcastFloat32s(expOverflowF32),
		posInf:    simd.BroadcastFloat32s(float32(math.Inf(1))),
		i127:      simd.BroadcastInt32s(127),
		i1:        simd.BroadcastInt32s(1),
		i255:      simd.BroadcastInt32s(255),
	}
}

// expF32CoreVec is expF32Core (exp.go), vectorized — same constants, same
// round-magic trick, same Cephes minimax chain, same op order, including the
// e>=255 two-step scale expF32Core uses for x close enough to the overflow
// bound that k reaches 128 while the true result is still finite (2^128
// would encode +Inf and poison it — see expF32Core's own comment). Valid for
// the same domain ExpF32 delegates to the scalar core: x in
// [expUnderflowF32, expOverflowF32]. x > expOverflowF32 or NaN is NOT this
// function's job — see expF32Vec, which adds those two guards on top for a
// caller (SiLU) whose argument is unbounded, unlike softmax's x <= 0.
func expF32CoreVec(x simd.Float32s, c *expSIMDConsts) simd.Float32s {
	z := x.Mul(c.log2e)
	kf := z.Add(c.magic).Sub(c.magic) // round-to-nearest-even via 1.5*2^23
	r := x.Sub(kf.Mul(c.ln2Hi)).Sub(kf.Mul(c.ln2Lo))

	p := c.c0.MulAdd(r, c.c1)
	p = p.MulAdd(r, c.c2)
	p = p.MulAdd(r, c.c3)
	p = p.MulAdd(r, c.c4)
	p = p.MulAdd(r, c.c5)
	res := p.Mul(r).Mul(r).Add(r).Add(c.one) // p*r^2 + r + 1, same op order

	// 2^k by bit construction: (k+127)<<23 reinterpreted as float32. For
	// softmax's x <= 0 domain, e is always < 255 (see the comment this used
	// to carry) and the branch below is inert; for SiLU's e^-x with x very
	// negative (e.g. x=-88), e CAN reach 255+, which is why the two-step
	// path exists at all — both branches are computed and mask-selected,
	// the two-branch pattern this codebase already uses for Erf/Tanh.
	e := kf.ConvertToInt32().Add(c.i127)
	scaleLo := e.ShiftAllLeft(23).ToBits().BitsToFloat32()
	scaleHi := e.Sub(c.i1).ShiftAllLeft(23).ToBits().BitsToFloat32()
	resLo := res.Mul(scaleLo)
	resHi := res.Mul(scaleHi).Mul(c.two)
	res = resHi.IfElse(e.GreaterEqual(c.i255), resLo)

	// Lanes with x < underflow have e <= 0; e=0 self-flushes via a zero
	// scale but e<0 shifts sign bits into garbage — mask keeps only
	// in-domain lanes (Masked zeroes where the mask is false).
	return res.Masked(x.GreaterEqual(c.underflow))
}

// expF32Vec is ExpF32 (exp.go), vectorized: expF32CoreVec plus the two guards
// it doesn't cover — overflow saturates to +Inf, NaN propagates — the full
// contract a caller with an UNBOUNDED argument needs (unlike softmax, whose
// domain after subtracting the row max is always x <= 0). This is the guard
// work ExpF32Into declined to build for its caller-less state; SiLU's real
// caller is what justifies building it now.
func expF32Vec(x simd.Float32s, c *expSIMDConsts) simd.Float32s {
	res := expF32CoreVec(x, c)
	res = c.posInf.IfElse(x.Greater(c.overflow), res)
	// NaN != NaN is the one IEEE comparison that's true for NaN and only
	// NaN, so this reproduces ExpF32's `if x != x { return x }` branch-free.
	return x.IfElse(x.NotEqual(x), res)
}

// siluIntoRaw is SiLUInto (exp.go) with the length/empty checks already done
// — SiLU's autoresearch T1 target (docs/prompts/simd-elementwise-autoresearch.md).
// x/(1+e^-x) is where item 7's SoftmaxRowInto could stay narrow: exp's
// argument here is UNBOUNDED, so this is built on expF32Vec (full guards),
// not the softmax-only expF32CoreVec. The saturating ends need no branch of
// their own — they fall out of expF32Vec's own guards exactly as the scalar
// SiLUF32's doc comment says: x far below the exp overflow bound makes
// e^-x=+Inf and x/(1+Inf) a signed zero; x far above it makes e^-x flush to
// 0 and the result exactly x (Div's IEEE semantics do the rest).
func siluIntoRaw(dst, src []float32) {
	n := len(src)
	c := newExpSIMDConsts()
	var probe simd.Float32s
	L := probe.Len()

	i := 0
	for ; i+L <= n; i += L {
		x := simd.LoadFloat32s(src[i : i+L])
		e := expF32Vec(x.Neg(), &c)
		x.Div(c.one.Add(e)).Store(dst[i : i+L])
	}
	if i < n {
		x := loadF32Part(src[i:])
		e := expF32Vec(x.Neg(), &c)
		storeF32Part(x.Div(c.one.Add(e)), dst[i:])
	}
}

// erfSIMDConsts holds ErfF32's broadcast constants — separate from
// expSIMDConsts (which erfVec still uses internally, for the tail branch's
// exp(-x^2)) so softmax/SiLU's hot path doesn't carry erf's ~20 extra
// broadcasts it never needs.
type erfSIMDConsts struct {
	zero, one, negOne, four, half, invSqrt2, erfScale      simd.Float32s
	sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8, sc9, sc10 simd.Float32s // series (|x|<1), Cephes
	tc0, tc1, tc2, tc3, tc4                                simd.Float32s // A&S tail (|x|>=1)
	a1                                                     simd.Float32s
}

func newErfSIMDConsts() erfSIMDConsts {
	return erfSIMDConsts{
		zero: simd.BroadcastFloat32s(0),
		one:  simd.BroadcastFloat32s(1), negOne: simd.BroadcastFloat32s(-1),
		four: simd.BroadcastFloat32s(4), half: simd.BroadcastFloat32s(0.5),
		invSqrt2: simd.BroadcastFloat32s(0.7071067811865476),
		erfScale: simd.BroadcastFloat32s(1.1283791670955126),
		sc0:      simd.BroadcastFloat32s(1.3122532963802806e-08),
		sc1:      simd.BroadcastFloat32s(-1.4503852223150468e-07),
		sc2:      simd.BroadcastFloat32s(1.4589169000933706e-06),
		sc3:      simd.BroadcastFloat32s(-1.3227513227513228e-05),
		sc4:      simd.BroadcastFloat32s(1.0683760683760684e-04),
		sc5:      simd.BroadcastFloat32s(-7.5757575757575758e-04),
		sc6:      simd.BroadcastFloat32s(4.6296296296296294e-03),
		sc7:      simd.BroadcastFloat32s(-2.3809523809523808e-02),
		sc8:      simd.BroadcastFloat32s(1.0000000000000001e-01),
		sc9:      simd.BroadcastFloat32s(-3.3333333333333331e-01),
		sc10:     simd.BroadcastFloat32s(1),
		tc0:      simd.BroadcastFloat32s(1.061405429),
		tc1:      simd.BroadcastFloat32s(-1.453152027),
		tc2:      simd.BroadcastFloat32s(1.421413741),
		tc3:      simd.BroadcastFloat32s(-0.284496736),
		tc4:      simd.BroadcastFloat32s(0.254829592),
		a1:       simd.BroadcastFloat32s(0.3275911),
	}
}

// erfVec is ErfF32 (exp.go), vectorized: both branches (series for |x|<1,
// A&S tail for |x|>=1) are computed unconditionally and mask-selected — the
// two-branch pattern this file's own comments describe for Erf/Tanh, now
// built. NOT full ExpF32 domain: the tail branch's exp(-x^2) argument is in
// [-16, -1] for the |x| in [1,4] it's ever evaluated at (x>4 saturates
// separately, before reaching it), comfortably inside expF32CoreVec's
// validated range with no overflow risk — so this reuses the narrow core,
// not expF32Vec, exactly as softmax does and for the same reason.
//
// NaN is NOT specially guarded, unlike expF32Vec's guard for SiLU's real
// caller: no production caller here has ever produced NaN input (GELU feeds
// on a forward pass's own activations), and the existing scalar ErfF32 has
// no stated/tested NaN contract either — this only vectorizes the domain
// both the scalar function and its real callers actually see.
func erfVec(x simd.Float32s, ec *erfSIMDConsts, xc *expSIMDConsts) simd.Float32s {
	absX := x.Abs()
	sgn := ec.negOne.IfElse(x.Less(ec.zero), ec.one)

	// Series branch, |x| < 1: erf(x) = (2/sqrt(pi))*x*P(x^2), no cancellation.
	z := absX.Mul(absX)
	p := ec.sc0.MulAdd(z, ec.sc1)
	p = p.MulAdd(z, ec.sc2)
	p = p.MulAdd(z, ec.sc3)
	p = p.MulAdd(z, ec.sc4)
	p = p.MulAdd(z, ec.sc5)
	p = p.MulAdd(z, ec.sc6)
	p = p.MulAdd(z, ec.sc7)
	p = p.MulAdd(z, ec.sc8)
	p = p.MulAdd(z, ec.sc9)
	p = p.MulAdd(z, ec.sc10)
	seriesMag := ec.erfScale.Mul(absX).Mul(p)

	// A&S 7.1.26 tail, |x| >= 1: erf ~ 1, so 1-(small) is stable.
	t := ec.one.Div(ec.one.Add(ec.a1.Mul(absX)))
	pt := ec.tc0.MulAdd(t, ec.tc1)
	pt = pt.MulAdd(t, ec.tc2)
	pt = pt.MulAdd(t, ec.tc3)
	pt = pt.MulAdd(t, ec.tc4)
	negX2 := absX.Mul(absX).Neg()
	tailMag := ec.one.Sub(pt.Mul(t).Mul(expF32CoreVec(negX2, xc)))

	mag := seriesMag.IfElse(absX.Less(ec.one), tailMag)
	mag = ec.one.IfElse(absX.Greater(ec.four), mag) // saturated: 1-erf(4) below f32 resolution
	return mag.Mul(sgn)
}

// geluIntoRaw is GELUInto (exp.go) with the length/empty checks already
// done — T1's second target, the exact (erf) GELU. Built directly on erfVec;
// no extra guard work of its own (GELU's own saturating ends, per GELUF32's
// doc comment, fall out of erf's ±1 saturation with no separate branch).
func geluIntoRaw(dst, src []float32) {
	n := len(src)
	ec := newErfSIMDConsts()
	xc := newExpSIMDConsts()
	var probe simd.Float32s
	L := probe.Len()

	i := 0
	for ; i+L <= n; i += L {
		x := simd.LoadFloat32s(src[i : i+L])
		erf := erfVec(x.Mul(ec.invSqrt2), &ec, &xc)
		ec.half.Mul(x).Mul(ec.one.Add(erf)).Store(dst[i : i+L])
	}
	if i < n {
		x := loadF32Part(src[i:])
		erf := erfVec(x.Mul(ec.invSqrt2), &ec, &xc)
		storeF32Part(ec.half.Mul(x).Mul(ec.one.Add(erf)), dst[i:])
	}
}

// tanhSIMDConsts holds TanhF32's broadcast constants — its own struct for
// the same reason erfSIMDConsts is separate from expSIMDConsts: these
// coefficients have no reason to ride softmax/SiLU/GELU's hot path.
type tanhSIMDConsts struct {
	zero, one, negOne, two, c625, nine simd.Float32s
	pc0, pc1, pc2, pc3, pc4            simd.Float32s // Cephes series, |x|<0.625
}

func newTanhSIMDConsts() tanhSIMDConsts {
	return tanhSIMDConsts{
		zero: simd.BroadcastFloat32s(0), one: simd.BroadcastFloat32s(1),
		negOne: simd.BroadcastFloat32s(-1), two: simd.BroadcastFloat32s(2),
		c625: simd.BroadcastFloat32s(0.625), nine: simd.BroadcastFloat32s(9),
		pc0: simd.BroadcastFloat32s(-5.70498872745e-3),
		pc1: simd.BroadcastFloat32s(2.06390887954e-2),
		pc2: simd.BroadcastFloat32s(-5.37397155531e-2),
		pc3: simd.BroadcastFloat32s(1.33314422036e-1),
		pc4: simd.BroadcastFloat32s(-3.33332819422e-1),
	}
}

// tanhVec is TanhF32 (exp.go), vectorized: the same two-branch shape as
// erfVec — Cephes' minimax polynomial for |x|<0.625 (no cancellation there),
// the exponential form 1-2/(e^2x+1) for |x|>=0.625 (stable: subtracts at
// most 0.445 from 1). exp(2*|x|) reuses the NARROW expF32CoreVec, not
// expF32Vec: |x| is bounded to [0.625,9] in this branch (|x|>9 saturates
// separately, before reaching it), so 2*|x| in [1.25,18] never approaches
// expF32CoreVec's overflow/underflow boundaries — same reasoning erfVec's
// tail branch already established. NaN is not specially guarded, same
// scoping call as erfVec (no production caller of TanhF32/GELUTanhF32 has
// ever produced NaN, and the scalar has no stated NaN contract either).
func tanhVec(x simd.Float32s, c *tanhSIMDConsts, xc *expSIMDConsts) simd.Float32s {
	sign := c.negOne.IfElse(x.Less(c.zero), c.one)
	absX := x.Abs()

	z := absX.Mul(absX)
	p := c.pc0.MulAdd(z, c.pc1)
	p = p.MulAdd(z, c.pc2)
	p = p.MulAdd(z, c.pc3)
	p = p.MulAdd(z, c.pc4)
	seriesMag := p.Mul(z).Mul(absX).Add(absX) // p*z*x + x

	e := expF32CoreVec(c.two.Mul(absX), xc)
	expMag := c.one.Sub(c.two.Div(e.Add(c.one))) // 1 - 2/(e^2x+1)

	mag := seriesMag.IfElse(absX.Less(c.c625), expMag)
	mag = c.one.IfElse(absX.Greater(c.nine), mag) // tanh(9) saturates below f32 resolution
	return mag.Mul(sign)
}

// tanhIntoRaw is TanhInto (exp.go) with the length/empty checks already
// done — T1's third and last target.
func tanhIntoRaw(dst, src []float32) {
	n := len(src)
	tc := newTanhSIMDConsts()
	xc := newExpSIMDConsts()
	var probe simd.Float32s
	L := probe.Len()

	i := 0
	for ; i+L <= n; i += L {
		x := simd.LoadFloat32s(src[i : i+L])
		tanhVec(x, &tc, &xc).Store(dst[i : i+L])
	}
	if i < n {
		x := loadF32Part(src[i:])
		storeF32Part(tanhVec(x, &tc, &xc), dst[i:])
	}
}

// geluTanhIntoRaw is GELUTanhInto (exp.go) with the length/empty checks
// already done. Built directly on tanhVec, same op order as scalar
// GELUTanhF32: 0.5*x*(1+tanh(c*(x+0.044715*x^3))).
func geluTanhIntoRaw(dst, src []float32) {
	n := len(src)
	const cConst = 0.7978845608028654 // sqrt(2/pi)
	tc := newTanhSIMDConsts()
	xc := newExpSIMDConsts()
	c := simd.BroadcastFloat32s(cConst)
	a := simd.BroadcastFloat32s(0.044715)
	half := simd.BroadcastFloat32s(0.5)
	one := simd.BroadcastFloat32s(1)
	var probe simd.Float32s
	L := probe.Len()

	i := 0
	for ; i+L <= n; i += L {
		x := simd.LoadFloat32s(src[i : i+L])
		x3 := x.Mul(x).Mul(x)
		arg := c.Mul(x.Add(x3.Mul(a)))
		th := tanhVec(arg, &tc, &xc)
		half.Mul(x).Mul(one.Add(th)).Store(dst[i : i+L])
	}
	if i < n {
		x := loadF32Part(src[i:])
		x3 := x.Mul(x).Mul(x)
		arg := c.Mul(x.Add(x3.Mul(a)))
		th := tanhVec(arg, &tc, &xc)
		storeF32Part(half.Mul(x).Mul(one.Add(th)), dst[i:])
	}
}
