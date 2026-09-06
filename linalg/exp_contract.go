package linalg

import "math"

// The f32 transcendental NUMERIC CONTRACT (docs/task-simd-audit.md S-06 step 2).
//
// The problem this exists to remove. exp.go's expF32Core is a Horner chain of
// `p = p*r + c` in float32. Go AUTO-FUSES that to FMADD on arm64 and does not on
// amd64 below GOAMD64=v3, so the same source produces different bits per
// architecture — and exp.go's own comment already records the consequence, that
// the GOEXPERIMENT=simd build "differs from the scalar one by up to 1 ULP (FMA
// contraction)". Any hand-written kernel makes that worse, because FMLA and
// vfmadd are the instructions one actually wants to use.
//
// The contract: EVERY multiply-add in these kernels is a correctly-rounded f32
// fused multiply-add. Not "fused if the compiler feels like it" — always, on
// every arch, in the Go path and in the assembly alike. An implementation that
// cannot provide one does not conform.
//
// fma32 is how the Go path provides one without an f32 FMA intrinsic, which Go
// does not have (math.FMA is float64-only). Two facts make the f64 detour exact
// rather than approximate:
//
//  1. An f32 × f32 product needs at most 48 significand bits and f64 has 53, so
//     the product is EXACT in f64. Nothing is lost before the add — and, the part
//     that matters for portability, whether the compiler contracts the f64
//     mul-add into FMADDD is IRRELEVANT, because rounding an exact product is the
//     identity. The contract does not depend on a compiler flag or a GOAMD64
//     level.
//  2. Rounding the f64 sum on to f32 is a double rounding, 53 → 24 bits, and
//     double rounding is innocuous when the intermediate carries at least 2p+2
//     bits of the target precision: 2·24+2 = 50 ≤ 53.
//
// So fma32(a,b,c) is the correctly-rounded f32 FMA of a,b,c — which is exactly
// what NEON's FMLA computes. Verified, not argued: TestFMA32_isExactF32FMA
// checks it against math/big at 200-bit precision, and a probe run before this
// file existed found 0 mismatches in 3,000,000 random triples over 60 binades.
func fma32(a, b, c float32) float32 {
	return float32(float64(a)*float64(b) + float64(c))
}

// fms32 is the contract's fused a*b − c, spelled separately so a reduction step
// reads as one operation and cannot become two by accident.
func fms32(a, b, c float32) float32 {
	return float32(float64(a)*float64(b) - float64(c))
}

// mulRound32 is a*b rounded to f32 and then FENCED: because the product is
// formed in f64 and rounded on return, no later addition can be contracted into
// it. Plain `a*b` in float32 leaves the compiler free to fuse with whatever adds
// it next, which is the difference between a Go path and an unfused SIMD kernel.
func mulRound32(a, b float32) float32 {
	return float32(float64(a) * float64(b))
}

// expF32Contract is expF32Core re-specified to the contract: identical
// algorithm, constants and operation ORDER, but every multiply-add pinned to a
// correctly-rounded f32 FMA rather than left to the compiler. It is the ORACLE
// the assembly kernels are gated against AND the scalar path callers get where
// no kernel exists, so the two cannot drift apart.
//
// Deliberately NOT bit-identical to expF32Core: on arm64 they mostly agree (Go
// already fuses there), on amd64 they differ wherever contraction did not
// happen. That divergence is the point — expF32Core's bits are an accident of
// the toolchain and these are chosen.
//
// Caller guarantees a finite input in [expUnderflowF32, expOverflowF32].
func expF32Contract(x float32) float32 {
	// Range-reduce: x = k·ln2 + r, |r| ≤ ln2/2.
	//
	// mulRound32 here is NOT decoration. Written as `z := x * log2eF32` followed
	// by `t := z + roundMagicF32`, Go fuses the pair into a single FMADDS on
	// arm64 — the product is then rounded ONCE with the add instead of twice,
	// which moves kf across its rounding boundary for some inputs and shifts the
	// final result by 1 ULP. A NEON kernel doing VFMUL then VFADD cannot match
	// that, so the contract has to pin it.
	//
	// An earlier draft of this file asserted that this step "contains no
	// multiply-add, so it needs no policy". That was wrong: `z + magic` where z
	// is a product IS a multiply-add, and the raw-bit gate against the NEON
	// kernel caught it at x = -85.603676. Forcing the product to round to f32
	// first leaves the add with nothing to fuse into.
	z := mulRound32(x, log2eF32)
	t := z + roundMagicF32
	kf := t - roundMagicF32
	k := int32(kf)

	// Two-step subtraction, each step ONE fused operation, so the intermediate
	// x − kf·ln2Hi is never rounded separately from its product. That is what a
	// NEON FMLS does and what an unfused compiler gets wrong.
	r := -fms32(kf, ln2HiF32, x)  // x − kf·ln2Hi
	r2 := -fms32(kf, ln2LoF32, r) // r − kf·ln2Lo
	r = r2

	// Cephes single-precision minimax for e^r on [−ln2/2, ln2/2]:
	// e^r = 1 + r + r²·P(r). Six fused steps, in this order, on every target.
	p := float32(1.9875691500e-4)
	p = fma32(p, r, 1.3981999507e-3)
	p = fma32(p, r, 8.3334519073e-3)
	p = fma32(p, r, 4.1665795894e-2)
	p = fma32(p, r, 1.6666665459e-1)
	p = fma32(p, r, 5.0000001201e-1)
	p = fma32(fma32(p, r, 1), r, 1) // p·r² + r + 1, as two fused steps

	// Scale by 2^k — same construction and the same two edge cases as
	// expF32Core; see there for why k=128 must be built in two steps.
	e := k + 127
	if e <= 0 {
		return 0
	}
	if e >= 255 {
		return p * math.Float32frombits(uint32(e-1)<<23) * 2
	}
	return p * math.Float32frombits(uint32(e)<<23)
}

// expClampLoF32 is the floor the softmax contract applies to (v − rowMax) before
// exponentiating. It sits below expUnderflowF32, so exp of it is 0 under the
// flush rule and nothing observable changes — its job is to keep the argument
// FINITE. A row containing −Inf otherwise reaches VFCVTZS(−Inf), which saturates
// to INT32_MIN, while Go's own float→int conversion of −Inf is
// implementation-specific: the two paths would then produce different garbage
// rather than the same answer, and a bit-identity gate would be checking noise.
const expClampLoF32 = -104.0

// softmaxRowContract is the ORDER-PINNED softmax reference: dst[i] =
// e^(src[i]−max) / Σ, with the denominator accumulated into four float64 lane
// partials (element i into partials[i&3]) and folded by the fixed tree
// (p0+p1)+(p2+p3).
//
// The pinning is the whole point. A denominator is a REDUCTION, so its value
// depends on the order of the adds: a sequential scalar sum and a 4-lane vector
// sum are different numbers, which is how a kernel that looks bit-identical
// quietly stops being one. Fixing the order in the CONTRACT rather than in an
// implementation is what lets the Go path and the NEON kernel agree exactly.
//
// A row whose exponentials all underflow yields a uniform distribution rather
// than NaN, matching the behaviour SoftmaxRowInto already documents.
func softmaxRowContract(dst, src []float32) {
	if len(dst) != len(src) {
		panic("linalg: softmaxRowContract length mismatch")
	}
	if len(src) == 0 {
		return
	}
	m := src[0]
	for _, v := range src[1:] {
		if v > m {
			m = v
		}
	}
	var p [4]float64
	for i, v := range src {
		d := v - m
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
	for i := range dst {
		dst[i] *= inv
	}
}

// expClampHiF32 is the ceiling the SiLU contract applies to −x before
// exponentiating, and it is required rather than defensive.
//
// expF32Contract's overflow branch builds its scale factor as
// uint32(e-1)<<23, which is a valid float32 exponent field only while e ≤ 255.
// At e ≥ 256 the shift overflows into the SIGN bit and the function returns −0
// instead of a large number. ExpF32's own range guard makes that unreachable
// through the public entry point — but SiLU feeds it −x, so an unclamped
// x < −89.4 walks straight into it.
const expClampHiF32 = expOverflowF32

// siluF32Contract is x/(1+e^−x) with the exponent argument clamped into the
// range where the contract exp is defined, so the scalar path and the NEON
// kernel agree bit for bit.
//
// It differs from SiLUF32 only in one extreme tail: SiLUF32 lets e^−x reach +Inf
// and returns a signed zero, whereas this returns x/(1+3.4e38) — a tiny nonzero.
// That is a deliberate consequence of keeping both paths inside a defined range,
// and it is far below any tolerance the consumers' parity gates apply.
func siluF32Contract(x float32) float32 {
	t := -x
	if t < expClampLoF32 {
		t = expClampLoF32
	}
	if t > expClampHiF32 {
		t = expClampHiF32
	}
	return x / (1 + expF32Contract(t))
}

// tanhF32Contract is TanhF32 re-specified to the contract: same branches, same
// coefficients, same order, every multiply-add pinned.
//
// TanhF32's explicit `x > 9 → 1` branch is NOT reproduced, and that costs one
// ULP in a narrow band rather than nothing. Saturation still arrives on its own —
// 1 − 2/(e^2x+1) reaches exactly 1 once the subtrahend drops below half an ULP of
// 1, at |x| ≥ 9.02 — but in (9, 9.02) this returns 0.99999994 where TanhF32
// returns 1. Measured across that band, not reasoned about. The trade is one
// fewer branch to mirror in assembly against a 1-ULP difference confined to a
// 0.02-wide interval where tanh is flat to seven digits, which is far below any
// tolerance the parity gates apply.
func tanhF32Contract(x float32) float32 {
	sign := float32(1)
	a := x
	if a < 0 {
		sign, a = -1, -a
	}
	if a < 0.625 {
		z := mulRound32(a, a)
		p := float32(-5.70498872745e-3)
		p = fma32(p, z, 2.06390887954e-2)
		p = fma32(p, z, -5.37397155531e-2)
		p = fma32(p, z, 1.33314422036e-1)
		p = fma32(p, z, -3.33332819422e-1)
		return sign * fma32(mulRound32(p, z), a, a)
	}
	t := mulRound32(2, a)
	if t > expClampHiF32 {
		t = expClampHiF32
	}
	return sign * (1 - 2/(expF32Contract(t)+1))
}

// geluTanhF32Contract is the tanh-approximation GELU, HF's "gelu_new" /
// "gelu_pytorch_tanh", built on tanhF32Contract.
func geluTanhF32Contract(x float32) float32 {
	const c = 0.7978845608028654 // √(2/π)
	x3 := mulRound32(mulRound32(x, x), x)
	inner := mulRound32(c, fma32(0.044715, x3, x))
	return mulRound32(mulRound32(0.5, x), 1+tanhF32Contract(inner))
}

// erfSeriesCoeffs is the Maclaurin branch's 1/(n!(2n+1)) coefficients at full
// precision, in Horner order. They are a named table rather than literals
// because the assembly walks the same values with a post-incrementing load, and
// two copies of sixteen long constants is exactly the kind of duplication that
// drifts.
var erfSeriesCoeffs = [11]float32{
	1.3122532963802806e-08, -1.4503852223150468e-07, 1.4589169000933706e-06,
	-1.3227513227513228e-05, 1.0683760683760684e-04, -7.5757575757575758e-04,
	4.6296296296296294e-03, -2.3809523809523808e-02, 1.0000000000000001e-01,
	-3.3333333333333331e-01, 1,
}

// erfASCoeffs is Abramowitz & Stegun 7.1.26's tail polynomial, Horner order.
var erfASCoeffs = [5]float32{
	1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592,
}

// erfF32Contract is ErfF32 under the contract. Same two branches, same
// coefficients, same order; every multiply-add pinned, and each site's
// fused-or-not decision made to match what the kernel writes rather than left to
// the compiler:
//
//   - both Horner chains are FUSED (FMLA)
//   - 1 + 0.3275911·a is FUSED, because VFMLA is the natural spelling
//   - 1 − (q·t)·e is FUSED as one FMLS, so the product is not rounded separately
//   - (2/√π · a) · p is two plain multiplies with nothing to fuse into
//
// ErfF32's explicit `x > 4 → sign` saturation is NOT reproduced, for the same
// reason as tanh's: the tail branch reaches it on its own, because e^(−x²)
// underflows and 1 − 0 is exactly 1. The exponent argument is clamped so that a
// large |x| cannot drive the kernel's exponent construction out of range.
func erfF32Contract(x float32) float32 {
	a := x
	neg := false
	if a < 0 {
		neg, a = true, -a
	}
	var r float32
	if a < 1 {
		z := mulRound32(a, a)
		p := erfSeriesCoeffs[0]
		for _, c := range erfSeriesCoeffs[1:] {
			p = fma32(p, z, c)
		}
		r = mulRound32(mulRound32(1.1283791670955126, a), p) // 2/√π
	} else {
		t := 1 / fma32(0.3275911, a, 1)
		q := erfASCoeffs[0]
		for _, c := range erfASCoeffs[1:] {
			q = fma32(q, t, c)
		}
		e := -mulRound32(a, a)
		if e < expClampLoF32 {
			e = expClampLoF32
		}
		r = fma32(-mulRound32(q, t), expF32Contract(e), 1) // 1 − q·t·e
	}
	if neg {
		return -r
	}
	return r
}

// geluF32Contract is the exact GELU — x·Φ(x) — built on erfF32Contract.
func geluF32Contract(x float32) float32 {
	const invSqrt2 = 0.7071067811865476
	return mulRound32(mulRound32(0.5, x), 1+erfF32Contract(mulRound32(invSqrt2, x)))
}
