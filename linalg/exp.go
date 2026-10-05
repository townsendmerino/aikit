package linalg

import "math"

// Float32 transcendentals for the transformer's elementwise stages.
//
// Go's math.Exp/Erf/Tanh are branchy pure-Go scalar float64 routines (range
// reduction, Ldexp, rational-segment selection). They cannot inline into a tight
// loop and the compiler does not auto-vectorize them, so an attention softmax —
// which is O(L²) calls — spends most of its time in function-call and f64
// overhead rather than in the arithmetic the model actually needs.
//
// These are float32 minimax kernels: same shape of algorithm (range-reduce,
// polynomial, scale by 2^k) done once, in f32, branch-free on the hot path.
//
// NOT bit-identical to math.Exp. That is the point — a bit-identical vector
// transcription of Go's f64 routine would preserve every f64 operation and give
// back most of the win. The accuracy contract is instead stated and gated
// explicitly below, and the consumers' parity goldens (which compare against
// PyTorch with tolerances, not bit-exactly) are the end-to-end check.
//
// Accuracy, measured by TestExpF32_accuracy over 3.8M points in [-104, 88]:
// max relative error 8.06e-08 (0.68 ULP of float32), mean 2.29e-08 (0.19 ULP).
// Monotone, and exact at the endpoint softmax depends on (ExpF32(0) == 1).

const (
	// expOverflowF32 is the largest x with e^x finite in float32
	// (ln(MaxFloat32) ≈ 88.7228).
	expOverflowF32 = 88.72283
	// expUnderflowF32 is ln(SmallestNormalFloat32) = ln(2^-126) ≈ -87.3365.
	// Below it the true result is subnormal, and this kernel FLUSHES TO ZERO
	// rather than producing subnormals — the same trade hardware FTZ mode makes,
	// and invisible to softmax, whose inputs are (v - rowMax) and whose sum is
	// at least 1. This is a deliberate divergence from math.Exp and is what the
	// accuracy test excludes from its ULP bound.
	expUnderflowF32 = -87.33654

	log2eF32 = 1.4426950408889634 // 1/ln2

	// ln2 split so that k*ln2Hi is exact in f32 for the |k| ≤ 150 this sees:
	// ln2Hi has its low mantissa bits zeroed, leaving room for the product.
	ln2HiF32 = 0.693145751953125     // ln2 to 12 fractional bits, exact in f32
	ln2LoF32 = 1.428606820309417e-06 // ln2 - ln2Hi

	// roundMagicF32 = 1.5 * 2^23. Adding it to a float32 in [-2^22, 2^22]
	// forces the mantissa to absorb the fractional bits, i.e. round-to-nearest-
	// even, and subtracting it back leaves the rounded integer. Branch-free, and
	// unlike math.Round it is not a function call — math.Round is NOT an amd64
	// intrinsic (perf-campaign item 26; math.Trunc is).
	roundMagicF32 = 12582912.0
)

// ExpF32 returns e^x in float32.
//
// Special cases:
//
//	ExpF32(+Inf) = +Inf
//	ExpF32(-Inf) = 0
//	ExpF32(NaN)  = NaN
//	x >  88.72   = +Inf   (overflow)
//	x < -87.34   = 0      (flushed rather than subnormal — see expUnderflowF32)
func ExpF32(x float32) float32 {
	// Ordered so the common finite case falls through with one comparison
	// pair. NaN fails both comparisons below, so it is handled explicitly.
	if x != x {
		return x
	}
	if x > expOverflowF32 {
		return float32(math.Inf(1))
	}
	if x < expUnderflowF32 {
		return 0
	}
	return expF32Core(x)
}

// expF32Core is ExpF32 without the range/NaN guards, for callers that have
// already bounded their input. softmax is exactly such a caller: it subtracts
// the row max, so every argument is in (-inf, 0], and the only guard it needs
// is the underflow flush.
func expF32Core(x float32) float32 {
	// Range-reduce: x = k*ln2 + r, |r| <= ln2/2.
	z := x * log2eF32
	t := z + roundMagicF32
	kf := t - roundMagicF32
	k := int32(kf)
	// Two-step subtraction: k*ln2Hi is exact, so the rounding error of the
	// reduction is only what k*ln2Lo contributes. Doing it in one step with a
	// single ln2 constant would lose ~10 bits for large |k|.
	r := x - kf*ln2HiF32 - kf*ln2LoF32

	// Minimax polynomial for e^r on [-ln2/2, ln2/2], Cephes single-precision
	// form: e^r = 1 + r + r²·P(r).
	p := float32(1.9875691500e-4)
	p = p*r + 1.3981999507e-3
	p = p*r + 8.3334519073e-3
	p = p*r + 4.1665795894e-2
	p = p*r + 1.6666665459e-1
	p = p*r + 5.0000001201e-1
	p = p*r*r + r + 1

	// Scale by 2^k by constructing the power of two directly.
	e := k + 127
	if e <= 0 {
		return 0 // subnormal or below: flushed (see expUnderflowF32)
	}
	if e >= 255 {
		// k=128 is reachable while the true result is still finite (p ≥ 0.707,
		// so p·2^128 can be under MaxFloat32). Building 2^128 directly would
		// encode +Inf and poison a finite answer, so scale in two steps: 2^127
		// is representable and the intermediate p·2^127 ≤ 1.42·1.70e38 is
		// finite. If the true result really does overflow, the doubling
		// produces +Inf on its own.
		return p * math.Float32frombits(uint32(e-1)<<23) * 2
	}
	return p * math.Float32frombits(uint32(e)<<23)
}

// ExpF32Into writes e^src[i] into dst[i]. dst and src must have the same
// length; they may be the same slice (the kernel reads each element before
// writing it).
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:len(src)]; dst may alias src (that is the in-place form), which is one more reason
// not to pre-zero it.
func ExpF32Into(dst, src []float32) {
	if len(dst) != len(src) {
		panic("linalg: ExpF32Into length mismatch")
	}
	for i, v := range src {
		dst[i] = ExpF32(v)
	}
}

// SoftmaxRowInto normalizes one row into dst: dst[i] = e^(src[i]-max) / Σ.
//
// The sum accumulates in float64 even though the exponentials are float32 —
// that is the encoder's existing contract (softmax accumulates in f64) and it
// costs nothing next to the transcendental.
//
// A row whose exponentials all underflow to zero yields a uniform distribution,
// matching the encoder's scalar softmaxRow rather than emitting NaN.
//
// The per-element work (max-scan, exp, scale) is vectorized when built with
// GOEXPERIMENT=simd (see exp_simd.go); the f64 sum is always scalar, same
// reduction order either way. The two builds are NOT bit-identical to each
// other — the SIMD exp differs from the scalar one by up to 1 ULP (FMA
// contraction; TestExpF32CoreVec_matchesScalarULP gates the bound — see that
// file for why this doesn't reach the encoder's cosine goldens) — but each
// build is internally deterministic and the default (non-experimental) build
// is byte-for-byte what shipped before this existed.
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:len(src)]; dst may alias src (that is the in-place form), which is one more reason
// not to pre-zero it.
func SoftmaxRowInto(dst, src []float32) {
	if len(dst) != len(src) {
		panic("linalg: SoftmaxRowInto length mismatch")
	}
	if len(src) == 0 {
		return
	}
	softmaxRowIntoRaw(dst, src)
}

// SoftmaxRowScaledInto is SoftmaxRowInto(dst, scale*src) — fused: the
// per-element `src[i] *= scale` a caller like attention would otherwise do
// in a separate O(L²) pass before calling SoftmaxRowInto is folded into the
// SAME pass that already touches every element (the exponential), instead
// of a fourth pass over the row. scale MUST be > 0 (attention's own
// 1/sqrt(headDim) always is) — that's what lets the row max be computed on
// the UNSCALED row and multiplied by scale exactly once, rather than
// requiring a max-scan over pre-scaled values: max(scale*x) == scale*max(x)
// for scale > 0, and scale*max(x) is the exact same float32 multiply either
// way, so this is provably bit-identical to the two-pass
// (scale-then-SoftmaxRowInto) sequence it replaces, in BOTH builds — see
// exp_simd.go's softmaxRowScaledIntoRaw for the SIMD build, where this
// genuinely eliminates the separate pass (dead-ends §4.4, reopened once
// item 7's SIMD exp made it worth chasing); the default build still runs it
// as two passes internally, since dead-ends §4.4's own measurement found no
// win there (the pass was ~2% of the cost, swamped by scalar math.Exp).
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:len(src)]; dst may alias src (that is the in-place form), which is one more reason
// not to pre-zero it.
func SoftmaxRowScaledInto(dst, src []float32, scale float32) {
	if len(dst) != len(src) {
		panic("linalg: SoftmaxRowScaledInto length mismatch")
	}
	if len(src) == 0 {
		return
	}
	softmaxRowScaledIntoRaw(dst, src, scale)
}

// ErfF32 returns erf(x) in float32.
//
// Accuracy (TestErfF32_accuracy, 4M points over [-8,8]): max ABSOLUTE error
// 1.92e-07, set by the tail branch. GELU built on it is accurate to 4.67e-07
// absolute and 1.41 ULP relative on its well-conditioned branch — see GELUF32.
//
// TWO BRANCHES, and the split is not optional. Abramowitz & Stegun 7.1.26 (the
// textbook 5-term form used for |x| ≥ 1 below) evaluates 1 − P(t)·e^(−x²). As
// x → 0 both terms tend to 1, so it loses almost all significance exactly where
// erf is nearly linear: measured, it gives 5.5e-07 ABSOLUTE error at x≈0.024,
// which against erf(0.024)≈0.027 is a relative error of 2e-05 — 170 ULP. A&S's
// quoted 1.5e-07 bound is a bound on the mathematical truncation, not on its
// floating-point evaluation, and this is the difference.
//
// For |x| < 1 the Maclaurin series erf(x) = (2/√π)·x·Σ(−x²)ⁿ/(n!(2n+1)) has no
// cancellation at all (every term shares the factor x), so it holds full
// relative precision through zero. Eleven terms put the truncation below 1.3e-08
// at the |x|=1 handover.
//
// Built on ExpF32, so it inherits that kernel's speed. Go's math.Erf is a
// multi-segment rational in float64 and is the most expensive transcendental in
// the forward — measured 29.4 ns/element inside GELU on a 3700X, against 15 for
// math.Exp.
func ErfF32(x float32) float32 {
	sign := float32(1)
	if x < 0 {
		sign, x = -1, -x
	}
	if x > 4 {
		return sign // saturated: 1-erf(4) ≈ 1.5e-8, below f32 resolution at 1
	}
	if x < 1 {
		// Series branch — exact through zero.
		// Coefficients are 1/(n!(2n+1)) at FULL precision. Rounding them to a
		// few digits is not harmless: at |x|=1 every power of x is 1, so a
		// coefficient's own error lands undiminished in the result — writing
		// 1/42 as 2.3810e-2 alone cost 5.4e-07, four times the whole error
		// budget.
		z := x * x
		p := float32(1.3122532963802806e-08)
		p = p*z - 1.4503852223150468e-07
		p = p*z + 1.4589169000933706e-06
		p = p*z - 1.3227513227513228e-05
		p = p*z + 1.0683760683760684e-04
		p = p*z - 7.5757575757575758e-04
		p = p*z + 4.6296296296296294e-03
		p = p*z - 2.3809523809523808e-02
		p = p*z + 1.0000000000000001e-01
		p = p*z - 3.3333333333333331e-01
		p = p*z + 1
		return sign * 1.1283791670955126 * x * p // 2/√π
	}
	// A&S 7.1.26 for the tail, where erf ≈ 1 and 1 − (something ≪ 1) is stable.
	t := 1 / (1 + 0.3275911*x)
	p := float32(1.061405429)
	p = p*t - 1.453152027
	p = p*t + 1.421413741
	p = p*t - 0.284496736
	p = p*t + 0.254829592
	return sign * (1 - p*t*expF32Core(-x*x))
}

// GELUF32 returns the exact (erf) GELU: 0.5·x·(1 + erf(x/√2)).
//
// This is the activation the checkpoints were trained with and the one HF
// evaluates — NOT the tanh approximation, which is a different function.
//
// Its accuracy contract is ABSOLUTE (≤1e-06; measured 4.67e-07), because that
// is what propagates: GELU feeds an FFN matmul over I≈3072 terms with |W|~0.02,
// so a per-element error ε contributes ~1.1ε to a hidden state, against HF
// goldens that already sit at maxΔ≈7.9e-06.
//
// A RELATIVE contract holds only for x ≥ 0. For x < 0 this evaluates erfc via
// (1+erf), which cancels as erf → -1; no erf accurate in the absolute sense can
// give relative accuracy there, and the vanishing outputs it produces do not
// matter downstream. Do not "fix" that by tightening a relative bound.
func GELUF32(x float32) float32 {
	const invSqrt2 = 0.7071067811865476
	return 0.5 * x * (1 + ErfF32(x*invSqrt2))
}

// GELUInto writes GELUF32(src[i]) into dst[i]. Same length required; may
// alias. Vectorized when built with GOEXPERIMENT=simd (see exp_simd.go),
// same NOT-bit-identical caveat as SiLUInto (up to a few ULP, within
// GELUF32's own 1e-6 absolute contract) — the default build is unchanged.
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:len(src)]; dst may alias src (that is the in-place form), which is one more reason
// not to pre-zero it.
func GELUInto(dst, src []float32) {
	if len(dst) != len(src) {
		panic("linalg: GELUInto length mismatch")
	}
	if len(src) == 0 {
		return
	}
	geluIntoRaw(dst, src)
}

// SiLUF32 returns x·sigmoid(x) = x/(1+e^−x), the SwiGLU gate activation.
//
// Accuracy is contracted RELATIVELY here, unlike GELUF32: x/(1+e^−x) is a
// division rather than a subtraction, so neither tail cancels and relative
// accuracy is achievable on both. Bound ≤4 ULP (TestSiLUF32_accuracy over
// [-40,40]).
//
// The saturating ends fall out of ExpF32's own special cases rather than needing
// branches here: for x below the exp overflow bound e^−x is +Inf and x/Inf is a
// signed zero; above it e^−x flushes to 0 and the result is exactly x.
func SiLUF32(x float32) float32 {
	return x / (1 + ExpF32(-x))
}

// SiLUInto writes SiLUF32(src[i]) into dst[i]. Same length required; may
// alias. Vectorized when built with GOEXPERIMENT=simd (see exp_simd.go); NOT
// bit-identical to the default build (up to 1 ULP per exponential, same
// class of difference SoftmaxRowInto documents) — the default (non-
// experimental) build is byte-for-byte what shipped before this existed.
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:len(src)]; dst may alias src (that is the in-place form), which is one more reason
// not to pre-zero it.
func SiLUInto(dst, src []float32) {
	if len(dst) != len(src) {
		panic("linalg: SiLUInto length mismatch")
	}
	if len(src) == 0 {
		return
	}
	siluIntoRaw(dst, src)
}

// TanhF32 returns tanh(x) in float32.
//
// TWO BRANCHES, for the same reason ErfF32 has them. The closed form
// tanh(x) = (e^2x − 1)/(e^2x + 1) cancels catastrophically as x → 0: both e^2x
// and 1 tend to 1, so the numerator loses every significant bit exactly where
// tanh is at its most linear. Cephes' minimax polynomial x·P(x²) is used below
// |x| = 0.625, where it has no subtraction at all; above it the exponential
// form is written as 1 − 2/(e^2x + 1), which at x ≥ 0.625 subtracts at most
// 0.445 from 1 and so is stable.
//
// Accuracy: ≤2 ULP relative (TestTanhF32_accuracy over [-20,20]).
func TanhF32(x float32) float32 {
	sign := float32(1)
	if x < 0 {
		sign, x = -1, -x
	}
	if x > 9 {
		return sign // tanh(9) = 1 - 3.0e-08, below f32 resolution at 1
	}
	if x < 0.625 {
		z := x * x
		p := float32(-5.70498872745e-3)
		p = p*z + 2.06390887954e-2
		p = p*z - 5.37397155531e-2
		p = p*z + 1.33314422036e-1
		p = p*z - 3.33332819422e-1
		return sign * (p*z*x + x)
	}
	return sign * (1 - 2/(ExpF32(2*x)+1))
}

// GELUTanhF32 returns the tanh approximation of GELU:
//
//	0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
//
// This is HF's "gelu_pytorch_tanh" / "gelu_new" — a DIFFERENT function from the
// exact erf GELU in GELUF32, not an approximation of it that may be swapped in.
// SigLIP (and so Gemma 3's vision tower) is trained with this one; substituting
// GELUF32 would be a silent model change.
//
// Contract is ABSOLUTE (≤1e-06), for the reason set out on GELUF32: it feeds a
// matmul, and its negative branch is cancellation-dominated so a relative bound
// there would constrain the wrong thing.
func GELUTanhF32(x float32) float32 {
	const c = 0.7978845608028654 // √(2/π)
	return 0.5 * x * (1 + TanhF32(c*(x+0.044715*x*x*x)))
}

// GELUTanhInto writes GELUTanhF32(src[i]) into dst[i]. Same length; may
// alias. Vectorized when built with GOEXPERIMENT=simd (see exp_simd.go),
// same NOT-bit-identical caveat as SiLUInto/GELUInto — the default build is
// unchanged.
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:len(src)]; dst may alias src (that is the in-place form), which is one more reason
// not to pre-zero it.
func GELUTanhInto(dst, src []float32) {
	if len(dst) != len(src) {
		panic("linalg: GELUTanhInto length mismatch")
	}
	if len(src) == 0 {
		return
	}
	geluTanhIntoRaw(dst, src)
}

// TanhInto writes TanhF32(src[i]) into dst[i]. Same length; may alias.
// Vectorized when built with GOEXPERIMENT=simd (see exp_simd.go), same
// NOT-bit-identical caveat as SiLUInto/GELUInto — the default build is
// unchanged.
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:len(src)]; dst may alias src (that is the in-place form), which is one more reason
// not to pre-zero it.
func TanhInto(dst, src []float32) {
	if len(dst) != len(src) {
		panic("linalg: TanhInto length mismatch")
	}
	if len(src) == 0 {
		return
	}
	tanhIntoRaw(dst, src)
}
