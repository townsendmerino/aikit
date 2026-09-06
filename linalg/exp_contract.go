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
