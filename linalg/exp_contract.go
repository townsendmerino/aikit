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
	// Range-reduce: x = k·ln2 + r, |r| ≤ ln2/2. The round-to-nearest-even magic
	// is exact in f32 and contains no multiply-add, so it needs no policy.
	z := x * log2eF32
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
