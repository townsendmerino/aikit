package linalg

import (
	"math"
	"math/big"
	"math/rand/v2"
	"testing"
)

// exactFMA32 is the correctly-rounded f32 fused multiply-add, computed at 200
// bits and rounded ONCE. This is the definition fma32 claims to implement, and
// math/big is used rather than another float path so the oracle cannot share a
// bug with the thing it checks.
func exactFMA32(a, b, c float32) float32 {
	A := new(big.Float).SetPrec(200).SetFloat64(float64(a))
	B := new(big.Float).SetPrec(200).SetFloat64(float64(b))
	C := new(big.Float).SetPrec(200).SetFloat64(float64(c))
	r := new(big.Float).SetPrec(200).Mul(A, B)
	r.Add(r, C)
	f, _ := r.Float32()
	return f
}

// TestFMA32_isExactF32FMA is the foundation the whole S-06 step-2 contract rests
// on: Go has no f32 FMA, so fma32 reaches one through f64, and if that detour is
// not exactly a correctly-rounded f32 FMA then the Go path and NEON's FMLA
// cannot be bit-identical and the contract is unsound.
//
// Adversarial by construction rather than uniform: operands are drawn across 60
// binades so products land near the top and bottom of the exponent range, and
// the c term is sometimes set to −a·b to force catastrophic cancellation, which
// is precisely where a double rounding would show if one were happening.
func TestFMA32_isExactF32FMA(t *testing.T) {
	rng := rand.New(rand.NewPCG(0xf3a, 0x32))
	rnd := func() float32 {
		return float32(math.Ldexp(rng.Float64()*2-1, rng.IntN(60)-30))
	}
	const N = 300_000
	for i := range N {
		a, b := rnd(), rnd()
		var c float32
		switch i % 4 {
		case 0:
			c = rnd()
		case 1: // near-total cancellation
			c = -a * b
		case 2: // tiny addend against a large product
			c = float32(math.Ldexp(float64(rnd()), -40))
		default: // huge addend against a tiny product
			c = float32(math.Ldexp(float64(rnd()), 40))
		}
		if got, want := fma32(a, b, c), exactFMA32(a, b, c); math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("fma32(%v,%v,%v) = %v (%08x), exact f32 FMA = %v (%08x)",
				a, b, c, got, math.Float32bits(got), want, math.Float32bits(want))
		}
	}

	// The specials are checked against IEEE-754 directly, NOT against the
	// math/big oracle: big.Float cannot represent 0×∞ and panics on it, so the
	// oracle is undefined exactly where the interesting cases live. What matters
	// is that fma32 does what a hardware FMA does, and IEEE fixes that.
	inf, ninf := float32(math.Inf(1)), float32(math.Inf(-1))
	nan := float32(math.NaN())
	for _, tc := range []struct {
		a, b, c float32
		want    float32
		wantNaN bool
		why     string
	}{
		{inf, 1, 0, inf, false, "∞·1+0"},
		{ninf, 1, 0, ninf, false, "−∞·1+0"},
		{0, inf, 1, 0, true, "0·∞ is IEEE-invalid → NaN"},
		{inf, 0, 1, 0, true, "∞·0 likewise"},
		{1, 1, ninf, ninf, false, "1·1+(−∞)"},
		{inf, 1, ninf, 0, true, "∞ + (−∞) → NaN"},
		{nan, 1, 1, 0, true, "NaN propagates"},
		{0, 0, 0, 0, false, "zero"},
	} {
		g := fma32(tc.a, tc.b, tc.c)
		if tc.wantNaN {
			if !math.IsNaN(float64(g)) {
				t.Fatalf("fma32(%v,%v,%v) = %v, want NaN (%s)", tc.a, tc.b, tc.c, g, tc.why)
			}
			continue
		}
		if math.Float32bits(g) != math.Float32bits(tc.want) {
			t.Fatalf("fma32(%v,%v,%v) = %v, want %v (%s)", tc.a, tc.b, tc.c, g, tc.want, tc.why)
		}
	}
}

// TestFMA32_contractionIndependent pins the property that makes the contract
// portable: because an f32×f32 product is EXACT in f64, fma32's result does not
// depend on whether the compiler contracts its internal f64 mul-add into an
// FMADD. Computing the same thing with an explicit f64 FMA must agree on every
// input — if it ever did not, fma32's value would depend on a build flag and the
// scalar path could not be an oracle for the assembly.
func TestFMA32_contractionIndependent(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	rnd := func() float32 { return float32(math.Ldexp(rng.Float64()*2-1, rng.IntN(60)-30)) }
	for range 300_000 {
		a, b, c := rnd(), rnd(), rnd()
		explicit := float32(math.FMA(float64(a), float64(b), float64(c)))
		if math.Float32bits(fma32(a, b, c)) != math.Float32bits(explicit) {
			t.Fatalf("fma32(%v,%v,%v)=%v but explicit f64 FMA gives %v — the contract "+
				"would depend on compiler contraction", a, b, c, fma32(a, b, c), explicit)
		}
	}
}

// TestExpF32Contract_accuracy records the contract exp's accuracy against
// math.Exp. RECORDED, NOT GATED at a tight bound — S-06's contract is
// self-consistency between the scalar path and the kernels, not agreement with
// the f64 library. A loose ceiling is asserted only so that a catastrophic
// regression (a mistyped coefficient) still fails.
func TestExpF32Contract_accuracy(t *testing.T) {
	var maxULP float64
	var at float64
	for i := range 2_000_001 {
		xf := float32(-104 + float64(i)*(192.0/2_000_000))
		x := float64(xf)
		if x < expUnderflowF32 || x > expOverflowF32 {
			continue
		}
		got := float64(expF32Contract(xf))
		// The oracle MUST be evaluated at the f32 input the kernel actually saw,
		// not at the f64 the loop generated. exp amplifies an input difference
		// into the output one-for-one in relative terms, so at x≈80 the ~2e-6
		// relative gap between an f64 x and its f32 rounding becomes ~17 ULP of
		// output — which reads as a catastrophic kernel error and is entirely an
		// artefact of the harness. This cost a debugging round; it is spelled out
		// so the next person does not repeat it.
		want := math.Exp(x)
		if want == 0 || math.IsInf(want, 0) {
			continue
		}
		ulpSize := float64(math.Nextafter32(float32(want), float32(math.Inf(1))) - float32(want))
		ulp := math.Abs(got-want) / math.Abs(ulpSize)
		if ulp > maxULP {
			maxULP, at = ulp, x
		}
	}
	t.Logf("expF32Contract vs math.Exp: max %.3f ULP (at x=%.4f)", maxULP, at)
	if maxULP > 2.0 {
		t.Fatalf("max %.3f ULP at x=%v exceeds the sanity ceiling of 2.0 — a coefficient is wrong",
			maxULP, at)
	}
	if e := expF32Contract(0); e != 1 {
		t.Fatalf("expF32Contract(0) = %v, want exactly 1 (softmax depends on it)", e)
	}
}

// TestExpF32Contract_vsExpF32Core records how far the contract moved the bits
// from the shipped kernel. Recorded, not gated: the contract deliberately
// changes them, and the number is here so the size of that change is known
// rather than discovered downstream in someone's goldens.
func TestExpF32Contract_vsExpF32Core(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 5))
	var differ, n int
	var maxULP float64
	for range 500_000 {
		x := float32(rng.Float64()*170 - 87)
		a, b := expF32Contract(x), expF32Core(x)
		n++
		if math.Float32bits(a) != math.Float32bits(b) {
			differ++
			if b != 0 {
				u := math.Abs(float64(a-b)) / math.Abs(float64(math.Nextafter32(b, float32(math.Inf(1)))-b))
				if u > maxULP {
					maxULP = u
				}
			}
		}
	}
	// The share differs BY ARCHITECTURE and that is the finding, not noise: Go
	// fuses expF32Core's Horner chain on arm64 and does not below GOAMD64=v3 on
	// amd64, so the shipped kernel's own bits are arch-dependent. The contract's
	// are not. Logged without naming an arch, because this test runs on both.
	t.Logf("contract vs shipped expF32Core: %.2f%% of inputs differ, max %.2f ULP",
		100*float64(differ)/float64(n), maxULP)
}
