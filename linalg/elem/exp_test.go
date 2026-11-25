package elem

import (
	"math"
	"math/rand"
	"testing"
)

// TestExpF32_accuracy is the gate that decides whether ExpF32 may be used at
// all. It is NOT bit-identical to math.Exp — it is a float32 minimax kernel —
// so the contract it must meet is a stated relative-error bound, checked
// against math.Exp evaluated in float64 and rounded, which is the best
// available reference.
//
// The bound is 1 ULP of float32 (2^-23 ≈ 1.19e-07). Anything looser would start
// to show up in the encoders' cosine goldens.
func TestExpF32_accuracy(t *testing.T) {
	const ulp = 1.0 / (1 << 23)
	rng := rand.New(rand.NewSource(5))

	var worst, mean float64
	var worstAt float32
	var n int
	check := func(x float32) {
		got := ExpF32(x)
		want := math.Exp(float64(x))
		if math.IsInf(want, 1) || want < math.SmallestNonzeroFloat32 {
			return // overflow handled by the special-case test
		}
		if want < 1.1754944e-38 {
			// Subnormal territory: this kernel flushes to zero by contract.
			if got != 0 {
				t.Errorf("ExpF32(%v) = %v, want 0 (subnormal results are flushed)", x, got)
			}
			return
		}
		rel := math.Abs((float64(got) - want) / want)
		if rel > worst {
			worst, worstAt = rel, x
		}
		mean += rel
		n++
	}

	// Dense sweep over the whole finite range, plus the softmax range (x <= 0)
	// at higher density since that is where every attention call lands.
	for i := range 2_000_000 {
		check(float32(-103.9 + 192.6*float64(i)/2_000_000))
	}
	for range 2_000_000 {
		check(float32(-rng.Float64() * 40)) // softmax: x = v - rowMax <= 0
	}

	mean /= float64(n)
	if worst > ulp {
		t.Errorf("max relative error %.3g (%.2f ULP) at x=%v — exceeds the 1 ULP contract",
			worst, worst/ulp, worstAt)
	}
	t.Logf("over %d points: max rel err %.3g (%.2f ULP) at x=%v, mean %.3g (%.2f ULP)",
		n, worst, worst/ulp, worstAt, mean, mean/ulp)
}

func TestExpF32_specialCases(t *testing.T) {
	inf := float32(math.Inf(1))
	for _, tc := range []struct {
		in, want float32
		desc     string
	}{
		{0, 1, "exp(0) must be exactly 1 — softmax relies on the row max mapping to 1"},
		{inf, inf, "+Inf"},
		{-inf, 0, "-Inf"},
		{200, inf, "overflow"},
		{-200, 0, "underflow flushes to zero"},
		{-87.34, 0, "just below the underflow bound — flushed, not subnormal"},
	} {
		if got := ExpF32(tc.in); got != tc.want {
			t.Errorf("ExpF32(%v) = %v, want %v — %s", tc.in, got, tc.want, tc.desc)
		}
	}
	// The top of the range: k reaches 128 while the true result is still
	// finite, so a one-step 2^k scale would encode +Inf and poison it.
	// exp(88.376335) ≈ 2.406e38, under MaxFloat32 ≈ 3.403e38.
	if got := ExpF32(88.376335); math.IsInf(float64(got), 1) {
		t.Errorf("ExpF32(88.376335) = +Inf, want ≈2.406e38 — the 2^k scale overflowed early")
	}
	if got := ExpF32(float32(math.NaN())); !math.IsNaN(float64(got)) {
		t.Errorf("ExpF32(NaN) = %v, want NaN", got)
	}
}

// TestExpF32_monotone matters for softmax: a non-monotone exp could reorder two
// nearly-equal logits and flip an argmax downstream.
func TestExpF32_monotone(t *testing.T) {
	prev := ExpF32(-104)
	for x := float32(-104); x < 88.7; x += 0.0001 {
		got := ExpF32(x)
		if got < prev {
			t.Fatalf("not monotone at x=%v: %v then %v", x, prev, got)
		}
		prev = got
	}
}

// TestSoftmaxRowInto_matchesReference checks the fused row softmax against a
// float64 reference. Softmax normalizes, so a uniform relative error in the
// exponentials cancels — the output tolerance is tighter than ExpF32's own.
func TestSoftmaxRowInto_matchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	for _, n := range []int{1, 2, 7, 64, 691} {
		for trial := range 200 {
			src := make([]float32, n)
			scale := []float64{1, 10, 100}[trial%3]
			for i := range src {
				src[i] = float32(rng.NormFloat64() * scale)
			}
			got := make([]float32, n)
			SoftmaxRowInto(got, src)

			// Reference: float64 throughout.
			maxV := float64(src[0])
			for _, v := range src[1:] {
				if float64(v) > maxV {
					maxV = float64(v)
				}
			}
			var sum float64
			ref := make([]float64, n)
			for i, v := range src {
				ref[i] = math.Exp(float64(v) - maxV)
				sum += ref[i]
			}
			var worst float64
			var total float64
			for i := range ref {
				ref[i] /= sum
				total += float64(got[i])
				if d := math.Abs(float64(got[i]) - ref[i]); d > worst {
					worst = d
				}
			}
			if worst > 1e-7 {
				t.Fatalf("n=%d scale=%g: max abs deviation %g from the f64 reference", n, scale, worst)
			}
			if total < 0.999 || total > 1.001 {
				t.Fatalf("n=%d scale=%g: row sums to %v, want ~1", n, scale, total)
			}
		}
	}
}

// TestSoftmaxRowScaledInto_bitIdenticalToTwoPass is the load-bearing gate for
// T2 (dead-ends §4.4): SoftmaxRowScaledInto(dst, src, scale) must be
// BIT-IDENTICAL, not just close, to the two-pass sequence it replaces
// (manually scale src, then SoftmaxRowInto) — the whole fusion's premise is
// that eliminating the separate pass changes nothing about the arithmetic,
// in both builds. A mismatch here means the "provably bit-identical" claim
// in SoftmaxRowScaledInto's doc comment is wrong, not just imprecise.
func TestSoftmaxRowScaledInto_bitIdenticalToTwoPass(t *testing.T) {
	rng := rand.New(rand.NewSource(73))
	for _, n := range []int{1, 2, 7, 8, 15, 16, 17, 64, 80, 691} {
		for _, scale := range []float32{1, 0.125, 0.0883883476 /* 1/sqrt(128), a real headDim */, 3.7} {
			for trial := range 50 {
				src := make([]float32, n)
				mag := []float64{1, 10, 100}[trial%3]
				for i := range src {
					src[i] = float32(rng.NormFloat64() * mag)
				}

				fused := make([]float32, n)
				SoftmaxRowScaledInto(fused, src, scale)

				twoPass := make([]float32, n)
				for i, v := range src {
					twoPass[i] = v * scale
				}
				SoftmaxRowInto(twoPass, twoPass)

				for i := range fused {
					if fused[i] != twoPass[i] {
						t.Fatalf("n=%d scale=%v trial=%d i=%d: fused=%v two-pass=%v — NOT bit-identical",
							n, scale, trial, i, fused[i], twoPass[i])
					}
				}
			}
		}
	}
}

func BenchmarkExpF32_vs_mathExp(b *testing.B) {
	const n = 65536
	rng := rand.New(rand.NewSource(1))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(-rng.Float64() * 20)
	}
	dst := make([]float32, n)

	b.Run("math.Exp/f64", func(b *testing.B) {
		for b.Loop() {
			for i, v := range src {
				dst[i] = float32(math.Exp(float64(v)))
			}
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
	})
	b.Run("ExpF32", func(b *testing.B) {
		for b.Loop() {
			ExpF32Into(dst, src)
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
	})
}

// BenchmarkSoftmaxRow_vs_scalar compares the fused f32 row softmax against the
// shape the encoder currently runs (scalar math.Exp, f64 accumulate). This is
// the number that decides item 13, not ExpF32 in isolation: the encoder's loop
// also does a max pass and a normalize pass, which do not get faster.
func BenchmarkSoftmaxRow_vs_scalar(b *testing.B) {
	rng := rand.New(rand.NewSource(2))
	for _, n := range []int{256, 691} {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64() * 4)
		}
		dst := make([]float32, n)

		b.Run("n"+itoaBench(n)+"/scalar", func(b *testing.B) {
			for b.Loop() {
				copy(dst, src)
				softmaxRowScalarRef(dst)
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
		})
		b.Run("n"+itoaBench(n)+"/f32", func(b *testing.B) {
			for b.Loop() {
				SoftmaxRowInto(dst, src)
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
		})
	}
}

// BenchmarkSoftmaxRowScaled_vs_twoPass is T2's own measurement (dead-ends
// §4.4, reopened after item 7's SIMD exp): the two-pass sequence a caller
// used to run (its own `src[i] *= scale` loop, then SoftmaxRowInto) against
// the fused SoftmaxRowScaledInto, at the two shapes dead-ends §4.4 named —
// L=80 (the rerank shape, where the separate pass was ~2% of the cost) and
// L=690 (the GTE shape, where softmax was half the pre-item-13 forward).
func BenchmarkSoftmaxRowScaled_vs_twoPass(b *testing.B) {
	rng := rand.New(rand.NewSource(11))
	const scale = 0.0883883476 // 1/sqrt(128)
	for _, n := range []int{80, 691} {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64() * 4)
		}
		dst := make([]float32, n)

		b.Run("n"+itoaBench(n)+"/twoPass", func(b *testing.B) {
			for b.Loop() {
				for i, v := range src {
					dst[i] = v * scale
				}
				SoftmaxRowInto(dst, dst)
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
		})
		b.Run("n"+itoaBench(n)+"/fused", func(b *testing.B) {
			for b.Loop() {
				SoftmaxRowScaledInto(dst, src, scale)
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
		})
	}
}

// softmaxRowScalarRef is a copy of encoder.softmaxRow, so the comparison is
// against what actually ships rather than a strawman.
func softmaxRowScalarRef(row []float32) {
	if len(row) == 0 {
		return
	}
	maxV := row[0]
	for _, v := range row[1:] {
		if v > maxV {
			maxV = v
		}
	}
	var sum float64
	for i, v := range row {
		e := math.Exp(float64(v - maxV))
		row[i] = float32(e)
		sum += e
	}
	if sum == 0 {
		for i := range row {
			row[i] = 1.0 / float32(len(row))
		}
		return
	}
	inv := float32(1.0 / sum)
	for i := range row {
		row[i] *= inv
	}
}

func itoaBench(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// TestErfF32_accuracy measures ErfF32's error against math.Erf rather than
// trusting Abramowitz & Stegun's quoted 1.5e-07 bound, and asserts the bound
// this package documents. It also checks the derived GELU, which is what the
// encoders actually call: GELU multiplies erf's error by 0.5·|x|, so a bound
// that is fine on erf is not automatically fine on GELU.
func TestErfF32_accuracy(t *testing.T) {
	const ulp = 1.0 / (1 << 23)
	var worstErf, worstGeluRel, worstGeluAbs float64
	var atErf, atGelu float32

	for i := range 4_000_001 {
		x := float32(-8 + 16*float64(i)/4_000_000)
		if d := math.Abs(float64(ErfF32(x)) - math.Erf(float64(x))); d > worstErf {
			worstErf, atErf = d, x
		}
		want := 0.5 * float64(x) * (1 + math.Erf(float64(x)*0.7071067811865476))
		abs := math.Abs(float64(GELUF32(x)) - want)
		if abs > worstGeluAbs {
			worstGeluAbs = abs
		}
		// Relative error is only meaningful where GELU is well conditioned.
		// In the saturating negative tail the true value is formed as
		// 0.5·x·(1+erf(x/√2)) with erf → -1, so it is pure cancellation:
		// at x=-3.44 the result is ~1e-3 and at x=-5.5 it is ~1e-7, and
		// ANY implementation shows a large relative error there while
		// being absolutely accurate to far under one float32 ULP of the
		// activations around it. That region is judged by `abs` below,
		// which is what actually propagates into the next matmul.
		// Relative error is checked on the POSITIVE branch only, and that is a
		// property of the function, not a convenience. For x < 0,
		// GELU(x) = 0.5·x·(1+erf(x/√2)) evaluates erfc, and relative accuracy
		// of erfc cannot be had from an erf accurate in the ABSOLUTE sense —
		// (1+erf) → 0 there, so the subtraction cancels however good erf is.
		// A relative bound on that branch would be a bound on the wrong thing.
		// For x ≥ 0, (1+erf) ∈ [1,2] and GELU is well conditioned.
		if x >= 0.1 {
			if rel := abs / math.Abs(want); rel > worstGeluRel {
				worstGeluRel, atGelu = rel, x
			}
		}
	}

	// erf's bound is set by the A&S tail branch (|x| ≥ 1); the series branch
	// below it is an order of magnitude better. 2.5e-07 leaves headroom over
	// the 1.92e-07 measured.
	if worstErf > 2.5e-7 {
		t.Errorf("erf max abs error %.3g at x=%v — exceeds 2.5e-07", worstErf, atErf)
	}
	// THE BINDING CONTRACT IS ABSOLUTE, not relative, and it is derived rather
	// than tuned. GELU output feeds an FFN matmul: out = Σ_j gelu(x_j)·W_j over
	// I=3072 terms with |W| ~ 0.02, so an absolute error ε per element
	// contributes ~ε·√I·|W| ≈ 1.1ε to a hidden state. The encoders' HF goldens
	// currently sit at maxΔ ≈ 7.9e-06, so a 1e-06 bound here spends under 15%
	// of an already-passing budget. The relative check below is a sanity bound
	// on the well-conditioned region only.
	if worstGeluRel > 4*ulp {
		t.Errorf("GELU max relative error %.3g (%.2f ULP) at x=%v (positive branch) — exceeds 4 ULP",
			worstGeluRel, worstGeluRel/ulp, atGelu)
	}
	// The tail is judged absolutely: an error here is added to an activation
	// of order |x| ≤ 8, so 1e-6 is comfortably below its float32 resolution.
	if worstGeluAbs > 1e-6 {
		t.Errorf("GELU max absolute error %.3g — exceeds 1e-06", worstGeluAbs)
	}
	t.Logf("erf max abs %.3g (at %v); GELU max abs %.3g (binding); GELU max rel on x≥0.1 %.3g (%.2f ULP, at %v)",
		worstErf, atErf, worstGeluAbs, worstGeluRel, worstGeluRel/ulp, atGelu)
}

func TestErfF32_specialCases(t *testing.T) {
	for _, tc := range []struct{ in, want float32 }{
		{0, 0}, {10, 1}, {-10, -1}, {5, 1}, {-5, -1},
	} {
		if got := ErfF32(tc.in); got != tc.want {
			t.Errorf("ErfF32(%v) = %v, want %v", tc.in, got, tc.want)
		}
	}
	// GELU's saturating ends: large positive -> x, large negative -> 0.
	if got := GELUF32(10); got != 10 {
		t.Errorf("GELUF32(10) = %v, want 10", got)
	}
	if got := GELUF32(-10); got != 0 {
		t.Errorf("GELUF32(-10) = %v, want 0", got)
	}
	if got := GELUF32(0); got != 0 {
		t.Errorf("GELUF32(0) = %v, want 0", got)
	}
}

func BenchmarkGELU_vs_mathErf(b *testing.B) {
	const n = 65536
	rng := rand.New(rand.NewSource(3))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rng.NormFloat64() * 2)
	}
	dst := make([]float32, n)
	b.Run("math.Erf/f64", func(b *testing.B) {
		for b.Loop() {
			for i, v := range src {
				dst[i] = float32(0.5 * float64(v) * (1 + math.Erf(float64(v)*0.7071067811865476)))
			}
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
	})
	b.Run("GELUF32", func(b *testing.B) {
		for b.Loop() {
			GELUInto(dst, src)
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
	})
}

// BenchmarkSiLU_vs_scalar compares SiLUInto (vectorized under GOEXPERIMENT=simd,
// scalar loop over SiLUF32 in the default build) against a math.Exp-based
// scalar reference — the shape encoder/linalg.go's silu wraps.
func BenchmarkSiLU_vs_scalar(b *testing.B) {
	const n = 65536
	rng := rand.New(rand.NewSource(4))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rng.NormFloat64() * 4)
	}
	dst := make([]float32, n)
	b.Run("math.Exp/f64", func(b *testing.B) {
		for b.Loop() {
			for i, v := range src {
				dst[i] = float32(float64(v) / (1 + math.Exp(-float64(v))))
			}
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
	})
	b.Run("SiLUInto", func(b *testing.B) {
		for b.Loop() {
			SiLUInto(dst, src)
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
	})
}

func TestSiLUF32_accuracy(t *testing.T) {
	// SiLU is judged RELATIVELY, unlike GELU. x/(1+e^-x) is a division, not a
	// subtraction, so neither tail cancels and relative accuracy is achievable
	// on both — and it is the right bound, since silu(x) → x for large x and an
	// absolute bound would be meaninglessly strict there (at x=13.8 the value
	// IS 13.8, so a 1e-06 absolute error is 0.77 ULP).
	const ulp = 1.0 / (1 << 23)
	var worstRel, worstAbs float64
	var at float32
	for i := range 4_000_001 {
		x := float32(-40 + 80*float64(i)/4_000_000)
		xf := float64(x)
		want := xf / (1 + math.Exp(-xf))
		d := math.Abs(float64(SiLUF32(x)) - want)
		if d > worstAbs {
			worstAbs = d
		}
		if math.Abs(want) > 1e-6 {
			if rel := d / math.Abs(want); rel > worstRel {
				worstRel, at = rel, x
			}
		}
	}
	if worstRel > 4*ulp {
		t.Errorf("SiLU max relative error %.3g (%.2f ULP) at x=%v — exceeds 4 ULP",
			worstRel, worstRel/ulp, at)
	}
	t.Logf("SiLU max rel error %.3g (%.2f ULP) at x=%v; max abs %.3g", worstRel, worstRel/ulp, at, worstAbs)

	// Saturating ends must not produce NaN or Inf.
	for _, x := range []float32{-200, -100, 100, 200, 0} {
		if got := SiLUF32(x); math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
			t.Errorf("SiLUF32(%v) = %v, want finite", x, got)
		}
	}
	if got := SiLUF32(100); got != 100 {
		t.Errorf("SiLUF32(100) = %v, want 100 (saturates to x)", got)
	}
	if got := SiLUF32(0); got != 0 {
		t.Errorf("SiLUF32(0) = %v, want 0", got)
	}
}

// BenchmarkTanh_vs_scalar compares TanhInto (vectorized under
// GOEXPERIMENT=simd, scalar loop over TanhF32 in the default build) against
// a math.Tanh-based scalar reference — the shape encoder/crossencoder.go's
// pooling wraps.
func BenchmarkTanh_vs_scalar(b *testing.B) {
	const n = 65536
	rng := rand.New(rand.NewSource(6))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rng.NormFloat64() * 3)
	}
	dst := make([]float32, n)
	b.Run("math.Tanh/f64", func(b *testing.B) {
		for b.Loop() {
			for i, v := range src {
				dst[i] = float32(math.Tanh(float64(v)))
			}
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
	})
	b.Run("TanhInto", func(b *testing.B) {
		for b.Loop() {
			TanhInto(dst, src)
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
	})
}

// BenchmarkGELUTanh_vs_scalar is BenchmarkGELU_vs_mathErf's twin for the
// tanh-approximation GELU (SigLIP/Gemma 3's vision tower activation).
func BenchmarkGELUTanh_vs_scalar(b *testing.B) {
	const n = 65536
	rng := rand.New(rand.NewSource(7))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rng.NormFloat64() * 2)
	}
	dst := make([]float32, n)
	const c = 0.7978845608028654
	b.Run("math.Tanh/f64", func(b *testing.B) {
		for b.Loop() {
			for i, v := range src {
				vf := float64(v)
				dst[i] = float32(0.5 * vf * (1 + math.Tanh(c*(vf+0.044715*vf*vf*vf))))
			}
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
	})
	b.Run("GELUTanhInto", func(b *testing.B) {
		for b.Loop() {
			GELUTanhInto(dst, src)
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
	})
}

func TestTanhF32_accuracy(t *testing.T) {
	// tanh is judged relatively on both branches: neither the polynomial nor
	// 1 - 2/(e^2x+1) cancels in its own domain, which is the whole point of the
	// split. A single-branch (e^2x-1)/(e^2x+1) would fail this near zero.
	const ulp = 1.0 / (1 << 23)
	var worstRel float64
	var at float32
	for i := range 4_000_001 {
		x := float32(-20 + 40*float64(i)/4_000_000)
		want := math.Tanh(float64(x))
		if want == 0 {
			continue
		}
		if rel := math.Abs((float64(TanhF32(x)) - want) / want); rel > worstRel {
			worstRel, at = rel, x
		}
	}
	if worstRel > 2*ulp {
		t.Errorf("tanh max relative error %.3g (%.2f ULP) at x=%v — exceeds 2 ULP",
			worstRel, worstRel/ulp, at)
	}
	t.Logf("tanh max rel error %.3g (%.2f ULP) at x=%v", worstRel, worstRel/ulp, at)

	for _, tc := range []struct{ in, want float32 }{
		{0, 0}, {20, 1}, {-20, -1}, {10, 1}, {-10, -1},
	} {
		if got := TanhF32(tc.in); got != tc.want {
			t.Errorf("TanhF32(%v) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

func TestGELUTanhF32_accuracy(t *testing.T) {
	const c = 0.7978845608028654
	var worstAbs float64
	var at float32
	for i := range 4_000_001 {
		x := float32(-10 + 20*float64(i)/4_000_000)
		xf := float64(x)
		want := 0.5 * xf * (1 + math.Tanh(c*(xf+0.044715*xf*xf*xf)))
		if d := math.Abs(float64(GELUTanhF32(x)) - want); d > worstAbs {
			worstAbs, at = d, x
		}
	}
	if worstAbs > 1e-6 {
		t.Errorf("GELU-tanh max absolute error %.3g at x=%v — exceeds 1e-06", worstAbs, at)
	}
	t.Logf("GELU-tanh max abs error %.3g at x=%v", worstAbs, at)

	// It must NOT be interchangeable with the erf GELU — they are different
	// functions, and a future refactor that "simplifies" one into the other
	// should fail here rather than silently change every SigLIP output.
	var maxDiff float64
	for i := range 20001 {
		x := float32(-5 + 10*float64(i)/20000)
		if d := math.Abs(float64(GELUTanhF32(x)) - float64(GELUF32(x))); d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff < 1e-4 {
		t.Errorf("GELUTanhF32 and GELUF32 differ by only %g — they are supposed to be "+
			"different activations; has one been aliased to the other?", maxDiff)
	}
}
