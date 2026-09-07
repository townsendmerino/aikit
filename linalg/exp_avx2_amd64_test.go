//go:build amd64

package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestExpF32AVX2_bitIdenticalToContract is the amd64 acceptance test, and it
// asserts the same raw bits the arm64 kernel does — which means, transitively,
// that the two KERNELS agree with each other. That cross-architecture identity
// is what the whole contract was built to buy: a golden generated on an M1 is
// valid on a Zen 2.
func TestExpF32AVX2_bitIdenticalToContract(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2+FMA3 on this core; the kernel does not dispatch")
	}
	var xs []float32
	for _, base := range []float64{expUnderflowF32, expOverflowF32, 0, 1, -1, 88, -87} {
		for d := -8; d <= 8; d++ {
			xs = append(xs, float32(base)+float32(d)*float32(math.Ldexp(1, -20)))
		}
	}
	for n := -126; n <= 128; n++ {
		c := float64(n) * 0.6931471805599453
		for _, d := range []float64{-1e-5, -1e-7, 0, 1e-7, 1e-5, 0.3465735} {
			if v := c + d; v >= expUnderflowF32 && v <= expOverflowF32 {
				xs = append(xs, float32(v))
			}
		}
	}
	rng := rand.New(rand.NewPCG(0xa64, 0x2))
	for range 200_000 {
		xs = append(xs, float32(expUnderflowF32+rng.Float64()*(expOverflowF32-expUnderflowF32)))
	}
	for len(xs)%8 != 0 {
		xs = append(xs, 0)
	}
	dst := make([]float32, len(xs))
	expF32ContractAVX2(&dst[0], &xs[0], len(xs))
	for i, x := range xs {
		if want := expF32Contract(x); math.Float32bits(dst[i]) != math.Float32bits(want) {
			t.Fatalf("x=%v (bits %08x): AVX2 %v (%08x) != contract %v (%08x)",
				x, math.Float32bits(x), dst[i], math.Float32bits(dst[i]), want, math.Float32bits(want))
		}
	}
	t.Logf("bit-identical over %d inputs spanning [%g, %g]", len(xs), expUnderflowF32, expOverflowF32)
}

// TestExpF32AVX2_overflowBranch sweeps the sliver where k reaches 128 and the
// branchless two-step 2^k replaces the scalar's branch — the same targeted check
// the arm64 kernel gets, for the same reason: the random fill would not prove
// the branch was exercised.
func TestExpF32AVX2_overflowBranch(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2+FMA3")
	}
	var xs []float32
	for x := 88.35; x <= expOverflowF32; x += 1e-5 {
		xs = append(xs, float32(x))
	}
	for len(xs)%8 != 0 {
		xs = append(xs, expOverflowF32)
	}
	dst := make([]float32, len(xs))
	expF32ContractAVX2(&dst[0], &xs[0], len(xs))
	var big int
	for i, x := range xs {
		if int32(float32(float64(x)*float64(log2eF32))+roundMagicF32-roundMagicF32) >= 128 {
			big++
		}
		if want := expF32Contract(x); math.Float32bits(dst[i]) != math.Float32bits(want) {
			t.Fatalf("x=%v: AVX2 %v != contract %v", x, dst[i], want)
		}
	}
	if big == 0 {
		t.Fatal("no input reached k=128 — the branch this test exists for was never exercised")
	}
	t.Logf("overflow branch agrees over %d inputs, %d with k=128", len(xs), big)
}

// TestTanhF32AVX2_bitIdenticalToContract is the tanh acceptance test, and like
// the exp one it asserts raw bits against the scalar contract — which makes it,
// transitively, an assertion that the AVX2 and NEON tanh kernels agree.
//
// The input set deliberately piles up where a branchless blend can go wrong: on
// both sides of the 0.625 branch point, inside the (9, 9.02) band where this form
// saturates LATER than TanhF32 by one ULP, at the signed zeros, and out past the
// 2|x| clamp where the unused polynomial half diverges.
func TestTanhF32AVX2_bitIdenticalToContract(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2+FMA3 on this core; the kernel does not dispatch")
	}
	var xs []float32
	ulp := float32(math.Ldexp(1, -20))
	for _, base := range []float64{0, 0.625, -0.625, 1, -1, 9, -9, 9.02, -9.02, 4, 20, -20, 88, -88, 1e3, -1e3} {
		for d := -16; d <= 16; d++ {
			xs = append(xs, float32(base)+float32(d)*ulp)
		}
	}
	xs = append(xs, 0, float32(math.Copysign(0, -1)))
	rng := rand.New(rand.NewPCG(0x7a41, 0x5))
	for range 200_000 {
		xs = append(xs, float32((rng.Float64()*2-1)*12))
	}
	for len(xs)%8 != 0 {
		xs = append(xs, 0)
	}
	dst := make([]float32, len(xs))
	tanhF32ContractAVX2(&dst[0], &xs[0], len(xs))
	for i, x := range xs {
		if want := tanhF32Contract(x); math.Float32bits(dst[i]) != math.Float32bits(want) {
			t.Fatalf("x=%v (bits %08x): AVX2 %v (%08x) != contract %v (%08x)",
				x, math.Float32bits(x), dst[i], math.Float32bits(dst[i]), want, math.Float32bits(want))
		}
	}
	t.Logf("bit-identical over %d inputs", len(xs))
}

// TestErfF32AVX2_bitIdenticalToContract is the erf acceptance test. The input set
// straddles the |x| = 1 split between the Maclaurin series and the A&S tail, the
// |x| ≈ 4 point where the tail saturates to exactly 1 on its own, and the large
// arguments where -x² has to be clamped before the exponent construction sees it.
func TestErfF32AVX2_bitIdenticalToContract(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2+FMA3 on this core; the kernel does not dispatch")
	}
	var xs []float32
	ulp := float32(math.Ldexp(1, -20))
	for _, base := range []float64{0, 0.5, -0.5, 1, -1, 2, -2, 4, -4, 6, -6, 10, -10, 100, -100} {
		for d := -16; d <= 16; d++ {
			xs = append(xs, float32(base)+float32(d)*ulp)
		}
	}
	xs = append(xs, 0, float32(math.Copysign(0, -1)))
	rng := rand.New(rand.NewPCG(0xe4f0, 0x7))
	for range 200_000 {
		xs = append(xs, float32((rng.Float64()*2-1)*8))
	}
	for len(xs)%8 != 0 {
		xs = append(xs, 0)
	}
	dst := make([]float32, len(xs))
	erfF32ContractAVX2(&dst[0], &xs[0], len(xs))
	for i, x := range xs {
		if want := erfF32Contract(x); math.Float32bits(dst[i]) != math.Float32bits(want) {
			t.Fatalf("x=%v (bits %08x): AVX2 %v (%08x) != contract %v (%08x)",
				x, math.Float32bits(x), dst[i], math.Float32bits(dst[i]), want, math.Float32bits(want))
		}
	}
	t.Logf("bit-identical over %d inputs", len(xs))
}
