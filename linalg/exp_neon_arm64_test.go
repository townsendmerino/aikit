//go:build arm64

package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

// expNEONSlice is the kernel over a whole slice, for tests. Length must be a
// multiple of 4 (the kernel's contract; the tail is the caller's job).
func expNEONSlice(src []float32) []float32 {
	dst := make([]float32, len(src))
	if len(src) > 0 {
		expF32ContractNEON(&dst[0], &src[0], len(src))
	}
	return dst
}

// TestExpF32NEON_bitIdenticalToContract is S-06 step 2's acceptance test, and it
// asserts RAW BITS rather than a tolerance. That is the whole point of the
// contract: the Go oracle reaches a correctly-rounded f32 FMA through f64 and
// FMLA is one directly, so "close" would mean something is wrong.
//
// The inputs are chosen to be hostile rather than representative — uniform
// sampling would spend almost all its time in the easy middle of the range:
//   - both range-reduction boundaries, where k ticks over
//   - exact powers of two and values straddling k's rounding boundary
//   - the underflow edge, where the kernel must FLUSH to zero like the scalar
//   - the overflow edge, where the two-step 2^k scaling replaces a branch
func TestExpF32NEON_bitIdenticalToContract(t *testing.T) {
	var xs []float32
	// the two guarded endpoints and their neighbourhoods
	for _, base := range []float64{expUnderflowF32, expOverflowF32, 0, 1, -1, 88, -87} {
		for d := -8; d <= 8; d++ {
			xs = append(xs, float32(base)+float32(d)*float32(math.Ldexp(1, -20)))
		}
	}
	// k-rounding boundaries: x near (n+0.5)*ln2 is where kf flips
	for n := -126; n <= 128; n++ {
		c := float64(n) * 0.6931471805599453
		for _, d := range []float64{-1e-5, -1e-7, 0, 1e-7, 1e-5, 0.3465735} {
			v := c + d
			if v >= expUnderflowF32 && v <= expOverflowF32 {
				xs = append(xs, float32(v))
			}
		}
	}
	// wide random fill across the whole legal range
	rng := rand.New(rand.NewPCG(0xe4b, 0x9c))
	for range 200_000 {
		xs = append(xs, float32(expUnderflowF32+rng.Float64()*(expOverflowF32-expUnderflowF32)))
	}
	for len(xs)%4 != 0 {
		xs = append(xs, 0)
	}

	got := expNEONSlice(xs)
	for i, x := range xs {
		want := expF32Contract(x)
		if math.Float32bits(got[i]) != math.Float32bits(want) {
			t.Fatalf("x=%v (bits %08x): NEON %v (%08x) != contract %v (%08x)",
				x, math.Float32bits(x), got[i], math.Float32bits(got[i]), want, math.Float32bits(want))
		}
	}
	t.Logf("bit-identical over %d inputs spanning [%g, %g]", len(xs), expUnderflowF32, expOverflowF32)
}

// TestExpF32NEON_productionRowLengths runs the lengths goinfer actually calls
// with, because a kernel that is right at 4096 and wrong at 8960 is a kernel
// nobody has tested. 1536 and 8960 are the 1.5B's hidden and intermediate
// widths; the rest are the attention depth ladder.
func TestExpF32NEON_productionRowLengths(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 13))
	for _, n := range []int{4, 8, 128, 256, 512, 1024, 1536, 2048, 3900, 4096, 8192, 8960} {
		n4 := n - n%4 // the kernel's contract; the Go caller mops the remainder
		src := make([]float32, n4)
		for i := range src {
			src[i] = float32(expUnderflowF32 + rng.Float64()*(expOverflowF32-expUnderflowF32))
		}
		got := expNEONSlice(src)
		for i, x := range src {
			if want := expF32Contract(x); math.Float32bits(got[i]) != math.Float32bits(want) {
				t.Fatalf("n=%d i=%d x=%v: NEON %v != contract %v", n4, i, x, got[i], want)
			}
		}
	}
}

// TestExpF32NEON_mutationDetected proves the gate above can fail. A kernel test
// that has never been made to go red is not evidence — perturbing one input by a
// single ULP must be caught, since that is the smallest error a wrong constant
// or a reversed operand could produce.
func TestExpF32NEON_mutationDetected(t *testing.T) {
	src := make([]float32, 64)
	for i := range src {
		src[i] = float32(i)*0.5 - 16
	}
	got := expNEONSlice(src)
	// perturb one lane by one ULP and confirm a raw-bit comparison rejects it
	mutated := append([]float32(nil), got...)
	mutated[7] = math.Float32frombits(math.Float32bits(mutated[7]) + 1)
	same := 0
	for i := range src {
		if math.Float32bits(mutated[i]) == math.Float32bits(expF32Contract(src[i])) {
			same++
		}
	}
	if same == len(src) {
		t.Fatal("a one-ULP mutation went undetected — the comparison is not raw-bit")
	}
	if same != len(src)-1 {
		t.Fatalf("expected exactly one mismatch after a one-lane mutation, got %d", len(src)-same)
	}
}

// TestExpF32NEON_overflowBranch targets the ONE place where the scalar and the
// kernel are structurally different rather than merely differently spelled. The
// scalar takes a branch at e >= 255 (k = 128) and builds 2^k in two steps to
// avoid encoding an Inf exponent; the kernel has no branch at all and always
// splits k into halves. Both are exact, so they must agree — but "must" is the
// kind of claim that deserves its own dense sweep rather than a hope that the
// random fill wandered through it.
//
// k reaches 128 only for x*log2e >= 127.5, i.e. x >= ~88.4, which is a 0.3-wide
// sliver at the very top of the legal range.
func TestExpF32NEON_overflowBranch(t *testing.T) {
	var xs []float32
	for x := 88.35; x <= expOverflowF32; x += 1e-5 {
		xs = append(xs, float32(x))
	}
	xs = append(xs, expOverflowF32)
	for len(xs)%4 != 0 {
		xs = append(xs, expOverflowF32)
	}
	got := expNEONSlice(xs)
	var sawBigK int
	for i, x := range xs {
		if int32(float32(float64(x)*float64(log2eF32))+roundMagicF32-roundMagicF32) >= 128 {
			sawBigK++
		}
		if want := expF32Contract(x); math.Float32bits(got[i]) != math.Float32bits(want) {
			t.Fatalf("x=%v: NEON %v (%08x) != contract %v (%08x)",
				x, got[i], math.Float32bits(got[i]), want, math.Float32bits(want))
		}
	}
	if sawBigK == 0 {
		t.Fatal("no input reached k=128 — the branch this test exists for was never exercised")
	}
	t.Logf("overflow branch agrees over %d inputs, %d of them with k=128", len(xs), sawBigK)
}
