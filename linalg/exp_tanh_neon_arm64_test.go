//go:build arm64

package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

func tanhNEONSlice(src []float32) []float32 {
	dst := make([]float32, len(src))
	if n := len(src) &^ 3; n > 0 {
		tanhF32ContractNEON(&dst[0], &src[0], n)
	}
	for i := len(src) &^ 3; i < len(src); i++ {
		dst[i] = tanhF32Contract(src[i])
	}
	return dst
}

// TestTanhContractNEON_bitIdenticalToScalar gates the branchless two-branch
// blend. The sweep is dense AROUND THE 0.625 HANDOVER on both sides, because
// that is the only place the select can be wrong while everything else looks
// right — a reversed VBSL would show there and almost nowhere else.
func TestTanhContractNEON_bitIdenticalToScalar(t *testing.T) {
	var xs []float32
	for d := -400; d <= 400; d++ { // ±0.4 around the handover, both signs
		xs = append(xs, float32(0.625)+float32(d)*1e-3, -(float32(0.625) + float32(d)*1e-3))
	}
	for _, b := range []float64{0, 9, -9, 20, -20, 88, -88, 200, -200} {
		for d := -4; d <= 4; d++ {
			xs = append(xs, float32(b)+float32(d)*0.125)
		}
	}
	rng := rand.New(rand.NewPCG(0x7a, 0x17))
	for range 200_000 {
		xs = append(xs, float32(rng.NormFloat64()*6))
	}
	for len(xs)%4 != 0 {
		xs = append(xs, 0)
	}
	got := tanhNEONSlice(xs)
	for i, x := range xs {
		if want := tanhF32Contract(x); math.Float32bits(got[i]) != math.Float32bits(want) {
			t.Fatalf("x=%v: NEON %v (%08x) != scalar %v (%08x)",
				x, got[i], math.Float32bits(got[i]), want, math.Float32bits(want))
		}
	}
	t.Logf("bit-identical over %d inputs, dense across the 0.625 branch handover", len(xs))
}

// TestTanhContract_sanity pins the function's shape independently of the oracle,
// so two paths agreeing on nonsense still fails: odd, bounded by ±1, monotone,
// and matching the f64 definition.
func TestTanhContract_sanity(t *testing.T) {
	if v := tanhF32Contract(0); math.Float32bits(v) != 0 {
		t.Errorf("tanh(0) = %v, want +0", v)
	}
	prev := float32(-2)
	for x := -12.0; x <= 12.0; x += 0.001 {
		v := tanhF32Contract(float32(x))
		if v < -1 || v > 1 {
			t.Fatalf("tanh(%v) = %v, outside [-1,1]", x, v)
		}
		if v < prev {
			t.Fatalf("tanh not monotone at x=%v: %v < %v", x, v, prev)
		}
		prev = v
		if d := math.Abs(float64(v) - math.Tanh(float64(float32(x)))); d > 2e-7 {
			t.Fatalf("tanh(%v) = %v, f64 gives %v (diff %g)", x, v, math.Tanh(x), d)
		}
		// oddness
		if o := tanhF32Contract(float32(-x)); math.Float32bits(o) != math.Float32bits(-v) {
			t.Fatalf("tanh not odd at x=%v: f(-x)=%v, -f(x)=%v", x, o, -v)
		}
	}
	// Saturation arrives at |x| >= 9.02, one ULP later than TanhF32's explicit
	// x>9 branch; see tanhF32Contract. Asserting exactness at 9 would be
	// asserting a branch this deliberately does not have.
	for _, x := range []float32{9.02, 9.5, 20, 100} {
		if v := tanhF32Contract(x); v != 1 {
			t.Errorf("tanh(%v) = %v, want exactly 1", x, v)
		}
	}
	if v := tanhF32Contract(9); v != math.Float32frombits(math.Float32bits(1)-1) {
		t.Errorf("tanh(9) = %v, want exactly one ULP below 1 (the documented gap)", v)
	}
}
