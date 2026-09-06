//go:build arm64

package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

func siluNEONSlice(src []float32) []float32 {
	dst := make([]float32, len(src))
	if n := len(src) &^ 3; n > 0 {
		siluF32ContractNEON(&dst[0], &src[0], n)
	}
	for i := len(src) &^ 3; i < len(src); i++ {
		dst[i] = siluF32Contract(src[i])
	}
	return dst
}

// TestSiLUContractNEON_bitIdenticalToScalar gates the SwiGLU activation on raw
// bits. The input sweep deliberately straddles both clamp boundaries — a kernel
// that got the clamp wrong would still look correct across the ordinary range,
// and the failure would only appear on a real model's most negative gate values.
func TestSiLUContractNEON_bitIdenticalToScalar(t *testing.T) {
	var xs []float32
	for _, b := range []float64{0, 1, -1, 10, -10, 88.72283, -88.72283, 104, -104, 89.4, -89.4} {
		for d := -6; d <= 6; d++ {
			xs = append(xs, float32(b)+float32(d)*0.25)
		}
	}
	rng := rand.New(rand.NewPCG(0x51, 0x1c))
	for range 200_000 {
		xs = append(xs, float32(rng.NormFloat64()*30))
	}
	for range 2000 { // deep tails, where the clamp decides the answer
		xs = append(xs, float32(rng.NormFloat64()*400))
	}
	for len(xs)%4 != 0 {
		xs = append(xs, 0)
	}
	got := siluNEONSlice(xs)
	for i, x := range xs {
		if want := siluF32Contract(x); math.Float32bits(got[i]) != math.Float32bits(want) {
			t.Fatalf("x=%v: NEON %v (%08x) != scalar %v (%08x)",
				x, got[i], math.Float32bits(got[i]), want, math.Float32bits(want))
		}
	}
	t.Logf("bit-identical over %d inputs including both clamp boundaries", len(xs))
}

// TestSiLUContract_sanity pins the shape of the function itself, so a kernel
// that is self-consistently WRONG (both paths agreeing on nonsense) still fails.
// Bit-identity to an oracle proves agreement, not correctness.
func TestSiLUContract_sanity(t *testing.T) {
	if v := siluF32Contract(0); v != 0 {
		t.Errorf("silu(0) = %v, want 0", v)
	}
	// large positive: silu(x) -> x
	for _, x := range []float32{20, 50, 100} {
		if v := siluF32Contract(x); math.Abs(float64(v-x)) > 1e-3 {
			t.Errorf("silu(%v) = %v, want ~%v", x, v, x)
		}
	}
	// large negative: silu(x) -> 0 from below
	for _, x := range []float32{-20, -50, -100} {
		v := siluF32Contract(x)
		if v > 0 || v < -1e-6 {
			t.Errorf("silu(%v) = %v, want a small negative", x, v)
		}
	}
	// monotone-ish and matching the f64 definition within a loose bound
	for x := -30.0; x <= 30.0; x += 0.01 {
		want := x / (1 + math.Exp(-x))
		got := float64(siluF32Contract(float32(x)))
		if math.Abs(got-want) > 1e-5*math.Max(1, math.Abs(want)) {
			t.Fatalf("silu(%v) = %v, f64 definition gives %v", x, got, want)
		}
	}
}
