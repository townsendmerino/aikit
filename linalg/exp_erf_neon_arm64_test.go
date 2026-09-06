//go:build arm64

package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

func erfNEONSlice(src []float32) []float32 {
	dst := make([]float32, len(src))
	if n := len(src) &^ 3; n > 0 {
		erfF32ContractNEON(&dst[0], &src[0], n)
	}
	for i := len(src) &^ 3; i < len(src); i++ {
		dst[i] = erfF32Contract(src[i])
	}
	return dst
}

// TestErfContractNEON_bitIdenticalToScalar gates the two-branch blend. Dense
// around |x| = 1 on both signs — the only place the select can be wrong while
// the rest looks right — plus the region past 4 where the tail must reach
// exactly 1 on its own, and the far tail where the exponent clamp is what stops
// the kernel's 2^k construction going out of range.
func TestErfContractNEON_bitIdenticalToScalar(t *testing.T) {
	var xs []float32
	for d := -600; d <= 600; d++ { // ±0.6 around the branch handover
		xs = append(xs, float32(1)+float32(d)*1e-3, -(float32(1) + float32(d)*1e-3))
	}
	for _, b := range []float64{0, 2, 3, 4, 4.5, 6, 10, 50, 200} {
		for d := -4; d <= 4; d++ {
			xs = append(xs, float32(b)+float32(d)*0.05, -(float32(b) + float32(d)*0.05))
		}
	}
	rng := rand.New(rand.NewPCG(0xe4f, 0x2))
	for range 200_000 {
		xs = append(xs, float32(rng.NormFloat64()*3))
	}
	for len(xs)%4 != 0 {
		xs = append(xs, 0)
	}
	got := erfNEONSlice(xs)
	for i, x := range xs {
		if want := erfF32Contract(x); math.Float32bits(got[i]) != math.Float32bits(want) {
			t.Fatalf("x=%v: NEON %v (%08x) != scalar %v (%08x)",
				x, got[i], math.Float32bits(got[i]), want, math.Float32bits(want))
		}
	}
	t.Logf("bit-identical over %d inputs, dense across |x|=1 and out to |x|=200", len(xs))
}

// TestErfContract_sanity pins erf's shape independently of the oracle: odd,
// bounded, monotone, and close to math.Erf. Without this, a kernel and a
// reference that agreed on the same mistake would both pass.
func TestErfContract_sanity(t *testing.T) {
	if v := erfF32Contract(0); math.Float32bits(v) != 0 {
		t.Errorf("erf(0) = %v, want +0", v)
	}
	prev := float32(-2)
	var maxAbs float64
	for x := -6.0; x <= 6.0; x += 0.001 {
		v := erfF32Contract(float32(x))
		if v < -1 || v > 1 {
			t.Fatalf("erf(%v) = %v, outside [-1,1]", x, v)
		}
		if v < prev {
			t.Fatalf("erf not monotone at x=%v: %v < %v", x, v, prev)
		}
		prev = v
		if d := math.Abs(float64(v) - math.Erf(float64(float32(x)))); d > maxAbs {
			maxAbs = d
		}
		if o := erfF32Contract(float32(-x)); math.Float32bits(o) != math.Float32bits(-v) {
			t.Fatalf("erf not odd at x=%v: f(-x)=%v, -f(x)=%v", x, o, -v)
		}
	}
	t.Logf("erf max abs error vs math.Erf over [-6,6]: %.3g", maxAbs)
	if maxAbs > 5e-7 {
		t.Fatalf("erf accuracy %.3g exceeds the sanity ceiling", maxAbs)
	}
	for _, x := range []float32{4.5, 6, 10, 100} {
		if v := erfF32Contract(x); v != 1 {
			t.Errorf("erf(%v) = %v, want exactly 1", x, v)
		}
	}
}
