package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestContractDispatch_matchesScalarEverywhere runs on EVERY architecture and is
// the portable half of the contract's guarantee: whatever implementation the
// build selected must agree bit for bit with the scalar reference.
//
// On arm64 this re-checks the NEON kernels through the public entry points; on
// amd64 the two sides are the same code and it degenerates to a consistency
// check. That asymmetry is fine and is the point — the test that matters is the
// one that runs wherever the binary is built, rather than only where the kernel
// happens to exist.
func TestContractDispatch_matchesScalarEverywhere(t *testing.T) {
	rng := rand.New(rand.NewPCG(0xd15, 0xa7c))
	for _, n := range []int{0, 1, 3, 4, 7, 16, 17, 1536, 8960} {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64() * 12)
		}

		gotE := make([]float32, n)
		wantE := make([]float32, n)
		ExpContractInto(gotE, src)
		expContractScalarInto(wantE, src)
		for i := range gotE {
			if math.Float32bits(gotE[i]) != math.Float32bits(wantE[i]) {
				t.Fatalf("exp n=%d i=%d x=%v: dispatch %v != scalar %v", n, i, src[i], gotE[i], wantE[i])
			}
		}

		gotS := make([]float32, n)
		wantS := make([]float32, n)
		SiLUContractInto(gotS, src)
		siluContractScalarInto(wantS, src)
		for i := range gotS {
			if math.Float32bits(gotS[i]) != math.Float32bits(wantS[i]) {
				t.Fatalf("silu n=%d i=%d x=%v: dispatch %v != scalar %v", n, i, src[i], gotS[i], wantS[i])
			}
		}

		if n > 0 {
			gotM := make([]float32, n)
			wantM := make([]float32, n)
			SoftmaxRowContractInto(gotM, src)
			softmaxRowContract(wantM, src)
			for i := range gotM {
				if math.Float32bits(gotM[i]) != math.Float32bits(wantM[i]) {
					t.Fatalf("softmax n=%d i=%d: dispatch %v != scalar %v", n, i, gotM[i], wantM[i])
				}
			}
		}

		gotG := make([]float32, n)
		GELUTanhContractInto(gotG, src)
		for i, v := range src {
			if want := geluTanhF32Contract(v); math.Float32bits(gotG[i]) != math.Float32bits(want) {
				t.Fatalf("geluTanh n=%d i=%d: %v != %v", n, i, gotG[i], want)
			}
		}
	}
}

// TestContractDispatch_specials pins that the guarded entry points handle the
// values a kernel with no range checks cannot: NaN, ±Inf, and both sides of the
// overflow/underflow thresholds. On arm64 these must route around the NEON path,
// and getting that wrong is the classic way a fast path corrupts an edge case.
func TestContractDispatch_specials(t *testing.T) {
	inf := float32(math.Inf(1))
	src := []float32{
		float32(math.NaN()), inf, -inf, 0, -0,
		expOverflowF32, expOverflowF32 + 1, expUnderflowF32, expUnderflowF32 - 1,
		100, -100, 1, -1,
	}
	for len(src)%4 != 0 {
		src = append(src, 0)
	}
	got := make([]float32, len(src))
	want := make([]float32, len(src))
	ExpContractInto(got, src)
	expContractScalarInto(want, src)
	for i, x := range src {
		g, w := got[i], want[i]
		if math.IsNaN(float64(g)) != math.IsNaN(float64(w)) {
			t.Fatalf("x=%v: NaN-ness differs (%v vs %v)", x, g, w)
		}
		if !math.IsNaN(float64(g)) && math.Float32bits(g) != math.Float32bits(w) {
			t.Fatalf("x=%v: dispatch %v (%08x) != scalar %v (%08x)", x, g, math.Float32bits(g), w, math.Float32bits(w))
		}
	}
}

// TestContractDispatch_inPlace pins that dst may alias src. The arm64 GELU path
// chunks through a stack buffer and writes dst while still reading src, so
// aliasing is a real hazard there rather than a hypothetical one — and an
// in-place caller is the natural way to use these (goinfer's activation loops
// overwrite the gate row).
func TestContractDispatch_inPlace(t *testing.T) {
	rng := rand.New(rand.NewPCG(0xa11a, 0x5))
	for _, n := range []int{1, 4, 7, 255, 256, 257, 1000, 8960} {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64() * 5)
		}
		for _, tc := range []struct {
			name string
			fn   func(dst, src []float32)
		}{
			{"exp", ExpContractInto},
			{"silu", SiLUContractInto},
			{"gelutanh", GELUTanhContractInto},
		} {
			want := make([]float32, n)
			tc.fn(want, src)
			inplace := append([]float32(nil), src...)
			tc.fn(inplace, inplace)
			for i := range want {
				if math.Float32bits(inplace[i]) != math.Float32bits(want[i]) {
					t.Fatalf("%s n=%d i=%d: in-place %v != separate %v", tc.name, n, i, inplace[i], want[i])
				}
			}
		}
	}
}
