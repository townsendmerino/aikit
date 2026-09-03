package linalg

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// The activation quantizer is arch-dispatched (NEON on arm64, scalar elsewhere);
// quantizeRowInt8CoreScalar is the pre-vectorization body kept as the oracle. The
// contract is bit-identity — the same scale (as float32 bits) and the same codes —
// for every input, including the corners the scalar defines by accident of its
// comparisons: NaN skipped in the max and quantized to 0, -0.0 inert, ±Inf
// collapsing the row, exact .5 ties rounding away from zero, and magnitudes past
// int32 saturating into the clamp.

// quantRowCases enumerates lengths that cover the 16-wide vector body, the
// scalar tail, and both together — plus the production K's.
var quantRowCases = []int{1, 2, 3, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 100, 255, 256, 257, 768, 1536, 1537, 2048, 3072, 8960}

func quantRowFillers(rng *rand.Rand) map[string]func(n int) []float32 {
	return map[string]func(n int) []float32{
		"normal": func(n int) []float32 {
			r := make([]float32, n)
			for i := range r {
				r[i] = float32(rng.NormFloat64())
			}
			return r
		},
		"uniform-wide": func(n int) []float32 {
			r := make([]float32, n)
			for i := range r {
				r[i] = float32((rng.Float64()*2 - 1) * 1e3)
			}
			return r
		},
		"tiny": func(n int) []float32 { // denormal-adjacent magnitudes
			r := make([]float32, n)
			for i := range r {
				r[i] = float32(rng.NormFloat64()) * 1e-38
			}
			return r
		},
		"huge": func(n int) []float32 {
			r := make([]float32, n)
			for i := range r {
				r[i] = float32(rng.NormFloat64()) * 1e30
			}
			return r
		},
		"ties": func(n int) []float32 { // maxAbs 127 → inv exactly 1 → v is its own code
			r := make([]float32, n)
			for i := range r {
				r[i] = float32(rng.Intn(255)-127) + 0.5 // every value an exact .5 tie
			}
			if n > 0 {
				r[0] = 127 // pins the scale to exactly 1
			}
			return r
		},
		"integers": func(n int) []float32 {
			r := make([]float32, n)
			for i := range r {
				r[i] = float32(rng.Intn(2001) - 1000)
			}
			return r
		},
	}
}

func checkQuantRowIdentical(t *testing.T, name string, row []float32) {
	t.Helper()
	n := len(row)
	want := make([]int8, n)
	got := make([]int8, n)
	for i := range got {
		got[i] = 0x55 // a sentinel the kernel must overwrite
	}
	for _, zeroScale := range []float32{0, 1} {
		ws := quantizeRowInt8CoreScalar(row, want, zeroScale)
		gs := quantizeRowInt8Core(row, got, zeroScale)
		if math.Float32bits(ws) != math.Float32bits(gs) {
			t.Fatalf("%s n=%d: scale differs: scalar %v (%08x) dispatched %v (%08x)",
				name, n, ws, math.Float32bits(ws), gs, math.Float32bits(gs))
		}
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("%s n=%d: q[%d] differs: scalar %d dispatched %d (row[%d]=%v, scale=%v)",
					name, n, i, want[i], got[i], i, row[i], ws)
			}
		}
	}
}

func TestQuantizeRowInt8_bitIdenticalToScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5103))
	fillers := quantRowFillers(rng)
	for _, n := range quantRowCases {
		for name, fill := range fillers {
			checkQuantRowIdentical(t, name, fill(n))
		}
	}
}

func TestQuantizeRowInt8_corners(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	negZero := float32(math.Copysign(0, -1))
	base := func(n int) []float32 {
		r := make([]float32, n)
		for i := range r {
			r[i] = float32(i%23) - 11
		}
		return r
	}
	cases := map[string]func() []float32{
		"all-zero-16": func() []float32 { return make([]float32, 16) },
		"all-zero-19": func() []float32 { return make([]float32, 19) },
		"all-negzero": func() []float32 {
			r := make([]float32, 40)
			for i := range r {
				r[i] = negZero
			}
			return r
		},
		"all-nan": func() []float32 {
			r := make([]float32, 40)
			for i := range r {
				r[i] = nan
			}
			return r
		},
		"nan-in-body":         func() []float32 { r := base(64); r[5] = nan; r[37] = nan; return r },
		"nan-in-tail":         func() []float32 { r := base(37); r[35] = nan; return r },
		"nan-is-largest-slot": func() []float32 { r := base(32); r[0] = nan; r[1] = 1e30; return r },
		"posinf-in-body":      func() []float32 { r := base(64); r[9] = inf; return r },
		"neginf-in-tail":      func() []float32 { r := base(50); r[49] = -inf; return r },
		"both-inf":            func() []float32 { r := base(64); r[0] = inf; r[63] = -inf; return r },
		"negzero-scattered": func() []float32 {
			r := base(48)
			for i := 0; i < 48; i += 7 {
				r[i] = negZero
			}
			return r
		},
		"single-huge-rest-tiny": func() []float32 { r := base(64); r[20] = 3e38; return r },
		"max-float":             func() []float32 { r := base(32); r[3] = math.MaxFloat32; r[4] = -math.MaxFloat32; return r },
		"denormals": func() []float32 {
			r := make([]float32, 48)
			for i := range r {
				r[i] = math.Float32frombits(uint32(i + 1)) // smallest denormals
			}
			return r
		},
		"one-element-neg": func() []float32 { return []float32{-3} },
		"one-element-nan": func() []float32 { return []float32{nan} },
		"one-element-inf": func() []float32 { return []float32{inf} },
		"exact-ties-tail": func() []float32 {
			r := make([]float32, 21)
			r[0] = 127
			for i := 1; i < 21; i++ {
				r[i] = float32(i) - 10.5
			}
			return r
		},
		"scale-rounding": func() []float32 { // a maxAbs whose /127 and 1/s are inexact
			r := base(64)
			r[17] = 1234.5678
			return r
		},
	}
	for name, mk := range cases {
		checkQuantRowIdentical(t, name, mk())
	}
}

// The public entry points must see the same dispatch as the private core.
func TestQuantizeRowInt8_publicEntriesDispatch(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	row := make([]float32, 1536)
	for i := range row {
		row[i] = float32(rng.NormFloat64())
	}
	want := make([]int8, len(row))
	ws := quantizeRowInt8CoreScalar(row, want, 1)
	got := make([]int8, len(row))
	gs := QuantizeRowInt8(row, got)
	if ws != gs {
		t.Fatalf("QuantizeRowInt8 scale %v, scalar %v", gs, ws)
	}
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("QuantizeRowInt8 q[%d]=%d, scalar %d", i, got[i], want[i])
		}
	}
	aq := make([]int8, 3*len(row))
	scales := make([]float32, 3)
	a := append(append(append([]float32{}, row...), row...), row...)
	QuantizeActivationsInto(aq, scales, a, 3, len(row))
	for r := range 3 {
		if scales[r] != ws {
			t.Fatalf("QuantizeActivationsInto row %d scale %v, scalar %v", r, scales[r], ws)
		}
		for i := range want {
			if aq[r*len(row)+i] != want[i] {
				t.Fatalf("QuantizeActivationsInto row %d q[%d]=%d, scalar %d", r, i, aq[r*len(row)+i], want[i])
			}
		}
	}
}

func BenchmarkQuantizeRowInt8(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	for _, k := range []int{1536, 8960} {
		row := make([]float32, k)
		for i := range row {
			row[i] = float32(rng.NormFloat64())
		}
		q := make([]int8, k)
		b.Run(fmt.Sprintf("dispatched/K%d", k), func(b *testing.B) {
			b.SetBytes(int64(k) * 4)
			for b.Loop() {
				quantizeRowInt8Core(row, q, 0)
			}
		})
		b.Run(fmt.Sprintf("scalar/K%d", k), func(b *testing.B) {
			b.SetBytes(int64(k) * 4)
			for b.Loop() {
				quantizeRowInt8CoreScalar(row, q, 0)
			}
		})
	}
}
