//go:build goexperiment.simd

package elem

import (
	"math"
	"math/rand"
	"testing"

	"simd"
)

// ulpDiff32 is the distance in float32 representation order (0 = bit-equal).
func ulpDiff32(a, b float32) uint32 {
	ia, ib := int64(orderedBits32(a)), int64(orderedBits32(b))
	d := ia - ib
	if d < 0 {
		d = -d
	}
	return uint32(d)
}

func orderedBits32(f float32) uint32 {
	b := math.Float32bits(f)
	if b&0x80000000 != 0 {
		return ^b
	}
	return b | 0x80000000
}

// TestExpF32CoreVec_matchesScalarULP is expF32CoreVec's accuracy gate: the
// bound this GOEXPERIMENT=simd build depends on. expF32CoreVec's MulAdd is a
// fused op, and Go's scalar compiler already fuses p*r+c on arm64 but not
// amd64, so the vector kernel can differ from the scalar expF32Core by up to
// 1 ULP — validated against the prototype's own sweep
// (~/tmcode/go127-simd-audit, 2026-08-20: max 1 ULP over the full softmax
// domain on both apple-m1pro and nvidia-rtx2070s).
func TestExpF32CoreVec_matchesScalarULP(t *testing.T) {
	const n = 1 << 20
	c := newExpSIMDConsts()
	src := make([]float32, n)
	for i := range src {
		// Dense sweep of [-87.4, 0], crossing the underflow flush at -87.33654.
		src[i] = -87.4 * float32(i) / float32(n-1)
	}
	copy(src, []float32{0, -0.0, -1e-30, -87.33, -87.336, -87.337, -87.34, -50, -1, -0.5})

	got := make([]float32, n)
	var probe simd.Float32s
	L := probe.Len()
	i := 0
	for ; i+L <= n; i += L {
		expF32CoreVec(simd.LoadFloat32s(src[i:i+L]), &c).Store(got[i : i+L])
	}
	if i < n {
		v, _ := simd.LoadFloat32sPart(src[i:])
		expF32CoreVec(v, &c).StorePart(got[i:])
	}

	var maxULP uint32
	var argMax float32
	for i, x := range src {
		want := expF32Core(x)
		if x < expUnderflowF32 {
			if got[i] != 0 {
				t.Fatalf("x=%g: flush-to-zero expected, got %g", x, got[i])
			}
			continue
		}
		if d := ulpDiff32(got[i], want); d > maxULP {
			maxULP, argMax = d, x
		}
	}
	t.Logf("max vector-vs-scalar diff: %d ULP at x=%g", maxULP, argMax)
	if maxULP > 2 {
		t.Fatalf("max ULP diff %d at x=%g exceeds bound 2", maxULP, argMax)
	}
}

// TestExpF32Vec_matchesScalarULP is expF32Vec's accuracy gate — the full-range
// counterpart of TestExpF32CoreVec_matchesScalarULP, covering the overflow
// and NaN guards expF32CoreVec doesn't have, plus the e>=255 two-step scale
// boundary that guard's own domain (x <= 0) never reaches. Compares against
// scalar ExpF32 (not expF32Core), the actual full-contract reference.
func TestExpF32Vec_matchesScalarULP(t *testing.T) {
	const n = 1 << 20
	c := newExpSIMDConsts()
	rng := rand.New(rand.NewSource(31))
	src := make([]float32, n)
	for i := range src {
		// Dense sweep of the whole finite domain, both signs, so the e>=255
		// boundary (x close to expOverflowF32 from below) is actually hit.
		src[i] = float32(-104 + 293*rng.Float64())
	}
	copy(src, []float32{0, -0.0, expOverflowF32, expOverflowF32 - 0.001, expOverflowF32 + 0.001,
		200, -200, expUnderflowF32, expUnderflowF32 + 0.001, expUnderflowF32 - 0.001,
		88.376335, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))})

	got := make([]float32, n)
	var probe simd.Float32s
	L := probe.Len()
	i := 0
	for ; i+L <= n; i += L {
		expF32Vec(simd.LoadFloat32s(src[i:i+L]), &c).Store(got[i : i+L])
	}
	if i < n {
		v, _ := simd.LoadFloat32sPart(src[i:])
		expF32Vec(v, &c).StorePart(got[i:])
	}

	var maxULP uint32
	var argMax float32
	for i, x := range src {
		want := expF32(x)
		if x != x { // NaN: no ULP distance is meaningful
			if got[i] == got[i] {
				t.Fatalf("x=NaN: expF32Vec = %v, want NaN", got[i])
			}
			continue
		}
		if math.IsInf(float64(want), 0) {
			if got[i] != want {
				t.Fatalf("x=%g: expF32Vec = %v, want %v (Inf)", x, got[i], want)
			}
			continue
		}
		if d := ulpDiff32(got[i], want); d > maxULP {
			maxULP, argMax = d, x
		}
	}
	t.Logf("max vector-vs-scalar diff: %d ULP at x=%g", maxULP, argMax)
	if maxULP > 2 {
		t.Fatalf("max ULP diff %d at x=%g exceeds bound 2", maxULP, argMax)
	}
}

// TestSiLUIntoRaw_matchesScalar checks the SIMD SiLUInto against SiLUF32
// (the scalar per-element reference, same contract SiLUInto documents: up to
// a few ULP of drift is expected from expF32Vec's fused MulAdd, not a bug).
func TestSiLUIntoRaw_matchesScalar(t *testing.T) {
	const n = 1 << 18
	rng := rand.New(rand.NewSource(37))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(-120 + 240*rng.Float64())
	}
	copy(src, []float32{0, -0.0, 100, -100, 200, -200, 40, -40, expOverflowF32, expUnderflowF32})

	got := make([]float32, n)
	siluIntoRaw(got, src)

	var maxULP uint32
	var argMax float32
	for i, x := range src {
		want := SiLUF32(x)
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			t.Fatalf("x=%g: siluIntoRaw = %v, want finite", x, got[i])
		}
		if d := ulpDiff32(got[i], want); d > maxULP {
			maxULP, argMax = d, x
		}
	}
	t.Logf("max vector-vs-scalar diff: %d ULP at x=%g", maxULP, argMax)
	if maxULP > 4 {
		t.Fatalf("max ULP diff %d at x=%g exceeds bound 4", maxULP, argMax)
	}
}

// TestSiLUIntoRaw_tail walks every remainder length against SiLUF32 — the
// SIMD kernel's tail handling (LoadFloat32sPart/StorePart) is exactly what a
// fixed-size sweep under-covers, mirroring TestSoftmaxRowIntoRaw_tail below.
func TestSiLUIntoRaw_tail(t *testing.T) {
	var probe simd.Float32s
	L := probe.Len()
	rng := rand.New(rand.NewSource(41))
	for n := 1; n <= 3*L+1; n++ {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64()) * 30
		}
		got := make([]float32, n)
		siluIntoRaw(got, src)
		for i := range src {
			want := SiLUF32(src[i])
			if d := ulpDiff32(got[i], want); d > 4 {
				t.Fatalf("n=%d i=%d: got %g want %g (%d ULP)", n, i, got[i], want, d)
			}
		}
	}
}

// TestErfVec_matchesScalar checks erfVec against ErfF32 across both its
// branches (series |x|<1, A&S tail |x|>=1) and the saturation boundary at
// |x|=4, plus the sign flip — the seams a mask-select kernel can get wrong
// silently rather than crash on.
func TestErfVec_matchesScalar(t *testing.T) {
	const n = 1 << 20
	ec := newErfSIMDConsts()
	xc := newExpSIMDConsts()
	rng := rand.New(rand.NewSource(43))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(-9 + 18*rng.Float64())
	}
	copy(src, []float32{0, -0.0, 1, -1, 0.999999, 1.000001, 4, -4, 3.999999, 4.000001, 8, -8})

	got := make([]float32, n)
	var probe simd.Float32s
	L := probe.Len()
	i := 0
	for ; i+L <= n; i += L {
		erfVec(simd.LoadFloat32s(src[i:i+L]), &ec, &xc).Store(got[i : i+L])
	}
	if i < n {
		v, _ := simd.LoadFloat32sPart(src[i:])
		erfVec(v, &ec, &xc).StorePart(got[i:])
	}

	var worstAbs float64
	var argMax float32
	for i, x := range src {
		want := erfF32(x)
		if d := math.Abs(float64(got[i]) - float64(want)); d > worstAbs {
			worstAbs, argMax = d, x
		}
	}
	t.Logf("max vector-vs-scalar abs diff: %.3g at x=%g", worstAbs, argMax)
	if worstAbs > 2.5e-7 {
		t.Fatalf("max abs diff %.3g at x=%g exceeds bound 2.5e-07", worstAbs, argMax)
	}
}

// TestGELUIntoRaw_matchesScalar checks the SIMD GELUInto against GELUF32,
// against the SAME absolute bound TestErfF32_accuracy gates the scalar
// kernel with (1e-6) — GELU's contract is absolute, not relative (see
// GELUF32's doc comment for why).
func TestGELUIntoRaw_matchesScalar(t *testing.T) {
	const n = 1 << 18
	rng := rand.New(rand.NewSource(47))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(-9 + 18*rng.Float64())
	}
	copy(src, []float32{0, -0.0, 10, -10, 4, -4})

	got := make([]float32, n)
	geluIntoRaw(got, src)

	var worstAbs float64
	var argMax float32
	for i, x := range src {
		want := geluF32(x)
		if d := math.Abs(float64(got[i]) - float64(want)); d > worstAbs {
			worstAbs, argMax = d, x
		}
	}
	t.Logf("max vector-vs-scalar abs diff: %.3g at x=%g", worstAbs, argMax)
	if worstAbs > 1e-6 {
		t.Fatalf("max abs diff %.3g at x=%g exceeds bound 1e-06", worstAbs, argMax)
	}
}

// TestGELUIntoRaw_tail walks every remainder length against GELUF32,
// mirroring TestSiLUIntoRaw_tail's tail-length sweep for the same reason.
func TestGELUIntoRaw_tail(t *testing.T) {
	var probe simd.Float32s
	L := probe.Len()
	rng := rand.New(rand.NewSource(53))
	for n := 1; n <= 3*L+1; n++ {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64()) * 3
		}
		got := make([]float32, n)
		geluIntoRaw(got, src)
		for i := range src {
			want := geluF32(src[i])
			if d := math.Abs(float64(got[i]) - float64(want)); d > 1e-6 {
				t.Fatalf("n=%d i=%d: got %g want %g (diff %.3g)", n, i, got[i], want, d)
			}
		}
	}
}

// TestTanhVec_matchesScalar checks tanhVec against TanhF32 across both
// branches (series |x|<0.625, exponential |x|>=0.625) and the saturation
// boundary at |x|=9, plus the sign flip.
func TestTanhVec_matchesScalar(t *testing.T) {
	const n = 1 << 20
	tc := newTanhSIMDConsts()
	xc := newExpSIMDConsts()
	rng := rand.New(rand.NewSource(59))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(-20 + 40*rng.Float64())
	}
	copy(src, []float32{0, -0.0, 0.625, -0.625, 0.624999, 0.625001, 9, -9, 8.999999, 9.000001, 15, -15})

	got := make([]float32, n)
	var probe simd.Float32s
	L := probe.Len()
	i := 0
	for ; i+L <= n; i += L {
		tanhVec(simd.LoadFloat32s(src[i:i+L]), &tc, &xc).Store(got[i : i+L])
	}
	if i < n {
		v, _ := simd.LoadFloat32sPart(src[i:])
		tanhVec(v, &tc, &xc).StorePart(got[i:])
	}

	var maxULP uint32
	var argMax float32
	for i, x := range src {
		want := tanhF32(x)
		if d := ulpDiff32(got[i], want); d > maxULP {
			maxULP, argMax = d, x
		}
	}
	t.Logf("max vector-vs-scalar diff: %d ULP at x=%g", maxULP, argMax)
	if maxULP > 4 {
		t.Fatalf("max ULP diff %d at x=%g exceeds bound 4", maxULP, argMax)
	}
}

// TestGELUTanhIntoRaw_matchesScalar checks the SIMD GELUTanhInto against
// GELUTanhF32, against the same absolute bound TestGELUTanhF32_accuracy
// gates the scalar kernel with (1e-6).
func TestGELUTanhIntoRaw_matchesScalar(t *testing.T) {
	const n = 1 << 18
	rng := rand.New(rand.NewSource(61))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(-9 + 18*rng.Float64())
	}
	copy(src, []float32{0, -0.0, 10, -10})

	got := make([]float32, n)
	geluTanhIntoRaw(got, src)

	var worstAbs float64
	var argMax float32
	for i, x := range src {
		want := GELUTanhF32(x)
		if d := math.Abs(float64(got[i]) - float64(want)); d > worstAbs {
			worstAbs, argMax = d, x
		}
	}
	t.Logf("max vector-vs-scalar abs diff: %.3g at x=%g", worstAbs, argMax)
	if worstAbs > 1e-6 {
		t.Fatalf("max abs diff %.3g at x=%g exceeds bound 1e-06", worstAbs, argMax)
	}
}

// TestTanhIntoRaw_tail and TestGELUTanhIntoRaw_tail walk every remainder
// length against the scalar reference, mirroring the same tail-length
// sweep every kernel in this file gets.
func TestTanhIntoRaw_tail(t *testing.T) {
	var probe simd.Float32s
	L := probe.Len()
	rng := rand.New(rand.NewSource(67))
	for n := 1; n <= 3*L+1; n++ {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64()) * 6
		}
		got := make([]float32, n)
		tanhIntoRaw(got, src)
		for i := range src {
			want := tanhF32(src[i])
			if d := ulpDiff32(got[i], want); d > 4 {
				t.Fatalf("n=%d i=%d: got %g want %g (%d ULP)", n, i, got[i], want, d)
			}
		}
	}
}

func TestGELUTanhIntoRaw_tail(t *testing.T) {
	var probe simd.Float32s
	L := probe.Len()
	rng := rand.New(rand.NewSource(71))
	for n := 1; n <= 3*L+1; n++ {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64()) * 3
		}
		got := make([]float32, n)
		geluTanhIntoRaw(got, src)
		for i := range src {
			want := GELUTanhF32(src[i])
			if d := math.Abs(float64(got[i]) - float64(want)); d > 1e-6 {
				t.Fatalf("n=%d i=%d: got %g want %g (diff %.3g)", n, i, got[i], want, d)
			}
		}
	}
}

// TestSoftmaxRowIntoRaw_tail checks every remainder length against the
// non-SIMD reference (exp_scalar.go's algorithm, reproduced here since that
// file is excluded from this build) — the SIMD kernel's tail handling
// (LoadFloat32sPart/StorePart) is exactly the part a fixed-size sweep like
// TestSoftmaxRowInto_matchesReference (exp_test.go, runs in both builds)
// under-covers, since its n values don't walk every lane-count remainder.
func TestSoftmaxRowIntoRaw_tail(t *testing.T) {
	var probe simd.Float32s
	L := probe.Len()
	rng := rand.New(rand.NewSource(29))
	for n := 1; n <= 3*L+1; n++ {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64()) * 6
		}
		got := make([]float32, n)
		softmaxRowIntoRaw(got, src)

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
		for i := range ref {
			ref[i] /= sum
			if d := math.Abs(float64(got[i]) - ref[i]); d > 1e-6 {
				t.Fatalf("n=%d i=%d: got %g want %g (diff %g)", n, i, got[i], ref[i], d)
			}
		}
	}
}

// TestSoftmaxRowScaledIntoRaw_tail walks every remainder length, checking
// the SIMD fused kernel is bit-identical to the two-pass sequence at every
// tail width — the part-load/store handling both max-scan (pass 1) and the
// fused scale+exp (pass 2) have is exactly what a fixed-size sweep
// under-covers.
func TestSoftmaxRowScaledIntoRaw_tail(t *testing.T) {
	var probe simd.Float32s
	L := probe.Len()
	rng := rand.New(rand.NewSource(79))
	const scale = 0.0883883476 // 1/sqrt(128)
	for n := 1; n <= 3*L+1; n++ {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64()) * 6
		}
		got := make([]float32, n)
		softmaxRowScaledIntoRaw(got, src, scale)

		twoPass := make([]float32, n)
		for i, v := range src {
			twoPass[i] = v * scale
		}
		softmaxRowIntoRaw(twoPass, twoPass)

		for i := range got {
			if got[i] != twoPass[i] {
				t.Fatalf("n=%d i=%d: fused=%v two-pass=%v — NOT bit-identical", n, i, got[i], twoPass[i])
			}
		}
	}
}
