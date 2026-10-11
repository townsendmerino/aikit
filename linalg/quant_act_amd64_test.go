//go:build amd64

package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// The raw AVX2 kernels against the scalar passes, so a dispatch mistake in the wrapper cannot hide
// a kernel defect behind a scalar fallback (and vice versa) — the amd64 twin of
// TestQuantActNEONKernels_matchScalar (quant_act_arm64_test.go).
func TestQuantActAVX2Kernels_matchScalar(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU (or none usable by the OS): the raw kernels would be an illegal instruction here, which is what the wrappers' check exists to prevent")
	}
	rng := rand.New(rand.NewSource(0x5104))
	nan := float32(math.NaN())
	for _, n := range []int{8, 16, 24, 32, 40, 256, 1536, 8960, 18944} {
		row := make([]float32, n)
		for i := range row {
			row[i] = float32(rng.NormFloat64()) * 37
		}
		row[n/2] = nan // must be skipped by the max and quantize to 0
		row[n/3] = float32(math.Copysign(0, -1))

		wantMax := maxAbsF32Scalar(row, 0)
		gotMax := maxAbsF32AVX2(&row[0], n)
		if math.Float32bits(wantMax) != math.Float32bits(gotMax) {
			t.Fatalf("n=%d maxAbs: AVX2 %v (%08x) scalar %v (%08x)", n, gotMax, math.Float32bits(gotMax), wantMax, math.Float32bits(wantMax))
		}
		inv := 1 / (wantMax / 127)
		want := make([]int8, n)
		quantizeRowScaledScalar(row, want, inv)
		got := make([]int8, n)
		quantizeF32AVX2(&row[0], &got[0], n, inv)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("n=%d q[%d]: AVX2 %d scalar %d (row[%d]=%v, inv=%v)", n, i, got[i], want[i], i, row[i], inv)
			}
		}
	}
}

// Direct-kernel adversarial (row, inv) pairs — NOT self-consistently derived the way
// quantizeRowInt8Core's real callers always compute inv, which is why the arch-independent
// quant_act_test.go corners (posinf-in-body etc.) never actually push a genuine ±Inf into the
// round/clamp/truncate pipeline (a row containing Inf collapses inv to 0 first). This test drives
// the raw kernel directly with a plain, unrelated inv so a real ±Inf DOES reach quantizeF32AVX2,
// exercising the pre-truncate float clamp quant_act_amd64.go's doc comment names — belt and
// suspenders for a building block wider callers than today's one may eventually use directly.
func TestQuantActAVX2Kernels_adversarialInfDoesNotFlipSign(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU (or none usable by the OS)")
	}
	posInf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))
	nan := float32(math.NaN())
	const n = 8
	row := []float32{posInf, negInf, nan, 1, -1, 0, 63.5, -63.5}
	const inv = 1.0 // an ordinary, unrelated scale -- NOT derived from this row's own max
	want := make([]int8, n)
	quantizeRowScaledScalar(row, want, inv)
	got := make([]int8, n)
	quantizeF32AVX2(&row[0], &got[0], n, inv)
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("q[%d]: AVX2 %d scalar %d (row[%d]=%v)", i, got[i], want[i], i, row[i])
		}
	}
	if want[0] != 127 {
		t.Fatalf("test premise broke: scalar itself no longer clamps +Inf to +127 (got %d) -- update the test, not the kernel", want[0])
	}
	if want[1] != -127 {
		t.Fatalf("test premise broke: scalar itself no longer clamps -Inf to -127 (got %d)", want[1])
	}
}

// TestQuantizeRowInt8_withoutAVX2 holds the activation quantizer's entry points to the scalar oracle with hasAVX2 false: quantizeRowInt8Core, maxAbsF32 and
// quantizeRowScaled return the scalar passes' bytes and scale on rows whose lengths are and are not multiples of 8, NaN, +-Inf and -0 included.
//
// WHAT IT DOES NOT SHOW. On an AVX2 CPU the kernel and the scalar give the same bytes, so this test passes with the hasAVX2 check removed from maxAbsF32 and
// quantizeRowScaled: it cannot tell whether the kernel was called. The check is proved on a CPU model that lacks AVX, by running this package's test
// binary under Intel SDE (`sde64 -nhm -- ./linalg.test`), where calling the kernel is an illegal instruction (the procedure is in the v1.67.1 CHANGELOG entry). This test is the bytes-equal half; the
// emulated run is the was-it-called half.
func TestQuantizeRowInt8_withoutAVX2(t *testing.T) {
	saved := hasAVX2
	hasAVX2 = false
	defer func() { hasAVX2 = saved }()

	rng := rand.New(rand.NewSource(0x5105))
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	for _, n := range []int{1, 7, 8, 9, 16, 31, 64, 768, 1000} {
		row := make([]float32, n)
		for i := range row {
			row[i] = float32(rng.NormFloat64()) * 29
		}
		if n >= 8 {
			row[1], row[n/2], row[n-1] = nan, -inf, float32(math.Copysign(0, -1))
		}
		if got, want := maxAbsF32(row), maxAbsF32Scalar(row, 0); math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("n=%d maxAbsF32 %v, scalar %v", n, got, want)
		}
		gotQ, wantQ := make([]int8, n), make([]int8, n)
		gotS := quantizeRowInt8Core(row, gotQ, 0)
		wantS := quantizeRowInt8CoreScalar(row, wantQ, 0)
		if math.Float32bits(gotS) != math.Float32bits(wantS) {
			t.Fatalf("n=%d scale %v, scalar %v", n, gotS, wantS)
		}
		for i := range gotQ {
			if gotQ[i] != wantQ[i] {
				t.Fatalf("n=%d q[%d] = %d, scalar %d (row[%d]=%v)", n, i, gotQ[i], wantQ[i], i, row[i])
			}
		}
		inv := float32(127) / 31
		gotR, wantR := make([]int8, n), make([]int8, n)
		quantizeRowScaled(row, gotR, inv)
		quantizeRowScaledScalar(row, wantR, inv)
		for i := range gotR {
			if gotR[i] != wantR[i] {
				t.Fatalf("n=%d quantizeRowScaled q[%d] = %d, scalar %d", n, i, gotR[i], wantR[i])
			}
		}
	}
}
