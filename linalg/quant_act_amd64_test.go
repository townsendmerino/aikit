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
