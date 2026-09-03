//go:build arm64

package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// The raw NEON kernels against the scalar passes, so a dispatch mistake in the
// wrapper cannot hide a kernel defect behind a scalar fallback (and vice versa).
func TestQuantActNEONKernels_matchScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5104))
	nan := float32(math.NaN())
	for _, n := range []int{16, 32, 48, 64, 256, 1536, 8960} {
		row := make([]float32, n)
		for i := range row {
			row[i] = float32(rng.NormFloat64()) * 37
		}
		row[n/2] = nan // must be skipped by the max and quantize to 0
		row[n/3] = float32(math.Copysign(0, -1))

		wantMax := maxAbsF32Scalar(row, 0)
		gotMax := maxAbsF32NEON(&row[0], n)
		if math.Float32bits(wantMax) != math.Float32bits(gotMax) {
			t.Fatalf("n=%d maxAbs: NEON %v scalar %v", n, gotMax, wantMax)
		}
		inv := 1 / (wantMax / 127)
		want := make([]int8, n)
		quantizeRowScaledScalar(row, want, inv)
		got := make([]int8, n)
		quantizeF32NEON(&row[0], &got[0], n, inv)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("n=%d q[%d]: NEON %d scalar %d (row=%v)", n, i, got[i], want[i], row[i])
			}
		}
	}
}
