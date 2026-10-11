//go:build amd64

package linalg

import (
	"math/rand"
	"testing"
)

// TestQuantizeGroupInt4Row_withoutAVX2 holds the amd64 path to the CPU check: with hasAVX2 false no group is offered
// to the AVX2 kernel, and a row quantizes to the scalar oracle's bytes and scales.
func TestQuantizeGroupInt4Row_withoutAVX2(t *testing.T) {
	saved := hasAVX2
	hasAVX2 = false
	defer func() { hasAVX2 = saved }()

	for _, n := range []int{8, 32, 64} {
		if int4GroupHasKernel(0, n) {
			t.Fatalf("int4GroupHasKernel(0, %d) is true with hasAVX2 false: the AVX2 kernel would run on a CPU without it", n)
		}
	}
	const cols, group = 256, 32
	rng := rand.New(rand.NewSource(7))
	row := make([]float32, cols)
	for i := range row {
		row[i] = float32(rng.NormFloat64())
	}
	got, want := make([]byte, cols/2), make([]byte, cols/2)
	gotS, wantS := make([]float32, cols/group), make([]float32, cols/group)
	QuantizeGroupInt4Row(row, cols, group, got, gotS)
	quantizeGroupInt4RowScalar(row, cols, group, want, wantS)
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("byte %d is %#x, the scalar's %#x", i, got[i], want[i])
		}
	}
	for g := range gotS {
		if gotS[g] != wantS[g] {
			t.Fatalf("scale %d is %g, the scalar's %g", g, gotS[g], wantS[g])
		}
	}
}
