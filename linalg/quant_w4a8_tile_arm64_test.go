//go:build arm64

package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

// w4a8TileFixture builds a canonical-layout W4A8 problem: M activation rows,
// N weight rows of K group-32 int4 values.
func w4a8TileFixture(t *testing.T, M, K, N int) (aq []int8, aScales []float32, packed []byte, wScales []float32, nGroups, bpr int) {
	t.Helper()
	rng := rand.New(rand.NewPCG(17, 23))
	w := make([]float32, N*K)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	packed, wScales = QuantizeGroupsInt4(w, N, K, 32)
	nGroups, bpr = groupsFor(K, 32)
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	aq = make([]int8, M*K)
	aScales = make([]float32, M)
	QuantizeActivationsInto(aq, aScales, a, M, K)
	return
}

// TestW4A8Tile4RowSDOT_isLoadBearing is the first thing to check about a new
// kernel: that it RUNS. w4a8TileRows returns 0 on every decline path (no
// DotProd, M<4, group!=32, K<32), and a kernel that silently declines would
// leave every other test in this file passing against the old scalar span while
// proving nothing about the assembly.
func TestW4A8Tile4RowSDOT_isLoadBearing(t *testing.T) {
	if !hasDotProd {
		t.Skip("no FEAT_DotProd on this CPU — the tile declines by design")
	}
	const M, K, N = 8, 128, 16
	aq, aScales, packed, wScales, nGroups, bpr := w4a8TileFixture(t, M, K, N)
	dst := make([]float32, M*N)
	if got := w4a8TileRows(aq, aScales, packed, wScales, dst, M, K, N, 32, nGroups, bpr, 0, N); got != 8 {
		t.Fatalf("w4a8TileRows took %d rows, want 8 — the tile did not engage", got)
	}
	// And it must have written something: an all-zero dst would mean the
	// kernel ran but computed nothing.
	nonzero := false
	for _, v := range dst {
		if v != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		t.Fatal("tile claimed 8 rows but wrote all zeros")
	}
}

// TestW4A8Tile4RowSDOT_bitIdenticalToFoldSDOT is the contract that matters:
// the tile must reproduce dotW4A8FoldSDOT EXACTLY, not merely closely. If it
// drifts, the W4A8 result depends on M — which TestMatmulBTW4A8_MConsistent
// forbids and goinfer's speculative verify relies on.
//
// K values straddle the group boundary (128 and 160 are exact multiples of 32;
// 144 and 100 leave a ragged final group) so the scalar mop-up path is covered
// too.
func TestW4A8Tile4RowSDOT_bitIdenticalToFoldSDOT(t *testing.T) {
	if !hasDotProd {
		t.Skip("no FEAT_DotProd on this CPU")
	}
	for _, K := range []int{32, 64, 100, 128, 144, 160} {
		const M, N = 4, 5
		aq, aScales, packed, wScales, nGroups, bpr := w4a8TileFixture(t, M, K, N)
		got := make([]float32, M*N)
		if n := w4a8TileRows(aq, aScales, packed, wScales, got, M, K, N, 32, nGroups, bpr, 0, N); n != 4 {
			t.Fatalf("K=%d: tile took %d rows, want 4", K, n)
		}
		// Reference: the canonical per-(row, column) kernel, one call each.
		for i := range M {
			for j := range N {
				want := dotW4A8(aq[i*K:(i+1)*K], packed[j*bpr:(j+1)*bpr], wScales[j*nGroups:(j+1)*nGroups], 32, K) * aScales[i]
				if g := got[i*N+j]; math.Float32bits(g) != math.Float32bits(want) {
					t.Fatalf("K=%d i=%d j=%d: tile %v (%08x) != dotW4A8 %v (%08x)",
						K, i, j, g, math.Float32bits(g), want, math.Float32bits(want))
				}
			}
		}
	}
}

// TestW4A8Tile4RowSDOT_zeroActivationRow pins the aScale==0 shortcut the span
// has always had: a row that quantized to all zeros stores a literal 0.
func TestW4A8Tile4RowSDOT_zeroActivationRow(t *testing.T) {
	if !hasDotProd {
		t.Skip("no FEAT_DotProd on this CPU")
	}
	const M, K, N = 4, 64, 3
	aq, aScales, packed, wScales, nGroups, bpr := w4a8TileFixture(t, M, K, N)
	aScales[2] = 0
	dst := make([]float32, M*N)
	w4a8TileRows(aq, aScales, packed, wScales, dst, M, K, N, 32, nGroups, bpr, 0, N)
	for j := range N {
		if v := dst[2*N+j]; v != 0 {
			t.Errorf("zero-scale row col %d = %v, want 0", j, v)
		}
	}
}
