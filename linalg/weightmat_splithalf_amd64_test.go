//go:build amd64

package linalg

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// TestWeightMatSplitHalf_matchesCanonical is the wiring's correctness gate: a WeightMat with the
// split-half layout attached must produce the same outputs as the same WeightMat without it.
//
// It compares the two DISPATCH paths, not the two kernels — that is the thing wiring can break.
// The kernels themselves are already gated against the scalar oracle
// (TestDotW4A8SplitHalfAVX2_matchesOracle); what this adds is that RepackInt4SplitHalf, the
// M=1 branch, the span's row/scale indexing and the parallel fan-out all agree with the
// canonical path on real matrix shapes.
func TestWeightMatSplitHalf_matchesCanonical(t *testing.T) {
	if !cpufeat.HasAVX2 {
		t.Skip("AVX2 required")
	}
	// On a VNNI host the repack declines by design (weightmat_splithalf_amd64.go):
	// canonical would use the AVX-512 VNNI tier and split-half only exists at AVX2,
	// so opting in would be both a downgrade and a numerics change. Assert the
	// decline rather than merely skipping past it — a repack that stopped declining
	// here is exactly the regression this gate should catch, and a bare Skip would
	// hide it. (This is how the VNNI problem was found: before the gate existed,
	// this test passed on Zen 2 and failed on a VNNI runner at rel 1.09e-4.)
	if cpufeat.HasAVX512VNNIVL {
		probe := wrapInt4F32(make([]byte, 32*16), make([]float32, 32), 32, 32, 32)
		if probe.RepackInt4SplitHalf() {
			t.Fatal("VNNI host: RepackInt4SplitHalf ACCEPTED, but canonical dispatches to the VNNI tier here — the layout would downgrade the kernel and diverge numerically")
		}
		t.Skip("VNNI host: repack correctly declines; the AVX2-vs-AVX2 equivalence this asserts is not defined here")
	}
	const group = 32
	rng := rand.New(rand.NewSource(3))
	// Shapes chosen to exercise both the serial and the parallel branch of the span split, and a
	// row count that is NOT a multiple of 4 (row4's constraint, deliberately not shared here).
	for _, sh := range []struct{ rows, cols int }{
		{1, 32}, {2, 64}, {7, 128}, {129, 512}, {512, 5120},
	} {
		w := make([]float32, sh.rows*sh.cols)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		packed, scales := QuantizeGroupsInt4(w, sh.rows, sh.cols, group)

		canon := wrapInt4F32(packed, scales, sh.rows, sh.cols, group)
		split := wrapInt4F32(packed, scales, sh.rows, sh.cols, group)
		if !split.RepackInt4SplitHalf() {
			t.Fatalf("rows=%d cols=%d: RepackInt4SplitHalf declined a qualifying tensor", sh.rows, sh.cols)
		}

		a := make([]float32, sh.cols)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		ws := &Workspace{}
		gotC := make([]float32, sh.rows)
		gotS := make([]float32, sh.rows)
		canon.MatmulBTW4A8Into(ws, a, gotC, 1)
		split.MatmulBTW4A8Into(ws, a, gotS, 1)

		for j := range gotC {
			den := math.Abs(float64(gotC[j]))
			if den < 1e-6 {
				den = 1e-6
			}
			if rel := math.Abs(float64(gotS[j]-gotC[j])) / den; rel > 1e-4 {
				t.Fatalf("rows=%d cols=%d row %d: split-half %v vs canonical %v (rel %.3g)",
					sh.rows, sh.cols, j, gotS[j], gotC[j], rel)
			}
		}
		t.Logf("rows=%-4d cols=%-5d OK — both dispatch paths agree across %d outputs", sh.rows, sh.cols, sh.rows)
	}
}

// TestWeightMatSplitHalf_canonicalUntouched pins the invariant the .giw kind=3 zero-copy load
// path depends on: repacking must not disturb the canonical bytes, which may be an mmap alias of
// a file on disk. Rewriting them in place would silently misdecode every existing bundle.
func TestWeightMatSplitHalf_canonicalUntouched(t *testing.T) {
	if !cpufeat.HasAVX2 {
		t.Skip("AVX2 required")
	}
	// On a VNNI host the repack declines by design (weightmat_splithalf_amd64.go):
	// canonical would use the AVX-512 VNNI tier and split-half only exists at AVX2,
	// so opting in would be both a downgrade and a numerics change. Assert the
	// decline rather than merely skipping past it — a repack that stopped declining
	// here is exactly the regression this gate should catch, and a bare Skip would
	// hide it. (This is how the VNNI problem was found: before the gate existed,
	// this test passed on Zen 2 and failed on a VNNI runner at rel 1.09e-4.)
	if cpufeat.HasAVX512VNNIVL {
		probe := wrapInt4F32(make([]byte, 32*16), make([]float32, 32), 32, 32, 32)
		if probe.RepackInt4SplitHalf() {
			t.Fatal("VNNI host: RepackInt4SplitHalf ACCEPTED, but canonical dispatches to the VNNI tier here — the layout would downgrade the kernel and diverge numerically")
		}
		t.Skip("VNNI host: repack correctly declines; the AVX2-vs-AVX2 equivalence this asserts is not defined here")
	}
	const rows, cols, group = 8, 256, 32
	rng := rand.New(rand.NewSource(5))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	packed, scales := QuantizeGroupsInt4(w, rows, cols, group)
	before := append([]byte(nil), packed...)

	m := wrapInt4F32(packed, scales, rows, cols, group)
	if !m.RepackInt4SplitHalf() {
		t.Fatal("RepackInt4SplitHalf declined a qualifying tensor")
	}
	for i := range packed {
		if packed[i] != before[i] {
			t.Fatalf("canonical packed bytes mutated at %d (%#x -> %#x) — a .giw kind=3 mmap alias would be corrupted", i, before[i], packed[i])
		}
	}
	if m.q4SplitHalf == nil {
		t.Fatal("split-half layout not populated")
	}
	if &m.q4SplitHalf[0] == &packed[0] {
		t.Fatal("split-half layout aliases the canonical bytes — it must be a separate allocation")
	}
}
