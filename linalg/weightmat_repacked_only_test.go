package linalg

import (
	"math/rand/v2"
	"testing"
)

// Audit M-22: repacked-only int4 WeightMat, representation fixed at
// construction. These tests are all NEW — none replace or modify an
// existing test, matching the ADDITIVE brief.

// weightmatInt4Shapes are the production shapes step 3 of the brief asks
// for: group 32, cols spanning real projection widths, rows a multiple of 4
// including the rows==4 edge (the smallest a row4-eligible tensor can be).
var weightmatInt4Shapes = []struct{ rows, cols int }{
	{4, 1536},
	{16, 2048},
	{32, 2560},
	{64, 4096},
	{128, 8960},
}

func quantizeInt4Random(rng *rand.Rand, rows, cols, group int) (packed []byte, scales []float32) {
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	return QuantizeGroupsInt4(w, rows, cols, group)
}

func cloneBytes(b []byte) []byte {
	c := make([]byte, len(b))
	copy(c, b)
	return c
}

func cloneFloats(f []float32) []float32 {
	c := make([]float32, len(f))
	copy(c, f)
	return c
}

// TestWeightMatInt4_truthTable is the Kind()/IsInt4()/Int4()/Int4Layout()
// cross-check across all three representation policies, step 3 of the M-22
// brief. Int4() means "canonical bytes present"; the other three mean
// "is/which int4" — this is the distinction the whole task exists to make,
// asserted directly rather than left implicit in the matmul/Row tests.
//
// RepackInt4Row4InPlace/RepackInt4SplitHalfInPlace MUTATE their q4/q4s
// inputs, so each sub-test that calls one gets its OWN fresh
// quantizeInt4Random output — never the shared q4/q4s the canonical-only/
// "both" sub-tests read.
func TestWeightMatInt4_truthTable(t *testing.T) {
	const group = 32
	rng := rand.New(rand.NewPCG(22, 1))
	q4, q4s := quantizeInt4Random(rng, 8, 1536, group)

	check := func(t *testing.T, w *WeightMat, wantKind string, wantIsInt4 bool, wantInt4OK bool, wantLayout string) {
		t.Helper()
		if k := w.Kind(); k != wantKind {
			t.Errorf("Kind() = %q, want %q", k, wantKind)
		}
		if got := w.IsInt4(); got != wantIsInt4 {
			t.Errorf("IsInt4() = %v, want %v", got, wantIsInt4)
		}
		if _, _, _, ok := w.Int4(); ok != wantInt4OK {
			t.Errorf("Int4() ok = %v, want %v", ok, wantInt4OK)
		}
		if l := w.Int4Layout(); l != wantLayout {
			t.Errorf("Int4Layout() = %q, want %q", l, wantLayout)
		}
	}

	t.Run("empty", func(t *testing.T) {
		var w WeightMat
		check(t, &w, "", false, false, "")
	})
	t.Run("f32", func(t *testing.T) {
		w := WrapF32(make([]float32, 8*1536), 8, 1536)
		check(t, &w, "f32", false, false, "")
	})
	t.Run("int8", func(t *testing.T) {
		w := WrapInt8(make([]int8, 8*1536), make([]float32, 8), 8, 1536, false)
		check(t, &w, "int8", false, false, "")
	})
	t.Run("canonical-only", func(t *testing.T) {
		w := WrapInt4(cloneBytes(q4), cloneFloats(q4s), 8, 1536, group)
		check(t, &w, "int4", true, true, "canonical")
	})
	t.Run("both", func(t *testing.T) {
		w := WrapInt4(cloneBytes(q4), cloneFloats(q4s), 8, 1536, group)
		haveRow4 := w.RepackInt4Row4()
		haveSplitHalf := false
		if !haveRow4 {
			haveSplitHalf = w.RepackInt4SplitHalf()
		}
		switch {
		case haveRow4:
			check(t, &w, "int4", true, true, "canonical+row4")
		case haveSplitHalf:
			check(t, &w, "int4", true, true, "canonical+splithalf")
		default:
			// Neither repack layout usable on this core (e.g. no DotProd, no
			// AVX2, or a VNNI host declining split-half by design) — the
			// WeightMat is still exactly canonical-only.
			check(t, &w, "int4", true, true, "canonical")
		}
	})
	t.Run("repacked-only-row4", func(t *testing.T) {
		if !Int4Row4Usable(8, 1536, group) {
			t.Skip("row4 not usable on this core")
		}
		w, ok := RepackInt4Row4InPlace(cloneBytes(q4), cloneFloats(q4s), 8, 1536, group)
		if !ok {
			t.Fatal("RepackInt4Row4InPlace: ok=false, Int4Row4Usable said yes")
		}
		check(t, &w, "int4", true, false, "row4")
	})
	t.Run("repacked-only-splithalf", func(t *testing.T) {
		if !Int4SplitHalfUsable(1536, group) {
			t.Skip("split-half not usable on this core")
		}
		w, ok := RepackInt4SplitHalfInPlace(cloneBytes(q4), cloneFloats(q4s), 8, 1536, group)
		if !ok {
			t.Fatal("RepackInt4SplitHalfInPlace: ok=false, Int4SplitHalfUsable said yes")
		}
		check(t, &w, "int4", true, false, "splithalf")
	})
}

// TestWeightMatInt4_repackedOnlyHoldsNoCanonical is the memory-shape
// assertion M-22 is FOR: a repacked-only WeightMat retains no canonical
// bytes at all, not "retains them until something drops them."
func TestWeightMatInt4_repackedOnlyHoldsNoCanonical(t *testing.T) {
	const group = 32
	rng := rand.New(rand.NewPCG(22, 2))
	q4, q4s := quantizeInt4Random(rng, 8, 1536, group)

	if Int4Row4Usable(8, 1536, group) {
		w, ok := RepackInt4Row4InPlace(cloneBytes(q4), cloneFloats(q4s), 8, 1536, group)
		if !ok {
			t.Fatal("RepackInt4Row4InPlace: ok=false")
		}
		if q4got, q4sgot, _, ok := w.Int4(); ok || len(q4got) != 0 || len(q4sgot) != 0 {
			t.Errorf("row4-only: Int4() = (%d bytes, %d scales, ok=%v), want (0, 0, false)",
				len(q4got), len(q4sgot), ok)
		}
	}
	if Int4SplitHalfUsable(1536, group) {
		w, ok := RepackInt4SplitHalfInPlace(cloneBytes(q4), cloneFloats(q4s), 8, 1536, group)
		if !ok {
			t.Fatal("RepackInt4SplitHalfInPlace: ok=false")
		}
		// split-half-only DOES retain q4s (shared scales, see the q4SplitHalf
		// field comment) but never the packed canonical NIBBLES — Int4()'s ok
		// is false either way, since ok means "q4 (nibbles) present".
		if q4got, _, _, ok := w.Int4(); ok || len(q4got) != 0 {
			t.Errorf("splithalf-only: Int4() q4 = %d bytes ok=%v, want (0, false)", len(q4got), ok)
		}
	}
}

// TestWeightMatInt4_repackedOnlyGateFailureLeavesInputUntouched: when
// Int4Row4Usable/Int4SplitHalfUsable is false, RepackInt4Row4InPlace/
// RepackInt4SplitHalfInPlace must return before writing anything — checked
// directly by comparing before/after bytes, not inferred from ok=false
// alone (ok=false with a partially-mutated input would be a worse bug than
// ok=false with an untouched one).
func TestWeightMatInt4_repackedOnlyGateFailureLeavesInputUntouched(t *testing.T) {
	const group = 32
	rng := rand.New(rand.NewPCG(22, 7))
	// rows=6 fails Int4Row4Usable's rows%4==0 check unconditionally,
	// regardless of CPU features — a gate failure guaranteed on every arch.
	q4, q4s := quantizeInt4Random(rng, 6, 1536, group)
	before4, beforeS := cloneBytes(q4), cloneFloats(q4s)

	if Int4Row4Usable(6, 1536, group) {
		t.Fatal("test premise broken: rows=6 should never satisfy Int4Row4Usable's rows%4==0")
	}
	w, ok := RepackInt4Row4InPlace(q4, q4s, 6, 1536, group)
	if ok {
		t.Fatal("RepackInt4Row4InPlace: ok=true for rows=6, want false")
	}
	if w.IsInt4() {
		t.Error("RepackInt4Row4InPlace: ok=false but returned a non-zero WeightMat")
	}
	for i := range q4 {
		if q4[i] != before4[i] {
			t.Fatalf("RepackInt4Row4InPlace wrote to q4[%d] despite ok=false", i)
		}
	}
	for i := range q4s {
		if q4s[i] != beforeS[i] {
			t.Fatalf("RepackInt4Row4InPlace wrote to q4s[%d] despite ok=false", i)
		}
	}
}

// TestWeightMatInt4_existingPoliciesUnchanged asserts canonical-only and
// "both" WeightMats behave EXACTLY as before M-22 — Row/Kind/Int4/
// MatmulBTInto outputs, for the same input, unchanged. This is the
// brief's ADDITIVE contract checked directly rather than assumed from
// "the new cases are additive-looking": it drives the SAME entry points a
// pre-M-22 caller would have, and would fail if the new cases had shadowed
// or reordered the existing ones. Neither sub-test here calls an *InPlace
// constructor, so q4/q4s are never mutated and the shared copy is safe to
// reuse.
func TestWeightMatInt4_existingPoliciesUnchanged(t *testing.T) {
	const group = 32
	rng := rand.New(rand.NewPCG(22, 3))
	q4, q4s := quantizeInt4Random(rng, 8, 1536, group)
	a := make([]float32, 1536)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}

	build := func() WeightMat {
		w := WrapInt4(q4, q4s, 8, 1536, group)
		w.RepackInt4Row4()
		w.RepackInt4SplitHalf()
		return w
	}

	for _, name := range []string{"canonical-only", "both-if-usable"} {
		t.Run(name, func(t *testing.T) {
			var w WeightMat
			if name == "canonical-only" {
				w = WrapInt4(q4, q4s, 8, 1536, group)
			} else {
				w = build()
			}
			if k := w.Kind(); k != "int4" {
				t.Errorf("Kind() = %q, want int4", k)
			}
			if _, _, _, ok := w.Int4(); !ok {
				t.Error("Int4() ok = false, want true — canonical must still be present")
			}
			dst := make([]float32, 1536)
			w.Row(3, dst)
			want := make([]float32, 1536)
			DequantizeRowInt4(q4[3*768:4*768], q4s[3*48:4*48], group, 1536, want)
			for i := range dst {
				if dst[i] != want[i] {
					t.Fatalf("Row(3)[%d] = %v, want %v (bit-identical to direct DequantizeRowInt4)", i, dst[i], want[i])
					break
				}
			}
			var ws Workspace
			got := make([]float32, 8)
			w.MatmulBTInto(&ws, a, got, 1)
			wantOut := make([]float32, 8)
			MatmulBTW4A8Into(&ws, a, q4, q4s, wantOut, 1, 1536, 8, group)
			for i := range got {
				if got[i] != wantOut[i] {
					t.Errorf("MatmulBTInto[%d] = %v, want %v (bit-identical to the free function)", i, got[i], wantOut[i])
				}
			}
		})
	}
}

// TestWeightMatRow_repackedMatchesCanonical is step 3's Row() bit-identity
// gate: row4 and split-half gathers against DequantizeRowInt4's own
// canonical reconstruction, across the production shapes and every row of a
// small tensor. The canonical reference (canon) is built from its OWN
// clone, read before either mutating constructor runs, so it never observes
// the repacked-only WeightMats' in-place writes.
func TestWeightMatRow_repackedMatchesCanonical(t *testing.T) {
	rng := rand.New(rand.NewPCG(22, 4))
	for _, sh := range weightmatInt4Shapes {
		t.Run("rows"+itoa(sh.rows)+"_cols"+itoa(sh.cols), func(t *testing.T) {
			const group = 32
			q4, q4s := quantizeInt4Random(rng, sh.rows, sh.cols, group)
			canon := WrapInt4(cloneBytes(q4), cloneFloats(q4s), sh.rows, sh.cols, group)

			checkRow := func(t *testing.T, w *WeightMat, i int) {
				t.Helper()
				got := make([]float32, sh.cols)
				w.Row(i, got)
				want := make([]float32, sh.cols)
				canon.Row(i, want)
				for k := range got {
					if got[k] != want[k] {
						t.Fatalf("row %d, col %d: got %v, want %v (canonical)", i, k, got[k], want[k])
					}
				}
			}

			if Int4Row4Usable(sh.rows, sh.cols, group) {
				t.Run("row4", func(t *testing.T) {
					w, ok := RepackInt4Row4InPlace(cloneBytes(q4), cloneFloats(q4s), sh.rows, sh.cols, group)
					if !ok {
						t.Fatal("RepackInt4Row4InPlace: ok=false")
					}
					for i := range sh.rows {
						checkRow(t, &w, i)
					}
				})
			}
			if Int4SplitHalfUsable(sh.cols, group) {
				t.Run("splithalf", func(t *testing.T) {
					w, ok := RepackInt4SplitHalfInPlace(cloneBytes(q4), cloneFloats(q4s), sh.rows, sh.cols, group)
					if !ok {
						t.Fatal("RepackInt4SplitHalfInPlace: ok=false")
					}
					for i := range sh.rows {
						checkRow(t, &w, i)
					}
				})
			}
		})
	}
}

// TestRepackInt4Row4InPlace_matchesOutOfPlace and
// TestRepackInt4Row4Quad_streamedMatchesInPlace live in
// weightmat_row4_repacked_only_arm64_test.go (arm64-tagged): both call
// RepackW4A8Row4/RepackW4A8Row4Scales/RepackInt4Row4Quad/
// RepackInt4Row4ScalesQuad, which — unlike split-half — have no amd64 or
// portable twin at all (row4 is NEON-only, matmul_w4a8_row4_arm64.go /
// repack_w4a8_row4_arm64.go).

// TestRepackInt4Row4InPlace_allocsIndependentOfRows is the allocation test
// step 3 asks for: peak extra memory is the one-quad scratch, not O(rows).
func TestRepackInt4Row4InPlace_allocsIndependentOfRows(t *testing.T) {
	const group = 32
	if !Int4Row4Usable(128, 1536, group) {
		t.Skip("row4 not usable on this core")
	}
	rng := rand.New(rand.NewPCG(22, 10))

	// runs=1: RepackInt4Row4InPlace mutates its input, so it must only be
	// called ONCE per (q4, q4s) pair — testing.AllocsPerRun(1, ...) does
	// exactly that, measuring a single call in isolation.
	allocsFor := func(rows int) float64 {
		q4, q4s := quantizeInt4Random(rng, rows, 1536, group)
		return testing.AllocsPerRun(1, func() {
			if _, ok := RepackInt4Row4InPlace(q4, q4s, rows, 1536, group); !ok {
				t.Fatal("RepackInt4Row4InPlace: ok=false")
			}
		})
	}

	small := allocsFor(8)
	large := allocsFor(128)
	// Exactly 2 allocations either way (the nibble scratch, the scale
	// scratch), regardless of rows.
	t.Logf("allocs: rows=8 -> %.0f, rows=128 -> %.0f", small, large)
	if small != large {
		t.Errorf("RepackInt4Row4InPlace's allocation count depends on rows: rows=8 -> %.0f allocs, rows=128 -> %.0f allocs, want equal",
			small, large)
	}
}
