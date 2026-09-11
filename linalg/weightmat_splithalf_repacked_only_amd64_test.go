//go:build amd64

package linalg

import (
	"math/rand/v2"
	"testing"
)

// vnniDeclineCheck asserts (rather than merely skips past) that split-half
// declines on a VNNI host — same reasoning TestWeightMatSplitHalf_matches
// Canonical already documents: canonical would use the VNNI tier here, and
// split-half only exists at AVX2, so a repack that stopped declining would
// be a real regression, not a harmless one a bare Skip would hide. Returns
// true if the test should proceed (AVX2, no VNNI), having already Skipped
// otherwise (including plain !hasAVX2).
func vnniDeclineCheck(t *testing.T) bool {
	t.Helper()
	if !hasAVX2 {
		t.Skip("AVX2 required")
	}
	if hasAVX512VNNIVL {
		if Int4SplitHalfUsable(32, 32) {
			t.Fatal("VNNI host: Int4SplitHalfUsable(32,32) = true, but canonical dispatches to the VNNI tier here")
		}
		t.Skip("VNNI host: split-half correctly declines; the AVX2-vs-AVX2 equivalence this asserts is not defined here")
	}
	return true
}

// TestWeightMatSplitHalf_repackedOnlyMatchesCanonical is TestWeightMatSplit
// Half_matchesCanonical's repacked-only twin (audit M-22): a split-half-only
// WeightMat, built by RepackInt4SplitHalfInPlace, must produce the same M=1
// output as canonical — split-half is M=1 only (MatmulBTW4A8Into's own
// guard), so unlike row4 this has no M sweep to run.
func TestWeightMatSplitHalf_repackedOnlyMatchesCanonical(t *testing.T) {
	if !vnniDeclineCheck(t) {
		return
	}
	const group = 32
	rng := rand.New(rand.NewPCG(22, 11))
	for _, sh := range []struct{ rows, cols int }{
		{4, 1536}, {16, 2048}, {32, 2560}, {64, 4096}, {128, 8960},
	} {
		t.Run("rows"+itoa(sh.rows)+"_cols"+itoa(sh.cols), func(t *testing.T) {
			if !Int4SplitHalfUsable(sh.cols, group) {
				t.Skip("split-half not usable for this shape")
			}
			q4, q4s := quantizeInt4Random(rng, sh.rows, sh.cols, group)
			canon := WrapInt4(cloneBytes(q4), cloneFloats(q4s), sh.rows, sh.cols, group)
			a := make([]float32, sh.cols)
			for i := range a {
				a[i] = float32(rng.NormFloat64())
			}

			var wsCanon Workspace
			want := make([]float32, sh.rows)
			canon.MatmulBTInto(&wsCanon, a, want, 1)

			w, ok := RepackInt4SplitHalfInPlace(cloneBytes(q4), cloneFloats(q4s), sh.rows, sh.cols, group)
			if !ok {
				t.Fatal("RepackInt4SplitHalfInPlace: ok=false")
			}
			var ws Workspace
			got := make([]float32, sh.rows)
			w.MatmulBTInto(&ws, a, got, 1)

			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("row %d: repacked-only %v, canonical %v", i, got[i], want[i])
				}
			}
		})
	}
}

// TestRepackInt4SplitHalfInPlace_matchesOutOfPlace is step 3's byte-for-byte
// gate: the in-place result must equal RepackW4A8SplitHalf's out-of-place
// output for the same canonical input.
func TestRepackInt4SplitHalfInPlace_matchesOutOfPlace(t *testing.T) {
	if !vnniDeclineCheck(t) {
		return
	}
	const group = 32
	if !Int4SplitHalfUsable(1536, group) {
		t.Skip("split-half not usable on this core")
	}
	rng := rand.New(rand.NewPCG(22, 12))
	q4, q4s := quantizeInt4Random(rng, 8, 1536, group)

	want := RepackW4A8SplitHalf(cloneBytes(q4), 8, 1536, group)

	w, ok := RepackInt4SplitHalfInPlace(q4, cloneFloats(q4s), 8, 1536, group)
	if !ok {
		t.Fatal("RepackInt4SplitHalfInPlace: ok=false")
	}
	if len(w.q4SplitHalf) != len(want) {
		t.Fatalf("split-half length %d, want %d", len(w.q4SplitHalf), len(want))
	}
	for i := range want {
		if w.q4SplitHalf[i] != want[i] {
			t.Fatalf("split-half byte %d: in-place %d, out-of-place %d", i, w.q4SplitHalf[i], want[i])
		}
	}
	// q4s is retained UNCHANGED (shared scales, not repacked) — assert that
	// directly rather than only via a passing matmul comparison.
	for i, s := range q4s {
		if w.q4s[i] != s {
			t.Fatalf("q4s[%d] = %v, want unchanged %v — split-half must not repack scales", i, w.q4s[i], s)
		}
	}
}

// TestRepackInt4SplitHalfRow_streamedMatchesInPlace mirrors
// TestRepackInt4Row4Quad_streamedMatchesInPlace for the row-granularity
// (non-interleaved) split-half layout: assembling a tensor's split-half
// bytes row-by-row via the exported RepackInt4SplitHalfRow primitive into a
// FRESH destination buffer must match RepackInt4SplitHalfInPlace's result.
func TestRepackInt4SplitHalfRow_streamedMatchesInPlace(t *testing.T) {
	if !vnniDeclineCheck(t) {
		return
	}
	const group = 32
	if !Int4SplitHalfUsable(1536, group) {
		t.Skip("split-half not usable on this core")
	}
	rng := rand.New(rand.NewPCG(22, 13))
	q4, q4s := quantizeInt4Random(rng, 8, 1536, group)
	_, bpr := groupsFor(1536, group)

	streamed := make([]byte, len(q4))
	for r := 0; r < 8; r++ {
		RepackInt4SplitHalfRow(streamed[r*bpr:(r+1)*bpr], q4[r*bpr:(r+1)*bpr], 1536)
	}

	w, ok := RepackInt4SplitHalfInPlace(cloneBytes(q4), cloneFloats(q4s), 8, 1536, group)
	if !ok {
		t.Fatal("RepackInt4SplitHalfInPlace: ok=false")
	}
	for i := range streamed {
		if streamed[i] != w.q4SplitHalf[i] {
			t.Fatalf("streamed split-half byte %d: %d, in-place %d", i, streamed[i], w.q4SplitHalf[i])
		}
	}

	wStream, ok := WrapInt4SplitHalfOnly(streamed, cloneFloats(q4s), 8, 1536, group)
	if !ok {
		t.Fatal("WrapInt4SplitHalfOnly: ok=false on a shape Int4SplitHalfUsable already accepted")
	}
	if _, _, _, canonOK := wStream.Int4(); canonOK {
		t.Error("WrapInt4SplitHalfOnly WeightMat reports Int4() ok=true")
	}
}

// TestRepackInt4SplitHalfInPlace_allocsIndependentOfRows is step 3's
// allocation test, the split-half twin of
// TestRepackInt4Row4InPlace_allocsIndependentOfRows: peak extra memory is
// the one-row scratch (split-half does not interleave rows, so its scratch
// is row-sized rather than quad-sized), not O(rows).
func TestRepackInt4SplitHalfInPlace_allocsIndependentOfRows(t *testing.T) {
	if !vnniDeclineCheck(t) {
		return
	}
	const group = 32
	if !Int4SplitHalfUsable(1536, group) {
		t.Skip("split-half not usable on this core")
	}
	rng := rand.New(rand.NewPCG(22, 14))

	allocsFor := func(rows int) float64 {
		q4, q4s := quantizeInt4Random(rng, rows, 1536, group)
		return testing.AllocsPerRun(1, func() {
			if _, ok := RepackInt4SplitHalfInPlace(q4, q4s, rows, 1536, group); !ok {
				t.Fatal("RepackInt4SplitHalfInPlace: ok=false")
			}
		})
	}

	small := allocsFor(8)
	large := allocsFor(128)
	t.Logf("allocs: rows=8 -> %.0f, rows=128 -> %.0f", small, large)
	if small != large {
		t.Errorf("RepackInt4SplitHalfInPlace's allocation count depends on rows: rows=8 -> %.0f allocs, rows=128 -> %.0f allocs, want equal",
			small, large)
	}
}
