//go:build amd64

package linalg

import (
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// vnniDeclineCheck asserts (rather than merely skips past) that split-half
// declines on a VNNI host — same reasoning TestWeightMatSplitHalf_matches
// Canonical already documents: canonical would use the VNNI tier here, and
// split-half only exists at AVX2, so a repack that stopped declining would
// be a real regression, not a harmless one a bare Skip would hide. Returns
// true if the test should proceed (AVX2, no VNNI), having already Skipped
// otherwise (including plain !cpufeat.HasAVX2).
func vnniDeclineCheck(t *testing.T) bool {
	t.Helper()
	if !cpufeat.HasAVX2 {
		t.Skip("AVX2 required")
	}
	if cpufeat.HasAVX512VNNIVL {
		if int4SplitHalfUsable(32, 32) {
			t.Fatal("VNNI host: int4SplitHalfUsable(32,32) = true, but canonical dispatches to the VNNI tier here")
		}
		t.Skip("VNNI host: split-half correctly declines; the AVX2-vs-AVX2 equivalence this asserts is not defined here")
	}
	return true
}

// TestWeightMatSplitHalf_repackedOnlyMatchesCanonical is TestWeightMatSplit
// Half_matchesCanonical's repacked-only twin (audit M-22, extended by the
// M-22 split-half-M>1 follow-up): a split-half-only WeightMat, built by
// RepackInt4InPlace(LayoutSplitHalf), must produce the same output as canonical at
// EVERY M in the sweep — M=1 through the decode kernel, M>=4 through the
// new tile, 2<=M<4 through the tile's own per-row remainder path — with no
// panic, closing the M>1 gap matmulBTW4A8Into used to guard against.
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
			if !int4SplitHalfUsable(sh.cols, group) {
				t.Skip("split-half not usable for this shape")
			}
			for _, M := range []int{1, 2, 3, 4, 8, 64} {
				t.Run("M"+itoa(M), func(t *testing.T) {
					if M >= 4 {
						assertSplitHalfTileEngages(t, M, sh.cols)
					}
					q4, q4s := quantizeInt4Random(rng, sh.rows, sh.cols, group)
					canon := wrapInt4F32(cloneBytes(q4), cloneFloats(q4s), sh.rows, sh.cols, group)
					a := make([]float32, M*sh.cols)
					for i := range a {
						a[i] = float32(rng.NormFloat64())
					}

					var wsCanon Workspace
					want := make([]float32, M*sh.rows)
					canon.MatmulBTInto(&wsCanon, a, want, M)

					w, ok := repackInt4SplitHalfInPlaceF32(cloneBytes(q4), cloneFloats(q4s), sh.rows, sh.cols, group)
					if !ok {
						t.Fatal("RepackInt4InPlace(LayoutSplitHalf): ok=false")
					}
					var ws Workspace
					got := make([]float32, M*sh.rows)
					w.MatmulBTInto(&ws, a, got, M)

					for i := range got {
						if got[i] != want[i] {
							t.Fatalf("M=%d idx=%d (row %d, col %d): repacked-only %v, canonical %v",
								M, i, i/sh.rows, i%sh.rows, got[i], want[i])
						}
					}
				})
			}
		})
	}
}

// oldRepackSplitHalfReference reimplements repackW4A8SplitHalf's documented
// transform independently ("byte i holds weight i (low nibble) and weight
// i+16 (high), within each 32-wide group") rather than by calling
// repackInt4SplitHalfRow — audit M-22's refactor (9b8defd) factored
// repackW4A8SplitHalf's own row body OUT into repackInt4SplitHalfRow, the
// same function RepackInt4InPlace(LayoutSplitHalf) now calls, so repackW4A8SplitHalf
// alone is no longer an independent reference (unlike row4's
// repackSplitHalf4RowBlock, which stayed a separate, untouched function —
// split-half had no such surviving twin to fall back on, so this rebuilds
// the transform from the doc's own description instead).
func oldRepackSplitHalfReference(packed []byte, rows, cols, group int) []byte {
	bpr := cols / 2
	nib := func(row []byte, k int) byte {
		b := row[k/2]
		if k%2 == 0 {
			return b & 0x0F
		}
		return b >> 4
	}
	out := make([]byte, len(packed))
	for r := range rows {
		row := packed[r*bpr : (r+1)*bpr]
		dst := out[r*bpr : (r+1)*bpr]
		for g := 0; g < cols/group; g++ {
			gk, ob := g*group, g*(group/2)
			for i := 0; i < group/2; i++ {
				dst[ob+i] = nib(row, gk+i) | (nib(row, gk+i+group/2) << 4)
			}
		}
	}
	return out
}

// TestRepackInt4SplitHalfInPlace_matchesOutOfPlace is step 3's byte-for-byte
// gate: the in-place result must equal repackW4A8SplitHalf's out-of-place
// output AND an independent reference (oldRepackSplitHalfReference) built
// from the layout's own documented description rather than shared code —
// repackW4A8SplitHalf alone is not a sufficient reference since audit M-22
// refactored it to share repackInt4SplitHalfRow with
// RepackInt4InPlace(LayoutSplitHalf) itself.
func TestRepackInt4SplitHalfInPlace_matchesOutOfPlace(t *testing.T) {
	if !vnniDeclineCheck(t) {
		return
	}
	const group = 32
	if !int4SplitHalfUsable(1536, group) {
		t.Skip("split-half not usable on this core")
	}
	rng := rand.New(rand.NewPCG(22, 12))
	q4, q4s := quantizeInt4Random(rng, 8, 1536, group)

	want := repackW4A8SplitHalf(cloneBytes(q4), 8, 1536, group)
	oldWant := oldRepackSplitHalfReference(cloneBytes(q4), 8, 1536, group)

	w, ok := repackInt4SplitHalfInPlaceF32(q4, cloneFloats(q4s), 8, 1536, group)
	if !ok {
		t.Fatal("RepackInt4InPlace(LayoutSplitHalf): ok=false")
	}
	if len(w.q4SplitHalf) != len(want) {
		t.Fatalf("split-half length %d, want %d", len(w.q4SplitHalf), len(want))
	}
	for i := range want {
		if w.q4SplitHalf[i] != want[i] {
			t.Fatalf("split-half byte %d: in-place %d, out-of-place (RepackW4A8SplitHalf) %d", i, w.q4SplitHalf[i], want[i])
		}
		if w.q4SplitHalf[i] != oldWant[i] {
			t.Fatalf("split-half byte %d: in-place %d, independent reference %d", i, w.q4SplitHalf[i], oldWant[i])
		}
	}
	// q4s is retained UNCHANGED (shared scales, not repacked) — assert that
	// directly rather than only via a passing matmul comparison.
	for i, s := range q4s {
		if w.q4s16[i] != F32ToF16(s) {
			t.Fatalf("q4s[%d] = %v, want unchanged %v — split-half must not repack scales", i, F16ToF32(w.q4s16[i]), s)
		}
	}
}

// TestRepackInt4SplitHalfRow_streamedMatchesInPlace mirrors
// TestRepackInt4Row4Quad_streamedMatchesInPlace for the row-granularity
// (non-interleaved) split-half layout: assembling a tensor's split-half
// bytes row-by-row via the exported repackInt4SplitHalfRow primitive into a
// FRESH destination buffer must match RepackInt4InPlace(LayoutSplitHalf)'s result.
func TestRepackInt4SplitHalfRow_streamedMatchesInPlace(t *testing.T) {
	if !vnniDeclineCheck(t) {
		return
	}
	const group = 32
	if !int4SplitHalfUsable(1536, group) {
		t.Skip("split-half not usable on this core")
	}
	rng := rand.New(rand.NewPCG(22, 13))
	q4, q4s := quantizeInt4Random(rng, 8, 1536, group)
	_, bpr := groupsFor(1536, group)

	streamed := make([]byte, len(q4))
	for r := range 8 {
		repackInt4SplitHalfRow(streamed[r*bpr:(r+1)*bpr], q4[r*bpr:(r+1)*bpr], 1536)
	}

	w, ok := repackInt4SplitHalfInPlaceF32(cloneBytes(q4), cloneFloats(q4s), 8, 1536, group)
	if !ok {
		t.Fatal("RepackInt4InPlace(LayoutSplitHalf): ok=false")
	}
	for i := range streamed {
		if streamed[i] != w.q4SplitHalf[i] {
			t.Fatalf("streamed split-half byte %d: %d, in-place %d", i, streamed[i], w.q4SplitHalf[i])
		}
	}

	wStream, ok := wrapInt4SplitHalfOnlyF32(streamed, cloneFloats(q4s), 8, 1536, group)
	if !ok {
		t.Fatal("WrapInt4(LayoutSplitHalf): ok=false on a shape int4SplitHalfUsable already accepted")
	}
	if _, _, _, canonOK := wStream.int4Widened(); canonOK {
		t.Error("WrapInt4(LayoutSplitHalf) WeightMat reports Int4() ok=true")
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
	if !int4SplitHalfUsable(1536, group) {
		t.Skip("split-half not usable on this core")
	}
	rng := rand.New(rand.NewPCG(22, 14))

	allocsFor := func(rows int) float64 {
		q4, q4s := quantizeInt4Random(rng, rows, 1536, group)
		return testing.AllocsPerRun(1, func() {
			if _, ok := repackInt4SplitHalfInPlaceF32(q4, q4s, rows, 1536, group); !ok {
				t.Fatal("RepackInt4InPlace(LayoutSplitHalf): ok=false")
			}
		})
	}

	small := allocsFor(8)
	large := allocsFor(128)
	t.Logf("allocs: rows=8 -> %.0f, rows=128 -> %.0f", small, large)
	if small != large {
		t.Errorf("RepackInt4InPlace(LayoutSplitHalf)'s allocation count depends on rows: rows=8 -> %.0f allocs, rows=128 -> %.0f allocs, want equal",
			small, large)
	}
}
