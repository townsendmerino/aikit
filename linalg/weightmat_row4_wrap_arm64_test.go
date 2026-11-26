package linalg

import (
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// TestWrapInt4Row4_matchesRepackInt4Row4 proves the two ways of getting q4Row4/
// q4Row4Scales onto a WeightMat produce byte-identical results: computing them
// internally at load time (RepackInt4Row4, the arm64 in-RAM path) vs.
// handing back already-repacked bytes computed elsewhere (WrapInt4 with LayoutRow4,
// the goinfer .giw kind-4 loader's path, where RepackW4A8Row4/repackW4A8Row4Scales
// ran once at prequant time and the bytes were written to disk). Both derive
// from the same pure functions, so this is a construction-path equivalence
// check, not a numerics one — but it's the correctness anchor kind 4 depends on:
// if this ever diverges, the on-disk bytes stop meaning what the loader assumes.
func TestWrapInt4Row4_matchesRepackInt4Row4(t *testing.T) {
	const group = 32
	shapes := []struct{ K, N int }{
		{1536, 8960}, // real gate/up shape
		{256, 4},     // small, still a multiple of 4/group
	}
	rng := rand.New(rand.NewSource(97))
	for _, sh := range shapes {
		K, N := sh.K, sh.N
		w := make([]float32, N*K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		q4, q4s := QuantizeGroupsInt4(w, N, K, group)

		// Path A: internal derivation via RepackInt4Row4 (today's load-time path).
		wmA := wrapInt4F32(q4, q4s, N, K, group)
		if !wmA.RepackInt4Row4() {
			t.Skipf("K=%d N=%d: RepackInt4Row4 declined (no DotProd on this build/core) — nothing to compare", K, N)
		}
		gotRow4A, gotScalesA, ok := wmA.int4Row4Widened()
		if !ok {
			t.Fatalf("K=%d N=%d: RepackInt4Row4 reported success but Int4Row4 ok=false", K, N)
		}

		// Path B: repack the SAME canonical bytes directly (what a prequant tool
		// does once, ahead of time), then hand the result to WrapInt4 with LayoutRow4 (what a
		// .giw kind-4 loader does, mmap-aliasing bytes read off disk).
		row4B := RepackW4A8Row4(q4, N, K, group)
		scalesB := repackW4A8Row4Scales(q4s, N, K, group)
		wmB, ok := wrapInt4Row4OnlyF32(row4B, scalesB, N, K, group)
		if !ok {
			t.Fatalf("K=%d N=%d: WrapInt4(LayoutRow4) declined a shape RepackInt4Row4 accepted", K, N)
		}
		gotRow4B, gotScalesB, ok := wmB.int4Row4Widened()
		if !ok {
			t.Fatalf("K=%d N=%d: WrapInt4(LayoutRow4) did not populate Int4Row4", K, N)
		}

		if len(gotRow4A) != len(gotRow4B) {
			t.Fatalf("K=%d N=%d: q4Row4 length mismatch %d vs %d", K, N, len(gotRow4A), len(gotRow4B))
		}
		for i := range gotRow4A {
			if gotRow4A[i] != gotRow4B[i] {
				t.Fatalf("K=%d N=%d: q4Row4[%d] mismatch: internal=%02x external=%02x", K, N, i, gotRow4A[i], gotRow4B[i])
			}
		}
		if len(gotScalesA) != len(gotScalesB) {
			t.Fatalf("K=%d N=%d: q4Row4Scales length mismatch %d vs %d", K, N, len(gotScalesA), len(gotScalesB))
		}
		for i := range gotScalesA {
			if gotScalesA[i] != gotScalesB[i] {
				t.Fatalf("K=%d N=%d: q4Row4Scales[%d] mismatch: internal=%v external=%v", K, N, i, gotScalesA[i], gotScalesB[i])
			}
		}

		// And the dispatch itself must agree: a WeightMat built via WrapInt4(LayoutRow4)
		// must compute the identical M=1 result as one built via RepackInt4Row4.
		a := make([]float32, K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		var wsA, wsB Workspace
		gotA := make([]float32, N)
		gotB := make([]float32, N)
		wmA.MatmulBTW4A8Into(&wsA, a, gotA, 1)
		wmB.MatmulBTW4A8Into(&wsB, a, gotB, 1)
		for i := range gotA {
			if gotA[i] != gotB[i] {
				t.Fatalf("K=%d N=%d: dispatch result[%d] mismatch: RepackInt4Row4=%v WrapInt4=%v", K, N, i, gotA[i], gotB[i])
			}
		}
	}
}

// TestWrapInt4Row4_panicsOnBadLength proves the shape check actually fires —
// a bad q4Row4 length (as if a prequant bug wrote the wrong number of bytes)
// must panic rather than silently alias a mismatched span the kernel would
// read out of bounds.
func TestWrapInt4Row4_panicsOnBadLength(t *testing.T) {
	if !cpufeat.HasDotProd {
		t.Skip("DotProd required: without it WrapInt4(LayoutRow4) declines before checking lengths")
	}
	const group, K, N = 32, 64, 8
	rng := rand.New(rand.NewSource(13))
	w := make([]float32, N*K)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	q4, q4s := QuantizeGroupsInt4(w, N, K, group)
	badRow4 := make([]byte, len(q4)-1) // one byte short
	defer func() {
		if recover() == nil {
			t.Fatal("expected a panic on a short q4Row4")
		}
	}()
	WrapInt4(badRow4, RepackW4A8Row4ScalesF16(F32ToF16Scales(q4s), N, K, group), N, K, group, LayoutRow4)
}
