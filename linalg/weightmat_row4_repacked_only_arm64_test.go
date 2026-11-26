//go:build arm64

package linalg

import (
	"math/rand/v2"
	"testing"
)

// TestWeightMatW4A8_MConsistentRepackedOnly is
// TestWeightMatW4A8_MConsistentAcrossRow4Dispatch (matmul_quant_mconsistent_
// test.go) run a second time against a repacked-only WeightMat built by
// RepackInt4InPlace(LayoutRow4) — audit M-22, step 3 of the brief. Same cases, same M
// sweep, same decode==prefill==verify guarantee, this time with no
// canonical bytes behind it at all: a repacked-only tensor must be exactly
// as M-consistent as a "both" one, since matmulBTW4A8Into's row4/tile
// dispatch is the SAME code path for both — the only difference is whether
// w.q4 is nil.
func TestWeightMatW4A8_MConsistentRepackedOnly(t *testing.T) {
	const group = 32
	rng := rand.New(rand.NewPCG(0xd1ce, 0x2202)) // audit M-22
	maxM := mconsistentMs[len(mconsistentMs)-1]

	for _, c := range quantMConsistentCases {
		t.Run(c.name, func(t *testing.T) {
			if !int4Row4Usable(c.N, c.K, group) {
				t.Skipf("row4 not usable for N=%d K=%d group=%d on this core", c.N, c.K, group)
			}
			a := mconsistentRand(rng, maxM*c.K)
			// q4/q4s stay UNTOUCHED — RepackInt4InPlace(LayoutRow4) below mutates its
			// input, and the "both" comparison further down needs the
			// original canonical bytes, so it gets its own clone too.
			q4, q4s := quantizeInt4Random(rng, c.N, c.K, group)
			wm, ok := repackInt4Row4InPlaceF32(cloneBytes(q4), cloneFloats(q4s), c.N, c.K, group)
			if !ok {
				t.Fatalf("RepackInt4InPlace(LayoutRow4): ok=false, int4Row4Usable said yes for N=%d K=%d", c.N, c.K)
			}
			if _, _, _, canonOK := wm.int4Widened(); canonOK {
				t.Fatal("repacked-only WeightMat unexpectedly reports Int4() ok=true")
			}

			solo := make([]float32, maxM*c.N)
			var wsSolo Workspace
			for i := range maxM {
				wm.MatmulBTW4A8Into(&wsSolo, a[i*c.K:(i+1)*c.K], solo[i*c.N:(i+1)*c.N], 1)
			}

			for _, M := range mconsistentMs {
				var ws Workspace
				got := make([]float32, M*c.N)
				wm.MatmulBTW4A8Into(&ws, a, got, M)
				for i := range M {
					for j := range c.N {
						if g, w := got[i*c.N+j], solo[i*c.N+j]; g != w {
							t.Fatalf("repacked-only: out[%d,%d] at M=%d is %v, alone at M=1 is %v (diff %v)",
								i, j, M, g, w, g-w)
						}
					}
				}
			}

			// Cross-check against a "both" WeightMat's output too, not just
			// internal self-consistency — repacked-only must be bit-identical
			// to canonical-driven-through-row4, not merely consistent with
			// itself.
			both := wrapInt4F32(q4, q4s, c.N, c.K, group)
			if !both.RepackInt4Row4() {
				t.Fatal("both: RepackInt4Row4 declined after int4Row4Usable said yes")
			}
			var wsBoth Workspace
			wantOut := make([]float32, maxM*c.N)
			both.MatmulBTW4A8Into(&wsBoth, a, wantOut, maxM)
			var wsRepacked Workspace
			gotOut := make([]float32, maxM*c.N)
			wm.MatmulBTW4A8Into(&wsRepacked, a, gotOut, maxM)
			for i := range gotOut {
				if gotOut[i] != wantOut[i] {
					t.Fatalf("repacked-only vs both at M=%d: out[%d] = %v, want %v", maxM, i, gotOut[i], wantOut[i])
				}
			}
		})
	}
}

// oldRepackRow4Reference builds the row4 nibble bytes and scales for an
// [N,K] canonical tensor using repackSplitHalf4RowBlock/interleaveScales4Row
// directly — the per-quad functions RepackW4A8Row4/repackW4A8Row4Scales
// called BEFORE the audit M-22 refactor (9b8defd) that moved them onto the
// shared repackInt4Row4Quad/repackInt4Row4ScalesQuad primitive. Those two
// functions are untouched by that refactor (several other tests still
// exercise them directly — see their own doc comments), so this is an
// independent reimplementation of "what row4 packing means", not the same
// code path RepackInt4InPlace(LayoutRow4)/RepackW4A8Row4 now share. Comparing
// against THIS, not just against RepackW4A8Row4's own output, is what makes
// TestRepackInt4Row4InPlace_matchesOutOfPlace actually test something: since
// the refactor, RepackW4A8Row4 calls repackInt4Row4Quad too, so a
// comparison against only RepackW4A8Row4 would be the new code checked
// against itself.
func oldRepackRow4Reference(packed []byte, scales []float32, N, K, group int) (row4 []byte, row4Scales []float32) {
	nGroups, bpr := groupsFor(K, group)
	row4 = make([]byte, N*bpr)
	row4Scales = make([]float32, N*nGroups)
	for q := 0; q < N/4; q++ {
		rowBase := q * 4 * bpr
		row0 := packed[rowBase : rowBase+bpr]
		row1 := packed[rowBase+bpr : rowBase+2*bpr]
		row2 := packed[rowBase+2*bpr : rowBase+3*bpr]
		row3 := packed[rowBase+3*bpr : rowBase+4*bpr]
		copy(row4[rowBase:rowBase+4*bpr], repackSplitHalf4RowBlock(row0, row1, row2, row3, K))

		scaleBase := q * 4 * nGroups
		s0 := scales[scaleBase : scaleBase+nGroups]
		s1 := scales[scaleBase+nGroups : scaleBase+2*nGroups]
		s2 := scales[scaleBase+2*nGroups : scaleBase+3*nGroups]
		s3 := scales[scaleBase+3*nGroups : scaleBase+4*nGroups]
		copy(row4Scales[scaleBase:scaleBase+4*nGroups], interleaveScales4Row(s0, s1, s2, s3, nGroups))
	}
	return row4, row4Scales
}

// TestRepackInt4Row4InPlace_matchesOutOfPlace is step 3's byte-for-byte
// gate: the in-place result must equal RepackW4A8Row4/repackW4A8Row4Scales's
// out-of-place output for the SAME canonical input, AND an independent
// reference built from the untouched pre-M-22 per-quad primitives
// (oldRepackRow4Reference) — RepackW4A8Row4 alone is not a sufficient
// reference since audit M-22 (9b8defd) refactored it to share
// repackInt4Row4Quad with RepackInt4InPlace(LayoutRow4) itself.
func TestRepackInt4Row4InPlace_matchesOutOfPlace(t *testing.T) {
	const group = 32
	if !int4Row4Usable(8, 1536, group) {
		t.Skip("row4 not usable on this core")
	}
	rng := rand.New(rand.NewPCG(22, 8))
	q4, q4s := quantizeInt4Random(rng, 8, 1536, group)

	wantRow4 := RepackW4A8Row4(cloneBytes(q4), 8, 1536, group)
	wantScales := repackW4A8Row4Scales(cloneFloats(q4s), 8, 1536, group)
	oldRow4, oldScales := oldRepackRow4Reference(cloneBytes(q4), cloneFloats(q4s), 8, 1536, group)

	w, ok := repackInt4Row4InPlaceF32(q4, q4s, 8, 1536, group)
	if !ok {
		t.Fatal("RepackInt4InPlace(LayoutRow4): ok=false")
	}
	gotRow4, gotScales, ok := w.int4Row4Widened()
	if !ok {
		t.Fatal("Int4Row4(): ok=false")
	}
	for i := range gotRow4 {
		if gotRow4[i] != wantRow4[i] {
			t.Fatalf("row4 byte %d: in-place %d, out-of-place (RepackW4A8Row4) %d", i, gotRow4[i], wantRow4[i])
		}
		if gotRow4[i] != oldRow4[i] {
			t.Fatalf("row4 byte %d: in-place %d, pre-M-22 reference (repackSplitHalf4RowBlock) %d", i, gotRow4[i], oldRow4[i])
		}
	}
	for i := range gotScales {
		if gotScales[i] != wantScales[i] {
			t.Fatalf("row4 scale %d: in-place %v, out-of-place (RepackW4A8Row4Scales) %v", i, gotScales[i], wantScales[i])
		}
		if gotScales[i] != oldScales[i] {
			t.Fatalf("row4 scale %d: in-place %v, pre-M-22 reference (interleaveScales4Row) %v", i, gotScales[i], oldScales[i])
		}
	}
}

// TestRepackInt4Row4Quad_streamedMatchesInPlace is the streaming-form gate:
// assembling a tensor's row4 bytes quad-by-quad via repackInt4Row4Quad (the
// primitive a loader streaming off mmap would call, writing into a FRESH
// destination buffer rather than permuting a live one) must match
// RepackInt4InPlace(LayoutRow4)'s result exactly.
func TestRepackInt4Row4Quad_streamedMatchesInPlace(t *testing.T) {
	const group = 32
	if !int4Row4Usable(8, 1536, group) {
		t.Skip("row4 not usable on this core")
	}
	rng := rand.New(rand.NewPCG(22, 9))
	q4, q4s := quantizeInt4Random(rng, 8, 1536, group)
	_, bpr := groupsFor(1536, group)
	nGroups, _ := groupsFor(1536, group)

	streamed := make([]byte, len(q4))
	for base := 0; base < len(q4); base += 4 * bpr {
		repackInt4Row4Quad(streamed[base:base+4*bpr], q4[base:base+4*bpr], 1536)
	}
	streamedScales := make([]float32, len(q4s))
	for base := 0; base < len(q4s); base += 4 * nGroups {
		repackInt4Row4ScalesQuad(streamedScales[base:base+4*nGroups], q4s[base:base+4*nGroups], nGroups)
	}

	w, ok := repackInt4Row4InPlaceF32(cloneBytes(q4), cloneFloats(q4s), 8, 1536, group)
	if !ok {
		t.Fatal("RepackInt4InPlace(LayoutRow4): ok=false")
	}
	gotRow4, gotScales, ok := w.int4Row4Widened()
	if !ok {
		t.Fatal("Int4Row4(): ok=false")
	}
	for i := range streamed {
		if streamed[i] != gotRow4[i] {
			t.Fatalf("streamed row4 byte %d: %d, in-place %d", i, streamed[i], gotRow4[i])
		}
	}
	for i := range streamedScales {
		if streamedScales[i] != gotScales[i] {
			t.Fatalf("streamed row4 scale %d: %v, in-place %v", i, streamedScales[i], gotScales[i])
		}
	}

	// The streamed result is also a valid WrapInt4(LayoutRow4) input — the
	// no-canonical-in-heap-at-all path.
	wStream, ok := wrapInt4Row4OnlyF32(streamed, streamedScales, 8, 1536, group)
	if !ok {
		t.Fatal("WrapInt4(LayoutRow4): ok=false on a shape int4Row4Usable already accepted")
	}
	if _, _, _, canonOK := wStream.int4Widened(); canonOK {
		t.Error("WrapInt4(LayoutRow4) WeightMat reports Int4() ok=true")
	}
}
