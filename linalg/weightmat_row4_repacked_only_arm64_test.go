//go:build arm64

package linalg

import (
	"math/rand/v2"
	"testing"
)

// TestWeightMatW4A8_MConsistentRepackedOnly is
// TestWeightMatW4A8_MConsistentAcrossRow4Dispatch (matmul_quant_mconsistent_
// test.go) run a second time against a repacked-only WeightMat built by
// RepackInt4Row4InPlace — audit M-22, step 3 of the brief. Same cases, same M
// sweep, same decode==prefill==verify guarantee, this time with no
// canonical bytes behind it at all: a repacked-only tensor must be exactly
// as M-consistent as a "both" one, since MatmulBTW4A8Into's row4/tile
// dispatch is the SAME code path for both — the only difference is whether
// w.q4 is nil.
func TestWeightMatW4A8_MConsistentRepackedOnly(t *testing.T) {
	const group = 32
	rng := rand.New(rand.NewPCG(0xd1ce, 0x2202)) // audit M-22
	maxM := mconsistentMs[len(mconsistentMs)-1]

	for _, c := range quantMConsistentCases {
		t.Run(c.name, func(t *testing.T) {
			if !Int4Row4Usable(c.N, c.K, group) {
				t.Skipf("row4 not usable for N=%d K=%d group=%d on this core", c.N, c.K, group)
			}
			a := mconsistentRand(rng, maxM*c.K)
			// q4/q4s stay UNTOUCHED — RepackInt4Row4InPlace below mutates its
			// input, and the "both" comparison further down needs the
			// original canonical bytes, so it gets its own clone too.
			q4, q4s := quantizeInt4Random(rng, c.N, c.K, group)
			wm, ok := RepackInt4Row4InPlace(cloneBytes(q4), cloneFloats(q4s), c.N, c.K, group)
			if !ok {
				t.Fatalf("RepackInt4Row4InPlace: ok=false, Int4Row4Usable said yes for N=%d K=%d", c.N, c.K)
			}
			if _, _, _, canonOK := wm.Int4(); canonOK {
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
			both := WrapInt4(q4, q4s, c.N, c.K, group)
			if !both.RepackInt4Row4() {
				t.Fatal("both: RepackInt4Row4 declined after Int4Row4Usable said yes")
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

// TestRepackInt4Row4InPlace_matchesOutOfPlace is step 3's byte-for-byte
// gate: the in-place result must equal RepackW4A8Row4/RepackW4A8Row4Scales's
// out-of-place output for the SAME canonical input.
func TestRepackInt4Row4InPlace_matchesOutOfPlace(t *testing.T) {
	const group = 32
	if !Int4Row4Usable(8, 1536, group) {
		t.Skip("row4 not usable on this core")
	}
	rng := rand.New(rand.NewPCG(22, 8))
	q4, q4s := quantizeInt4Random(rng, 8, 1536, group)

	wantRow4 := RepackW4A8Row4(cloneBytes(q4), 8, 1536, group)
	wantScales := RepackW4A8Row4Scales(cloneFloats(q4s), 8, 1536, group)

	w, ok := RepackInt4Row4InPlace(q4, q4s, 8, 1536, group)
	if !ok {
		t.Fatal("RepackInt4Row4InPlace: ok=false")
	}
	gotRow4, gotScales, ok := w.Int4Row4()
	if !ok {
		t.Fatal("Int4Row4(): ok=false")
	}
	for i := range gotRow4 {
		if gotRow4[i] != wantRow4[i] {
			t.Fatalf("row4 byte %d: in-place %d, out-of-place %d", i, gotRow4[i], wantRow4[i])
		}
	}
	for i := range gotScales {
		if gotScales[i] != wantScales[i] {
			t.Fatalf("row4 scale %d: in-place %v, out-of-place %v", i, gotScales[i], wantScales[i])
		}
	}
}

// TestRepackInt4Row4Quad_streamedMatchesInPlace is the streaming-form gate:
// assembling a tensor's row4 bytes quad-by-quad via RepackInt4Row4Quad (the
// primitive a loader streaming off mmap would call, writing into a FRESH
// destination buffer rather than permuting a live one) must match
// RepackInt4Row4InPlace's result exactly.
func TestRepackInt4Row4Quad_streamedMatchesInPlace(t *testing.T) {
	const group = 32
	if !Int4Row4Usable(8, 1536, group) {
		t.Skip("row4 not usable on this core")
	}
	rng := rand.New(rand.NewPCG(22, 9))
	q4, q4s := quantizeInt4Random(rng, 8, 1536, group)
	_, bpr := groupsFor(1536, group)
	nGroups, _ := groupsFor(1536, group)

	streamed := make([]byte, len(q4))
	for base := 0; base < len(q4); base += 4 * bpr {
		RepackInt4Row4Quad(streamed[base:base+4*bpr], q4[base:base+4*bpr], 1536)
	}
	streamedScales := make([]float32, len(q4s))
	for base := 0; base < len(q4s); base += 4 * nGroups {
		RepackInt4Row4ScalesQuad(streamedScales[base:base+4*nGroups], q4s[base:base+4*nGroups], nGroups)
	}

	w, ok := RepackInt4Row4InPlace(cloneBytes(q4), cloneFloats(q4s), 8, 1536, group)
	if !ok {
		t.Fatal("RepackInt4Row4InPlace: ok=false")
	}
	gotRow4, gotScales, ok := w.Int4Row4()
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

	// The streamed result is also a valid WrapInt4Row4Only input — the
	// no-canonical-in-heap-at-all path.
	wStream, ok := WrapInt4Row4Only(streamed, streamedScales, 8, 1536, group)
	if !ok {
		t.Fatal("WrapInt4Row4Only: ok=false on a shape Int4Row4Usable already accepted")
	}
	if _, _, _, canonOK := wStream.Int4(); canonOK {
		t.Error("WrapInt4Row4Only WeightMat reports Int4() ok=true")
	}
}
