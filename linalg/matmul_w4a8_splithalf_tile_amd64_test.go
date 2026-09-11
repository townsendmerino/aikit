//go:build amd64

package linalg

import (
	"math/rand/v2"
	"testing"
)

// assertSplitHalfTileEngages is the "did the fast path actually run" gate
// every split-half M>1 test opens with (audit M-22 follow-up). hasAVX2 &&
// !hasAVX512VNNIVL is necessary but not sufficient — a shape or dispatch
// bug could still route everything through the scalar per-row remainder
// and every bit-identity test would still pass, since both paths compute
// the same right answer (the same trap goinfer's w4a8SplitHalfRepacked
// dispatch counters exist for). So this ALSO calls w4a8SplitHalfTileRows
// directly — package-internal white-box access, no production
// instrumentation needed — on throwaway data and checks its return value
// (mFull, the row count the tile actually consumed) equals M&^3.
//
// t.Skip (not Fatal) when the hardware capability is absent — that is an
// environment limitation, not a bug. t.Fatal when the capability is
// present but the tile still didn't engage, since that IS a real dispatch
// bug: a test that silently passed on the fallback here would prove
// nothing about the tile at all.
func assertSplitHalfTileEngages(t *testing.T, M, K int) {
	t.Helper()
	if !hasAVX2 || hasAVX512VNNIVL {
		t.Skip("split-half tile requires AVX2 without AVX512VNNIVL on this core")
	}
	const group = 32
	nGroups, bpr := groupsFor(K, group)
	aq := make([]int8, M*K)
	aScales := make([]float32, M)
	for i := range aScales {
		aScales[i] = 1 // nonzero: a zero scale takes the zero-shortcut, which looks identical whether or not the tile engaged
	}
	w4sh := make([]byte, bpr) // one row's worth: the probe only needs mFull's value, not real weights
	wScales := make([]float32, nGroups)
	dst := make([]float32, M)
	mFull := w4a8SplitHalfTileRows(aq, aScales, w4sh, wScales, dst, M, K, 1, nGroups, bpr, 0, 1)
	if want := M &^ 3; mFull != want {
		t.Fatalf("split-half tile did not engage: w4a8SplitHalfTileRows returned %d rows at M=%d K=%d, want %d "+
			"(hasAVX2=%v hasAVX512VNNIVL=%v)", mFull, M, K, want, hasAVX2, hasAVX512VNNIVL)
	}
}

// splitHalfTileShapes covers the K values the M-22 split-half-M>1 brief
// asks for, plus one N not a multiple of 8 — ws.parallel's fan-out shards
// on multiples of 8, so an unaligned N exercises a shard boundary that
// lands inside a weight row range rather than only ever on one.
var splitHalfTileShapes = []struct{ K, N int }{
	{1536, 8960}, // Qwen2.5-Coder-1.5B gate/up
	{2048, 2048},
	{4096, 4096},
	{8960, 1536}, // Qwen2.5-Coder-1.5B down-proj
	{4096, 8957}, // N not a multiple of 8
}

// TestMatmulBTW4A8SplitHalfTile_bitIdenticalToCanonical is the split-half
// tile's own gate, held against the canonical M>1 path (MatmulBTW4A8Into
// over the SAME logical weights in canonical packing) — the same
// construction TestMatmulBTW4A8Row4TileInto_bitIdenticalToCanonical uses
// for row4, so it proves the thing callers care about: swapping the
// dispatch changes no bit of any logit.
func TestMatmulBTW4A8SplitHalfTile_bitIdenticalToCanonical(t *testing.T) {
	if !hasAVX2 || hasAVX512VNNIVL {
		t.Skip("split-half tile requires AVX2 without AVX512VNNIVL on this core")
	}
	const group = 32
	rng := rand.New(rand.NewPCG(0x59, 0x74))
	for _, sh := range splitHalfTileShapes {
		K, N := sh.K, sh.N
		assertSplitHalfTileEngages(t, 4, K)
		w := make([]float32, N*K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		q4, q4s := QuantizeGroupsInt4(w, N, K, group)
		splitHalf := RepackW4A8SplitHalf(cloneBytes(q4), N, K, group)

		for _, M := range []int{4, 5, 7, 8, 16, 64} {
			a := make([]float32, M*K)
			for i := range a {
				a[i] = float32(rng.NormFloat64())
			}
			var wsWant, wsGot Workspace
			want := make([]float32, M*N)
			got := make([]float32, M*N)
			MatmulBTW4A8Into(&wsWant, a, q4, q4s, want, M, K, N, group)
			matmulBTW4A8SplitHalfMultiInto(&wsGot, a, splitHalf, q4s, got, M, K, N, group)
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("K=%d N=%d M=%d idx=%d (row %d, col %d): split-half tile %v != canonical %v (diff %v)",
						K, N, M, i, i/N, i%N, got[i], want[i], got[i]-want[i])
				}
			}
		}
	}
}

// TestWeightMatW4A8_MConsistentAcrossSplitHalfDispatch is
// TestWeightMatW4A8_MConsistentAcrossRow4Dispatch's amd64/split-half twin
// (audit M-22 follow-up): WeightMat.MatmulBTW4A8Into now crosses a kernel
// boundary on amd64 too (M=1 decode kernel vs M>1 tile), the combination
// speculative verify actually exercises (draft at M=1, target at M=K), so
// this pins the M=1-alone-vs-inside-a-batch diagonal the same way the row4
// test does.
func TestWeightMatW4A8_MConsistentAcrossSplitHalfDispatch(t *testing.T) {
	const group = 32
	if !hasAVX2 || hasAVX512VNNIVL {
		t.Skip("split-half tile requires AVX2 without AVX512VNNIVL on this core")
	}
	rng := rand.New(rand.NewPCG(0xd1ce, 0x5817))
	maxM := mconsistentMs[len(mconsistentMs)-1]

	for _, c := range quantMConsistentCases {
		t.Run(c.name, func(t *testing.T) {
			if !Int4SplitHalfUsable(c.K, group) {
				t.Skipf("split-half not usable for K=%d group=%d on this core", c.K, group)
			}
			assertSplitHalfTileEngages(t, 4, c.K)
			a := mconsistentRand(rng, maxM*c.K)
			q4, q4s := QuantizeGroupsInt4(mconsistentRand(rng, c.N*c.K), c.N, c.K, group)
			wm := WrapInt4(q4, q4s, c.N, c.K, group)
			if !wm.RepackInt4SplitHalf() {
				t.Fatal("RepackInt4SplitHalf declined despite Int4SplitHalfUsable saying yes")
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
							t.Fatalf("M-dependent: out[%d,%d] at M=%d is %v, alone at M=1 is %v (diff %v); "+
								"WeightMat.MatmulBTW4A8Into must be bit-identical across M — "+
								"this is the decode==prefill==verify guarantee",
								i, j, M, g, w, g-w)
						}
					}
				}
			}
		})
	}
}
