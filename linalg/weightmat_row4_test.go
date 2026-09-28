package linalg

import (
	"math/rand"
	"testing"
)

// TestWeightMat_RepackInt4Row4_dispatchMatchesCanonical proves the new
// WeightMat.MatmulBTW4A8Into method (docs/task-w4a8-neon-bandwidth.md's
// plumbing phase, item 1) is bit-identical to the canonical path in every
// combination that matters: repacked vs not, M=1 vs M>1, and (on arm64) a
// shape RepackInt4Row4 rejects (rows not a multiple of 4) so the fallback
// branch is exercised too.
func TestWeightMat_RepackInt4Row4_dispatchMatchesCanonical(t *testing.T) {
	const group = 32
	shapes := []struct {
		K, N int
		M    int
	}{
		{1536, 8960, 1}, // real shape, M=1 — the repack's target case
		{1536, 8960, 3}, // M>1 — must fall back regardless of repack
		{64, 6, 1},      // N not a multiple of 4 — RepackInt4Row4 must reject
	}
	rng := rand.New(rand.NewSource(61))
	for _, sh := range shapes {
		K, N, M := sh.K, sh.N, sh.M
		a := make([]float32, M*K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		w := make([]float32, N*K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		q4, q4s := QuantizeGroupsInt4(w, N, K, group)
		f16RoundScales(q4s) // the WeightMat stores binary16 scales; the canonical reference gets the same values
		wm := WrapInt4(q4, q4s, N, K, group)

		var wsWant Workspace
		want := make([]float32, M*N)
		MatmulBTW4A8Into(&wsWant, a, q4, q4s, want, M, K, N, group)

		repacked := wm.RepackInt4Row4()
		if N%4 != 0 {
			if repacked {
				t.Fatalf("K=%d N=%d M=%d: RepackInt4Row4 should have rejected N%%4!=0", K, N, M)
			}
		}

		var wsGot Workspace
		got := make([]float32, M*N)
		wm.MatmulBTW4A8Into(&wsGot, a, got, M)

		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("K=%d N=%d M=%d idx=%d: dispatched %v != canonical %v (repacked=%v)", K, N, M, i, got[i], want[i], repacked)
			}
		}
	}
}
