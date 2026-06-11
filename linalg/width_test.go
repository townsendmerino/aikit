package linalg

import (
	"math/rand"
	"runtime"
	"testing"
)

// TestParallelWidth_bitIdentical: capping the fan-out width changes only which
// worker computes which output columns, never the K-reduction of any single
// output — so every width must be bit-identical (the task's parity gate). Run
// forced-parallel so the width actually takes effect.
func TestParallelWidth_bitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	rf := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	rq := func(n int) []int8 {
		v := make([]int8, n)
		for i := range v {
			v[i] = int8(rng.Intn(255) - 127)
		}
		return v
	}
	const M, K, N = 1, 896, 4864
	a, bq, bs := rf(M*K), rq(N*K), rf(N)
	af := rf(N * K) // f32 weights for the MatmulBT check

	oldW := parWidth
	defer SetParallelWidth(oldW)

	withThreshold(false, func() { // force parallel
		// Reference at the default (GOMAXPROCS) width.
		SetParallelWidth(0)
		var ws Workspace
		refQ := make([]float32, M*N)
		MatmulBTW8A8Into(&ws, a, bq, bs, refQ, M, K, N)
		refF := make([]float32, M*N)
		MatmulBT(a, af, refF, M, K, N)

		for _, w := range []int{1, 2, 3, 4, 5, 6, 8} {
			SetParallelWidth(w)
			if got := ParallelWidth(); got != w {
				t.Fatalf("ParallelWidth()=%d, set %d", got, w)
			}
			var ws2 Workspace
			gotQ := make([]float32, M*N)
			MatmulBTW8A8Into(&ws2, a, bq, bs, gotQ, M, K, N)
			gotF := make([]float32, M*N)
			MatmulBT(a, af, gotF, M, K, N)
			for i := range refQ {
				if gotQ[i] != refQ[i] {
					t.Fatalf("W8A8 width=%d elem %d: %v != %v", w, i, gotQ[i], refQ[i])
				}
			}
			for i := range refF {
				if gotF[i] != refF[i] {
					t.Fatalf("MatmulBT width=%d elem %d: %v != %v", w, i, gotF[i], refF[i])
				}
			}
		}
	})
}

// TestParallelWidth_effective: width is clamped to [1, GOMAXPROCS]; 0 = default.
func TestParallelWidth_effective(t *testing.T) {
	oldW := parWidth
	defer SetParallelWidth(oldW)

	SetParallelWidth(0)
	if resolveWidth(0) <= 0 {
		t.Error("width 0 should resolve to GOMAXPROCS (>0)")
	}
	SetParallelWidth(1 << 30) // absurdly high → clamped to GOMAXPROCS
	if got, max := resolveWidth(0), runtime.GOMAXPROCS(0); got != max {
		t.Errorf("over-large width resolved to %d, want GOMAXPROCS %d", got, max)
	}
	SetParallelWidth(2)
	if got := resolveWidth(0); got != 2 && runtime.GOMAXPROCS(0) >= 2 {
		t.Errorf("width 2 resolved to %d, want 2", got)
	}
}
