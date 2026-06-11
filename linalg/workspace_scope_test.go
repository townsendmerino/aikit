package linalg

import (
	"math/rand/v2"
	"sync"
	"testing"
)

func randVec(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}

// TestWorkspace_scopedThreshold: a per-Workspace SetThreshold overrides the global
// for that Workspace's matmuls (bit-identically — parallelization is numerically
// inert), without mutating the process default, and the zero value inherits it.
func TestWorkspace_scopedThreshold(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	M, K, N := 4, 64, 256
	a, b := randVec(rng, M*K), randVec(rng, N*K)
	ref := make([]float32, M*N)
	MatmulBT(a, b, ref, M, K, N)

	for _, thr := range []int{0, 1 << 30} { // always-parallel and never-parallel
		var ws Workspace
		ws.SetThreshold(thr)
		got := make([]float32, M*N)
		ws.MatmulBT(a, b, got, M, K, N)
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("thr=%d ws.MatmulBT[%d]=%v != %v", thr, i, got[i], ref[i])
			}
		}
	}

	if ParallelThreshold() != 1<<24 {
		t.Errorf("SetThreshold leaked to the global: %d", ParallelThreshold())
	}
	var z Workspace
	if z.thr() != ParallelThreshold() {
		t.Errorf("zero Workspace thr %d != global default %d", z.thr(), ParallelThreshold())
	}
}

// TestWorkspace_scopedW8A8: the W8A8 path honors the Workspace threshold too, and
// independent Workspaces with different thresholds run concurrently without racing
// on a global (run under -race).
func TestWorkspace_scopedW8A8(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	M, K, N := 3, 128, 512
	a := randVec(rng, M*K)
	bQ, bScales := QuantizeRowsInt8(randVec(rng, N*K), N, K)

	ref := make([]float32, M*N)
	var base Workspace
	base.SetThreshold(1 << 30) // serial reference
	MatmulBTW8A8Into(&base, a, bQ, bScales, ref, M, K, N)

	var wg sync.WaitGroup
	for _, thr := range []int{0, 1 << 30} {
		wg.Add(1)
		go func(thr int) {
			defer wg.Done()
			var ws Workspace
			ws.SetThreshold(thr)
			got := make([]float32, M*N)
			MatmulBTW8A8Into(&ws, a, bQ, bScales, got, M, K, N)
			for i := range ref {
				if got[i] != ref[i] {
					t.Errorf("thr=%d W8A8[%d]=%v != %v", thr, i, got[i], ref[i])
					return
				}
			}
		}(thr)
	}
	wg.Wait()
}

// TestWorkspace_scopedWidth: SetWorkers caps the fan-out per-Workspace and is
// numerically inert (parallel matmuls partition columns — any width is bit-identical
// to the serial reference), without touching the global SetParallelWidth default.
func TestWorkspace_scopedWidth(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	M, K, N := 4, 64, 300
	a, b := randVec(rng, M*K), randVec(rng, N*K)
	ref := make([]float32, M*N)
	MatmulBT(a, b, ref, M, K, N)

	for _, width := range []int{0, 1, 2, 3, 1 << 20} {
		var ws Workspace
		ws.SetThreshold(0) // force parallel
		ws.SetWorkers(width)
		got := make([]float32, M*N)
		ws.MatmulBT(a, b, got, M, K, N)
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("width=%d MatmulBT[%d]=%v != %v", width, i, got[i], ref[i])
			}
		}
	}
	if ParallelWidth() != 0 {
		t.Errorf("SetWorkers leaked to the global ParallelWidth: %d", ParallelWidth())
	}
}
