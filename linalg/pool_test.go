package linalg

import (
	"math/rand"
	"testing"
)

// TestPool_bitIdenticalToSpawn: the persistent worker pool must produce EXACTLY
// the same output as the spawn-per-call path (which already matches serial) —
// it's a dispatch change only, so goinfer's HF parity is untouched. Forced
// parallel (threshold 0) so the pool path actually runs.
func TestPool_bitIdenticalToSpawn(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
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

	withThreshold(false, func() { // force parallel
		for _, s := range []struct{ M, K, N int }{
			{1, 896, 896}, {1, 896, 4864}, {3, 640, 1000}, {1, 4864, 896},
		} {
			a, bq, bs := rf(s.M*s.K), rq(s.N*s.K), rf(s.N)

			var spawn Workspace // no pool → parallelSpawnCols
			gotSpawn := make([]float32, s.M*s.N)
			MatmulBTW8A8Into(&spawn, a, bq, bs, gotSpawn, s.M, s.K, s.N)

			for _, workers := range []int{1, 2, 4, 8} {
				var pooled Workspace
				pooled.SetWorkers(workers)
				gotPool := make([]float32, s.M*s.N)
				// Run several times: exercises gen reuse + spin/park cycles.
				for range 5 {
					MatmulBTW8A8Into(&pooled, a, bq, bs, gotPool, s.M, s.K, s.N)
				}
				pooled.Close()
				for i := range gotSpawn {
					if gotPool[i] != gotSpawn[i] {
						t.Fatalf("shape %+v workers=%d elem %d: pool %v != spawn %v",
							s, workers, i, gotPool[i], gotSpawn[i])
					}
				}
			}
		}
	})
}

// TestPool_batchBitIdentical: the batched primitive through the pool matches the
// spawn path too.
func TestPool_batchBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
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
	const M, K = 1, 896
	ns := []int{896, 128, 128}
	a := rf(M * K)
	mk := func() []W8A8Op {
		ops := make([]W8A8Op, len(ns))
		for i, N := range ns {
			ops[i] = W8A8Op{BQ: rq(N * K), Scales: rf(N), Dst: make([]float32, M*N), N: N}
		}
		return ops
	}
	withThreshold(false, func() {
		spawnOps := mk()
		var spawn Workspace
		MatmulBTW8A8Batch(&spawn, a, M, K, spawnOps)

		poolOps := mk()
		var pooled Workspace
		pooled.SetWorkers(6)
		defer pooled.Close()
		for range 4 {
			MatmulBTW8A8Batch(&pooled, a, M, K, poolOps)
		}
		// Reset spawn weights/scales must equal pool's? They use different
		// random weights (mk re-draws), so compare each path against a serial
		// reference of ITS OWN inputs instead. Recompute spawn reference here.
		for oi := range poolOps {
			ref := make([]float32, M*poolOps[oi].N)
			var ws Workspace
			MatmulBTW8A8Into(&ws, a, poolOps[oi].BQ, poolOps[oi].Scales, ref, M, K, poolOps[oi].N)
			for i := range ref {
				if poolOps[oi].Dst[i] != ref[i] {
					t.Fatalf("op %d elem %d: pool-batch %v != per-op %v", oi, i, poolOps[oi].Dst[i], ref[i])
				}
			}
		}
		_ = spawnOps
	})
}

// TestPool_reuseAndClose: many dispatches of varying N on one pool, then Close —
// exercises gen reuse, empty chunks (N < workers), and clean shutdown. Run under
// -race to catch dispatch/worker data races.
func TestPool_reuseAndClose(t *testing.T) {
	var ws Workspace
	ws.SetWorkers(8)
	a := make([]float32, 4864)
	for i := range a {
		a[i] = float32(i%7) - 3
	}
	withThreshold(false, func() {
		for _, N := range []int{1, 2, 3, 5, 128, 896, 4864} {
			K := 896
			bq := make([]int8, N*K)
			bs := make([]float32, N)
			for i := range bs {
				bs[i] = 0.01
			}
			dst := make([]float32, N)
			for range 3 {
				MatmulBTW8A8Into(&ws, a[:K], bq, bs, dst, 1, K, N)
			}
		}
	})
	ws.Close()
	ws.Close() // idempotent
}
