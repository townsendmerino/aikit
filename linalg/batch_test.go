package linalg

import (
	"math/rand"
	"testing"
)

// TestW8A8Batch_bitIdenticalToPerOp: MatmulBTW8A8Batch must produce EXACTLY the
// same bytes as calling MatmulBTW8A8Into once per op — the whole point is a
// dispatch optimization with zero numeric change (so goinfer's HF-logit parity
// is untouched). Checked under both the serial and parallel thresholds, since
// the column partitioning differs between them.
func TestW8A8Batch_bitIdenticalToPerOp(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	randF := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	randQ := func(n int) []int8 {
		v := make([]int8, n)
		for i := range v {
			v[i] = int8(rng.Intn(255) - 127)
		}
		return v
	}

	for _, M := range []int{1, 3} {
		const K = 896
		ns := []int{896, 128, 128} // qkv-like, uneven so spans cross op boundaries
		a := randF(M * K)
		ops := make([]W8A8Op, len(ns))
		ref := make([]W8A8Op, len(ns))
		for i, N := range ns {
			bq, bs := randQ(N*K), randF(N)
			ops[i] = W8A8Op{BQ: bq, Scales: bs, Dst: make([]float32, M*N), N: N}
			ref[i] = W8A8Op{BQ: bq, Scales: bs, Dst: make([]float32, M*N), N: N}
		}

		for _, serial := range []bool{true, false} {
			withThreshold(serial, func() {
				var ws Workspace
				// Reference: one Into call per op.
				for _, op := range ref {
					MatmulBTW8A8Into(&ws, a, op.BQ, op.Scales, op.Dst, M, K, op.N)
				}
				// Batched.
				MatmulBTW8A8Batch(&ws, a, M, K, ops)
				for i := range ops {
					for j := range ops[i].Dst {
						if ops[i].Dst[j] != ref[i].Dst[j] {
							mode := "parallel"
							if serial {
								mode = "serial"
							}
							t.Fatalf("M=%d %s op %d elem %d: batch %v != per-op %v",
								M, mode, i, j, ops[i].Dst[j], ref[i].Dst[j])
						}
					}
				}
			})
		}
	}
}

// TestW8A8Into_bitIdenticalToWrapper: the zero-alloc Into path must match the
// allocating MatmulBTW8A8 wrapper exactly.
func TestW8A8Into_bitIdenticalToWrapper(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const M, K, N = 2, 640, 1024
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	bq := make([]int8, N*K)
	for i := range bq {
		bq[i] = int8(rng.Intn(255) - 127)
	}
	bs := make([]float32, N)
	for i := range bs {
		bs[i] = float32(rng.NormFloat64())
	}
	want := make([]float32, M*N)
	MatmulBTW8A8(a, bq, bs, want, M, K, N)

	var ws Workspace
	got := make([]float32, M*N)
	MatmulBTW8A8Into(&ws, a, bq, bs, got, M, K, N)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("elem %d: Into %v != wrapper %v", i, got[i], want[i])
		}
	}
}

// TestW8A8Reblock_matchesPerRow: the column-outer re-block must be bit-identical
// to computing each row independently with an M=1 call (M=1 is the unchanged
// single-row order). Covers the M>1 path the re-block actually changes
// (speculative verify, prefill, encoder).
func TestW8A8Reblock_matchesPerRow(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
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
	for _, s := range []struct{ M, K, N int }{
		{4, 896, 1024}, {8, 640, 300}, {2, 256, 4864},
	} {
		a, bq, bs := rf(s.M*s.K), rq(s.N*s.K), rf(s.N)
		gotM := make([]float32, s.M*s.N)
		MatmulBTW8A8(a, bq, bs, gotM, s.M, s.K, s.N)
		for i := 0; i < s.M; i++ {
			row := make([]float32, s.N)
			MatmulBTW8A8(a[i*s.K:(i+1)*s.K], bq, bs, row, 1, s.K, s.N)
			for j := 0; j < s.N; j++ {
				if gotM[i*s.N+j] != row[j] {
					t.Fatalf("shape %+v row %d col %d: M>1 %v != per-row M=1 %v",
						s, i, j, gotM[i*s.N+j], row[j])
				}
			}
		}
	}
}

// TestW8A8Into_zeroAllocWhenReused: a reused Workspace must make steady-state
// Into calls allocation-free (the decode-alloc fix).
func TestW8A8Into_zeroAllocWhenReused(t *testing.T) {
	const M, K, N = 1, 896, 4864
	a := randF(M * K)
	bq := randI8(N * K)
	bs := randF(N)
	dst := make([]float32, M*N)
	var ws Workspace
	MatmulBTW8A8Into(&ws, a, bq, bs, dst, M, K, N) // warm the workspace

	avg := testing.AllocsPerRun(20, func() {
		MatmulBTW8A8Into(&ws, a, bq, bs, dst, M, K, N)
	})
	if avg != 0 {
		t.Errorf("reused Into allocs = %v/op, want 0", avg)
	}
}
