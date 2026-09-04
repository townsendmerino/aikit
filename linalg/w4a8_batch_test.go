package linalg

import (
	"math/rand/v2"
	"testing"
)

// TestW4A8Batch_bitIdenticalToPerOp is MatmulBTW4A8Batch's contract: it must
// produce EXACTLY the bits that calling MatmulBTW4A8Into once per op produces.
// The batch form exists to remove fork/join barriers and redundant activation
// quantization, never to change a number — goinfer's decode == batched prefill
// == speculative verify guarantee runs through these projections.
//
// The row4 cases are the interesting half. On arm64 an op carrying Row4 takes
// the split-half 4-row kernel while the per-op REFERENCE takes the canonical
// one, so those rows assert equality ACROSS two different kernels and two
// different weight layouts — which is exactly the property that makes the row4
// fast path safe to use inside a batch, and which a same-kernel comparison
// would not have tested at all.
func TestW4A8Batch_bitIdenticalToPerOp(t *testing.T) {
	const group = 32
	rng := rand.New(rand.NewPCG(0xba7c, 0x4a8))

	rv := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	// mkOp builds one op; withRow4 asks for the repacked layout, which
	// RepackInt4Row4's own gate may still decline (N%4, group, DotProd).
	mkOp := func(K, N int, withRow4 bool) W4A8Op {
		q4, q4s := QuantizeGroupsInt4(rv(N*K), N, K, group)
		op := W4A8Op{W4: q4, Scales: q4s, N: N}
		if withRow4 && row4Usable() && N%4 == 0 && K%group == 0 {
			wm := WrapInt4(q4, q4s, N, K, group)
			if wm.RepackInt4Row4() {
				op.Row4, op.Row4Scales = wm.q4Row4, wm.q4Row4Scales
			}
		}
		return op
	}

	cases := []struct {
		name     string
		M, K     int
		Ns       []int
		row4     bool
		forcePar bool
	}{
		{"qkv_row4_M1", 1, 1536, []int{1536, 256, 256}, true, true},
		{"gate_up_row4_M1", 1, 1536, []int{8960, 8960}, true, true},
		{"qkv_canonical_M1", 1, 1536, []int{1536, 256, 256}, false, true},
		// M>1: row4 declines (its kernel is M=1 only), so this pins that the
		// batch falls back correctly rather than mis-dispatching.
		{"prefill_M5", 5, 1536, []int{512, 512}, true, true},
		{"prefill_M8", 8, 512, []int{256, 128}, true, true},
		// N not a multiple of 4 on one op: the quad interior is empty there and
		// the whole op runs canonical inside a batch whose other ops do not.
		{"mixed_N_tail", 1, 512, []int{260, 6, 128}, true, true},
		// K with a ragged final group: row4 cannot represent it at all.
		{"k_ragged", 1, 100, []int{64, 32}, true, true},
		// Below the parallel threshold: the serial named-span fast path.
		{"small_serial", 1, 64, []int{8, 8}, true, false},
		{"single_op", 1, 1536, []int{2048}, true, true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := rv(c.M * c.K)
			ops := make([]W4A8Op, len(c.Ns))
			want := make([][]float32, len(c.Ns))
			for i, N := range c.Ns {
				ops[i] = mkOp(c.K, N, c.row4)
				ops[i].Dst = make([]float32, c.M*N)
				want[i] = make([]float32, c.M*N)
			}

			// Reference: one MatmulBTW4A8Into per op, canonical layout.
			var wsRef Workspace
			for i, op := range ops {
				MatmulBTW4A8Into(&wsRef, a, op.W4, op.Scales, want[i], c.M, c.K, op.N, group)
			}

			for _, forced := range []bool{false, true} {
				if forced && !c.forcePar {
					continue
				}
				for i := range ops {
					clear(ops[i].Dst)
				}
				var ws Workspace
				if forced {
					ws.SetThreshold(1)
					ws.SetWorkers(4)
				}
				MatmulBTW4A8Batch(&ws, a, c.M, c.K, group, ops)
				for i, op := range ops {
					usedRow4 := op.Row4 != nil
					for k := range op.Dst {
						if op.Dst[k] != want[i][k] {
							t.Fatalf("op %d (N=%d, row4=%v, forcedParallel=%v) idx %d: batch %v != per-op %v (diff %v)",
								i, op.N, usedRow4, forced, k, op.Dst[k], want[i][k], op.Dst[k]-want[i][k])
						}
					}
				}
			}
		})
	}
}

// TestW4A8Batch_widthInert pins that the batch's fan-out is numerically inert,
// the same contract TestParallelWidth_bitIdentical holds for the rest of the
// package: the parallel region partitions the ops' concatenated OUTPUT columns,
// never a reduction, so any worker count must give identical bits — including
// widths that split a single op's columns across workers and widths that put
// several whole ops in one shard.
func TestW4A8Batch_widthInert(t *testing.T) {
	const group, K, M = 32, 512, 1
	rng := rand.New(rand.NewPCG(0x1de, 0x11))
	rv := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	a := rv(M * K)
	Ns := []int{256, 128, 64}
	build := func() []W4A8Op {
		ops := make([]W4A8Op, len(Ns))
		r := rand.New(rand.NewPCG(0x1de, 0x22))
		for i, N := range Ns {
			w := make([]float32, N*K)
			for j := range w {
				w[j] = float32(r.NormFloat64())
			}
			q4, q4s := QuantizeGroupsInt4(w, N, K, group)
			ops[i] = W4A8Op{W4: q4, Scales: q4s, N: N, Dst: make([]float32, M*N)}
			if row4Usable() {
				wm := WrapInt4(q4, q4s, N, K, group)
				if wm.RepackInt4Row4() {
					ops[i].Row4, ops[i].Row4Scales = wm.q4Row4, wm.q4Row4Scales
				}
			}
		}
		return ops
	}

	serial := build()
	var wsS Workspace
	wsS.SetThreshold(1 << 62)
	MatmulBTW4A8Batch(&wsS, a, M, K, group, serial)

	for _, workers := range []int{1, 2, 3, 5, 8, 16} {
		got := build()
		var ws Workspace
		ws.SetThreshold(1)
		ws.SetWorkers(workers)
		MatmulBTW4A8Batch(&ws, a, M, K, group, got)
		for i := range got {
			for k := range got[i].Dst {
				if got[i].Dst[k] != serial[i].Dst[k] {
					t.Fatalf("workers=%d op %d idx %d: %v != serial %v",
						workers, i, k, got[i].Dst[k], serial[i].Dst[k])
				}
			}
		}
	}
}

// TestW4A8Batch_emptyOpsIsNoop pins the degenerate entry, which must not panic
// and must not touch the Workspace — a caller assembling ops conditionally can
// legitimately end up with none.
func TestW4A8Batch_emptyOpsIsNoop(t *testing.T) {
	var ws Workspace
	MatmulBTW4A8Batch(&ws, nil, 1, 64, 32, nil)
	MatmulBTW4A8Batch(&ws, nil, 1, 64, 32, []W4A8Op{})
}
