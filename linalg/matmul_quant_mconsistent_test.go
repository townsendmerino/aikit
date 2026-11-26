package linalg

import (
	"math/rand/v2"
	"testing"
)

// mconsistentMs are the batch sizes every quantized M-invariance case sweeps.
// The point of the list is the residues mod 4: a register-blocked M-tile
// (docs/task-simd-audit.md S-01) processes activation rows four at a time and
// hands the remainder to a narrower kernel, so 4 and 8 exercise the pure tile,
// 1/5/9 the 1-row remainder, 2/6 the 2-row, 3/7 the 3-row. M=1 is also the
// value the whole contract is anchored to.
var mconsistentMs = []int{1, 2, 3, 4, 5, 7, 8, 9}

// quantMConsistentCase is one weight shape, quantized once and swept over
// mconsistentMs — the quantization is the expensive part and is M-independent
// by construction, so it is hoisted out of the M loop.
type quantMConsistentCase struct {
	name string
	K, N int
}

var quantMConsistentCases = []quantMConsistentCase{
	// The two real Qwen2.5-Coder-1.5B projection shapes. Both are row4-eligible
	// on arm64 (N%4==0, K%32==0), so on that arch the WeightMat case below
	// crosses the row4/canonical kernel boundary here and nowhere else.
	{"prod_qkv_gate_up", 1536, 8960},
	{"prod_down_proj", 8960, 1536},
	// N not a multiple of 4: row4 is rejected, and any future M-tile must still
	// get the narrow-N column tail right.
	{"n_tail_not_mul4", 96, 6},
	// K not a multiple of the int4 group (32): 3 full groups + a 4-wide tail,
	// which the canonical W4A8 kernel supports and the row4 layout does not.
	{"k_ragged", 100, 8},
	// Small enough to stay under the parallelization threshold at every M in the
	// sweep, so the serial span is exercised as itself rather than incidentally.
	{"small_serial", 64, 8},
}

func mconsistentRand(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

// TestMatmulBTW4A8_MConsistent gates the M-invariance contract for the int4
// weight / int8 activation kernel: output row i of an M-row batch must be
// BIT-IDENTICAL to the same activation row computed alone at M=1.
//
// Why this is the gate that has to exist before S-01: goinfer's decode ==
// batched prefill == speculative verify guarantee is exactly this property.
// The draft model proposes tokens one at a time (M=1) and the target verifies
// a batch (M=K); if W4A8 is M-dependent, the two logit vectors differ by an f32
// reassociation, the occasional greedy argmax flips, and speculative acceptance
// silently drops below 1.0 — a correctness-shaped bug that shows up only as a
// throughput regression.
//
// Today the property holds trivially: w4a8Span loops activation rows INSIDE the
// column loop and calls the same single-row dotW4A8 per (row, column) pair, so
// every output is its own independent reduction. That is precisely the
// inefficiency docs/task-simd-audit.md S-01 proposes to remove with a 4x4
// register-blocked tile — which reduces four activation rows against four weight
// rows in one kernel call. This test is what makes that change safe to ship: it
// pins the numeric contract while the trivial implementation still satisfies it,
// so the tile is measured against a gate that predates it.
//
// TestMatmulBT_MConsistent is the f32 sibling; it covers MatmulBT only, which is
// why W4A8/W8A8 needed their own.
func TestMatmulBTW4A8_MConsistent(t *testing.T) {
	const group = 32
	rng := rand.New(rand.NewPCG(0x5eed, 0x1eaf))
	maxM := mconsistentMs[len(mconsistentMs)-1]

	for _, c := range quantMConsistentCases {
		t.Run(c.name, func(t *testing.T) {
			a := mconsistentRand(rng, maxM*c.K)
			q4, q4s := QuantizeGroupsInt4(mconsistentRand(rng, c.N*c.K), c.N, c.K, group)

			// The M=1 reference for every row, computed once.
			solo := make([]float32, maxM*c.N)
			var wsSolo Workspace
			for i := range maxM {
				matmulBTW4A8Into(&wsSolo, a[i*c.K:(i+1)*c.K], q4, q4s, solo[i*c.N:(i+1)*c.N], 1, c.K, c.N, group)
			}

			for _, M := range mconsistentMs {
				// Both dispatch decisions, since a future M-tile can differ
				// across them: the default threshold (which the production
				// shapes clear and the small shapes do not), and a forced
				// fan-out that shards columns across four workers.
				for _, forced := range []bool{false, true} {
					var ws Workspace
					if forced {
						ws.SetThreshold(1)
						ws.SetWorkers(4)
					}
					got := make([]float32, M*c.N)
					matmulBTW4A8Into(&ws, a, q4, q4s, got, M, c.K, c.N, group)
					for i := range M {
						for j := range c.N {
							if g, w := got[i*c.N+j], solo[i*c.N+j]; g != w {
								t.Fatalf("M-dependent (forcedParallel=%v): out[%d,%d] at M=%d is %v, alone at M=1 is %v (diff %v); "+
									"MatmulBTW4A8Into must be bit-identical across M",
									forced, i, j, M, g, w, g-w)
							}
						}
					}
				}
			}
		})
	}
}

// TestMatmulBTW8A8_MConsistent is TestMatmulBTW4A8_MConsistent's int8-weight
// twin. W8A8's reduction is exact integer arithmetic (int8xint8 accumulated in
// int32, rescaled once at the end), so unlike W4A8 it would survive a
// rearrangement on numerical grounds alone — but the rescale is f32 and the
// contract is what callers rely on, so it gets the same gate. S-01b proposes an
// M-tile here too (16 int32 accumulators, SDOT-issue-bound rather than
// latency-bound); this is the gate that change is measured against.
func TestMatmulBTW8A8_MConsistent(t *testing.T) {
	rng := rand.New(rand.NewPCG(0xb1a5, 0x0ffed))
	maxM := mconsistentMs[len(mconsistentMs)-1]

	for _, c := range quantMConsistentCases {
		t.Run(c.name, func(t *testing.T) {
			a := mconsistentRand(rng, maxM*c.K)
			bq, bs := QuantizeRowsInt8(mconsistentRand(rng, c.N*c.K), c.N, c.K)

			solo := make([]float32, maxM*c.N)
			var wsSolo Workspace
			for i := range maxM {
				matmulBTW8A8Into(&wsSolo, a[i*c.K:(i+1)*c.K], bq, bs, solo[i*c.N:(i+1)*c.N], 1, c.K, c.N)
			}

			for _, M := range mconsistentMs {
				for _, forced := range []bool{false, true} {
					var ws Workspace
					if forced {
						ws.SetThreshold(1)
						ws.SetWorkers(4)
					}
					got := make([]float32, M*c.N)
					matmulBTW8A8Into(&ws, a, bq, bs, got, M, c.K, c.N)
					for i := range M {
						for j := range c.N {
							if g, w := got[i*c.N+j], solo[i*c.N+j]; g != w {
								t.Fatalf("M-dependent (forcedParallel=%v): out[%d,%d] at M=%d is %v, alone at M=1 is %v (diff %v); "+
									"MatmulBTW8A8Into must be bit-identical across M",
									forced, i, j, M, g, w, g-w)
							}
						}
					}
				}
			}
		})
	}
}

// TestWeightMatW4A8_MConsistentAcrossRow4Dispatch is the M-invariance gate at
// the level goinfer actually calls: WeightMat.MatmulBTW4A8Into, which on arm64
// routes M=1 to the split-half 4-row-interleaved kernel and every M>1 to the
// canonical one. So this case is not merely "the same kernel at two M values" —
// it crosses a kernel AND a weight-layout boundary, which is the combination
// speculative verify actually exercises (draft at M=1 through row4, target at
// M=K through canonical).
//
// TestWeightMat_RepackInt4Row4_dispatchMatchesCanonical pins dispatch ==
// canonical at a FIXED M; TestMatmulBTW4A8Row4Into_bitIdenticalToMatmulBTW4A8Into
// pins row4 == canonical at M=1. Neither pins the row4-M=1 vs canonical-M>1
// diagonal, which is the one the guarantee rests on. On non-arm64 RepackInt4Row4
// is a no-op and this degrades to a second run of the canonical case — harmless,
// and it keeps the contract stated in one portable place.
func TestWeightMatW4A8_MConsistentAcrossRow4Dispatch(t *testing.T) {
	const group = 32
	rng := rand.New(rand.NewPCG(0xd1ce, 0x5177))
	maxM := mconsistentMs[len(mconsistentMs)-1]

	for _, c := range quantMConsistentCases {
		t.Run(c.name, func(t *testing.T) {
			a := mconsistentRand(rng, maxM*c.K)
			q4, q4s := QuantizeGroupsInt4(mconsistentRand(rng, c.N*c.K), c.N, c.K, group)
			wm := wrapInt4F32(q4, q4s, c.N, c.K, group)
			repacked := wm.RepackInt4Row4()

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
							t.Fatalf("M-dependent (repacked=%v): out[%d,%d] at M=%d is %v, alone at M=1 is %v (diff %v); "+
								"WeightMat.MatmulBTW4A8Into must be bit-identical across M — "+
								"this is the decode==prefill==verify guarantee",
								repacked, i, j, M, g, w, g-w)
						}
					}
				}
			}
		})
	}
}
