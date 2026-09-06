//go:build arm64

package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// TestDotW4A8FoldSDOTv2_matchesOracle checks the new kernel (uncentered nibbles
// + separate batched correction, dot_w4a8_arm64.s) against
// dotW4A8UncenteredScalar across many random shapes, including nGroups not a
// multiple of 4 (exercises the scalar tail) and nGroups==0 mod 4 (exercises
// the pure-vector path with no tail at all). group is fixed at 32 to match
// dotW4A8FoldSDOTv2's contract (whole groups only; the ragged-K mop-up stays
// in dotW4A8's Go dispatch, unchanged, and isn't this kernel's concern).
func TestDotW4A8FoldSDOTv2_matchesOracle(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	rng := rand.New(rand.NewSource(11))
	const group = 32
	// nGroups from 1 to 37 covers every residue mod 4 several times over.
	for nGroups := 1; nGroups <= 37; nGroups++ {
		K := nGroups * group
		for trial := range 5 {
			act := make([]int8, K)
			for i := range act {
				act[i] = int8(rng.Intn(255) - 128)
			}
			w := make([]float32, K)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
			}
			packed, scales := QuantizeGroupsInt4(w, 1, K, group)
			sumAct := make([]int32, nGroups)
			SumActGroupsInto(sumAct, act, 1, K, group)

			want := dotW4A8UncenteredScalar(act, packed, scales, sumAct, group, K)
			got := dotW4A8FoldSDOTv2(&act[0], &packed[0], &scales[0], &sumAct[0], nGroups)

			if math.Abs(float64(got-want)) > 1e-3*math.Abs(float64(want))+1e-3 {
				t.Fatalf("nGroups=%d trial=%d: got %v want %v", nGroups, trial, got, want)
			}
		}
	}
}

// TestDotW4A8SplitHalfSDOT_matchesOracle checks the item-3 split-half-layout
// kernel (dot_w4a8_arm64.s) against dotW4A8SplitHalfScalar
// (w4a8_item3_harness_test.go) — same shape coverage as the SDOTv2 test
// above: nGroups 1..37 covers the loop several times over (this kernel has
// no internal residue/tail of its own, unlike SDOTv2's 4-wide correction
// pass, since it processes exactly one group per loop iteration like the
// original dotW4A8FoldSDOT).
func TestDotW4A8SplitHalfSDOT_matchesOracle(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	rng := rand.New(rand.NewSource(13))
	const group = 32
	for nGroups := 1; nGroups <= 37; nGroups++ {
		K := nGroups * group
		for trial := range 5 {
			act := make([]int8, K)
			for i := range act {
				act[i] = int8(rng.Intn(255) - 128)
			}
			w := make([]float32, K)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
			}
			packed, scales := QuantizeGroupsInt4(w, 1, K, group)
			sh := repackSplitHalfRow(packed, K)

			want := dotW4A8SplitHalfScalar(act, sh, scales, group, K)
			got := dotW4A8SplitHalfSDOT(&act[0], &sh[0], &scales[0], nGroups)

			if math.Abs(float64(got-want)) > 1e-3*math.Abs(float64(want))+1e-3 {
				t.Fatalf("nGroups=%d trial=%d: got %v want %v", nGroups, trial, got, want)
			}
		}
	}
}

// TestDotW4A8FoldSDOT2Acc_matchesOracle checks the two-accumulator probe
// kernel (canonical layout, unchanged centering) against dotW4A8Scalar.
// nGroups must be even (the kernel's contract); this covers every even value
// 2..40.
func TestDotW4A8FoldSDOT2Acc_matchesOracle(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	rng := rand.New(rand.NewSource(17))
	const group = 32
	for nGroups := 2; nGroups <= 40; nGroups += 2 {
		K := nGroups * group
		for trial := range 5 {
			act := make([]int8, K)
			for i := range act {
				act[i] = int8(rng.Intn(255) - 128)
			}
			w := make([]float32, K)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
			}
			packed, scales := QuantizeGroupsInt4(w, 1, K, group)

			want := dotW4A8Scalar(act, packed, scales, group, K)
			got := dotW4A8FoldSDOT2Acc(&act[0], &packed[0], &scales[0], nGroups)

			if math.Abs(float64(got-want)) > 1e-3*math.Abs(float64(want))+1e-3 {
				t.Fatalf("nGroups=%d trial=%d: got %v want %v", nGroups, trial, got, want)
			}
		}
	}
}

// TestDotW4A8FoldSDOT4Acc_matchesOracle checks the four-accumulator probe
// kernel against dotW4A8Scalar. nGroups must be a multiple of 4.
func TestDotW4A8FoldSDOT4Acc_matchesOracle(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	rng := rand.New(rand.NewSource(19))
	const group = 32
	for nGroups := 4; nGroups <= 40; nGroups += 4 {
		K := nGroups * group
		for trial := range 5 {
			act := make([]int8, K)
			for i := range act {
				act[i] = int8(rng.Intn(255) - 128)
			}
			w := make([]float32, K)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
			}
			packed, scales := QuantizeGroupsInt4(w, 1, K, group)

			want := dotW4A8Scalar(act, packed, scales, group, K)
			got := dotW4A8FoldSDOT4Acc(&act[0], &packed[0], &scales[0], nGroups)

			if math.Abs(float64(got-want)) > 1e-3*math.Abs(float64(want))+1e-3 {
				t.Fatalf("nGroups=%d trial=%d: got %v want %v", nGroups, trial, got, want)
			}
		}
	}
}

// TestDotW4A8SplitHalf2Acc_matchesOracle checks the combined
// split-half-layout + 2-accumulator kernel against dotW4A8SplitHalfScalar.
// nGroups must be even.
func TestDotW4A8SplitHalf2Acc_matchesOracle(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	rng := rand.New(rand.NewSource(23))
	const group = 32
	for nGroups := 2; nGroups <= 40; nGroups += 2 {
		K := nGroups * group
		for trial := range 5 {
			act := make([]int8, K)
			for i := range act {
				act[i] = int8(rng.Intn(255) - 128)
			}
			w := make([]float32, K)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
			}
			packed, scales := QuantizeGroupsInt4(w, 1, K, group)
			sh := repackSplitHalfRow(packed, K)

			want := dotW4A8SplitHalfScalar(act, sh, scales, group, K)
			got := dotW4A8SplitHalf2Acc(&act[0], &sh[0], &scales[0], nGroups)

			if math.Abs(float64(got-want)) > 1e-3*math.Abs(float64(want))+1e-3 {
				t.Fatalf("nGroups=%d trial=%d: got %v want %v", nGroups, trial, got, want)
			}
		}
	}
}

// TestDotW4A8SplitHalf4Acc_matchesOracle checks the combined
// split-half-layout + 4-accumulator kernel against dotW4A8SplitHalfScalar.
// nGroups must be a multiple of 4.
func TestDotW4A8SplitHalf4Acc_matchesOracle(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	rng := rand.New(rand.NewSource(29))
	const group = 32
	for nGroups := 4; nGroups <= 40; nGroups += 4 {
		K := nGroups * group
		for trial := range 5 {
			act := make([]int8, K)
			for i := range act {
				act[i] = int8(rng.Intn(255) - 128)
			}
			w := make([]float32, K)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
			}
			packed, scales := QuantizeGroupsInt4(w, 1, K, group)
			sh := repackSplitHalfRow(packed, K)

			want := dotW4A8SplitHalfScalar(act, sh, scales, group, K)
			got := dotW4A8SplitHalf4Acc(&act[0], &sh[0], &scales[0], nGroups)

			if math.Abs(float64(got-want)) > 1e-3*math.Abs(float64(want))+1e-3 {
				t.Fatalf("nGroups=%d trial=%d: got %v want %v", nGroups, trial, got, want)
			}
		}
	}
}

// TestDotW4A8SplitHalf4Row_matchesOracle checks the item-4 (4-real-rows)
// kernel against dotW4A8SplitHalf4RowScalar across nGroups 1..37 (every
// residue — this kernel has no internal residue of its own, one group per
// outer loop iteration, unlike the artificial-lane kernels' mod-2/mod-4
// requirement).
func TestDotW4A8SplitHalf4Row_matchesOracle(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	rng := rand.New(rand.NewSource(41))
	const group = 32
	for nGroups := 1; nGroups <= 37; nGroups++ {
		K := nGroups * group
		for trial := range 5 {
			act := make([]int8, K)
			for i := range act {
				act[i] = int8(rng.Intn(255) - 128)
			}
			rows := make([][]byte, 4)
			scales := make([][]float32, 4)
			for r := range 4 {
				w := make([]float32, K)
				for i := range w {
					w[i] = float32(rng.NormFloat64())
				}
				p, s := QuantizeGroupsInt4(w, 1, K, group)
				rows[r] = p
				scales[r] = s
			}
			packed4 := repackSplitHalf4RowBlock(rows[0], rows[1], rows[2], rows[3], K)
			scales4 := interleaveScales4Row(scales[0], scales[1], scales[2], scales[3], nGroups)

			want := dotW4A8SplitHalf4RowScalar(act, packed4, scales4, group, K)
			var got [4]float32
			dotW4A8SplitHalf4Row(&act[0], &packed4[0], &scales4[0], &got[0], nGroups)

			for r := range 4 {
				if math.Abs(float64(got[r]-want[r])) > 1e-3*math.Abs(float64(want[r]))+1e-3 {
					t.Fatalf("nGroups=%d trial=%d row=%d: got %v want %v", nGroups, trial, r, got[r], want[r])
				}
			}
		}
	}
}

// TestDotW4A8SplitHalf4Row_bitIdenticalToCanonical is the load-bearing
// correctness gate for the plumbing phase (docs/prompts/w4a8-plumbing.md):
// dotW4A8SplitHalf4Row uses ONE accumulator per real output row (cross-row
// independence alone was enough — see the campaign doc's item-4 result), so
// its per-output fold is the SAME monotonic group-by-group order as
// dotW4A8FoldSDOT's, unlike dotW4A8SplitHalf2Acc's within-row 2-way split
// (which folds two independent partial sums together only at the end — a
// genuinely different float reduction tree). Verified here with exact `==`
// against dotW4A8FoldSDOT (canonical layout, production kernel), not
// tolerance: **the winning kernel is bit-identical to the kernel it
// replaces**, for the same logical weights. This means the plumbing phase
// owes no golden regeneration and no cosine re-gate for this kernel swap —
// the existing bit-identity guarantees (decode == prefill == verify) carry
// over unchanged, because the per-output arithmetic literally didn't change,
// only which bytes it reads from and which rows share an activation load.
func TestDotW4A8SplitHalf4Row_bitIdenticalToCanonical(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	rng := rand.New(rand.NewSource(97))
	const group = 32
	// 1..20 sweeps the small group counts densely — every residue mod 4 several
	// times, which is what catches a tail or unroll-boundary bug. But it stops at
	// K=640, and PRODUCTION K IS 1536 AND 8960 (task-simd-audit.md S-09.3): the
	// reference model's hidden and intermediate dims, i.e. nGroups 48 and 280.
	//
	// The structural argument says a group-count-dependent residue cannot exist —
	// the kernel's accumulator layout does not vary with nGroups — and the
	// production-shaped sibling gate (TestMatmulBTW4A8Row4Into_bitIdenticalTo-
	// MatmulBTW4A8Into, which already runs {1536, 8960} and {8960, 1536}) covers
	// the same ground one level up. This adds the two production counts to the RAW
	// KERNEL gate anyway, because "an argument plus a different test" is weaker
	// than the same test executing the shape, and it costs two iterations.
	counts := make([]int, 0, 22)
	for n := 1; n <= 20; n++ {
		counts = append(counts, n)
	}
	counts = append(counts, 48, 280) // K = 1536, 8960
	for _, nGroups := range counts {
		K := nGroups * group
		for trial := range 10 {
			act := make([]int8, K)
			for i := range act {
				act[i] = int8(rng.Intn(255) - 128)
			}
			rows := make([][]byte, 4)
			scales := make([][]float32, 4)
			for r := range 4 {
				w := make([]float32, K)
				for i := range w {
					w[i] = float32(rng.NormFloat64())
				}
				p, s := QuantizeGroupsInt4(w, 1, K, group)
				rows[r] = p
				scales[r] = s
			}
			packed4 := repackSplitHalf4RowBlock(rows[0], rows[1], rows[2], rows[3], K)
			scales4 := interleaveScales4Row(scales[0], scales[1], scales[2], scales[3], nGroups)

			var got [4]float32
			dotW4A8SplitHalf4Row(&act[0], &packed4[0], &scales4[0], &got[0], nGroups)

			for r := range 4 {
				want := dotW4A8FoldSDOT(&act[0], &rows[r][0], &scales[r][0], nGroups)
				if got[r] != want {
					t.Fatalf("nGroups=%d trial=%d row=%d: got %v want %v (bit mismatch — not identical to canonical)", nGroups, trial, r, got[r], want)
				}
			}
		}
	}
}
