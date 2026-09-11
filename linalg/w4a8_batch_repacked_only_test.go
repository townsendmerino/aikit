//go:build arm64

// row4 has no amd64/portable twin (RepackW4A8Row4/RepackW4A8Row4Scales live
// only in matmul_w4a8_row4_arm64.go), so this file — which builds row4 test
// data directly rather than going through a gated wrapper — is arm64-only.
// Int4Row4Usable/row4Usable are false everywhere off arm64 anyway, so
// nothing here would ever run on another arch.
package linalg

import (
	"math/rand/v2"
	"testing"
)

// TestMatmulBTW4A8Batch_repackedOnlyMatchesCanonical is step 3 of the M-22
// brief: a batch op with W4 nil / Row4 set (w4a8BatchOp's audit M-22 fix)
// must match the canonical batch result bit-for-bit, at both an aligned
// shape (every op's own N%4==0, so the quad-aligned interior covers the
// whole op and canonical() is never actually invoked) and — separately — a
// direct check that a genuinely unaligned edge panics instead of
// dereferencing a nil W4/Scales.
func TestMatmulBTW4A8Batch_repackedOnlyMatchesCanonical(t *testing.T) {
	const group, K = 32, 1536
	if !Int4Row4Usable(8, K, group) {
		t.Skip("row4 not usable on this core")
	}
	rng := rand.New(rand.NewPCG(22, 5))
	const M = 1
	a := mconsistentRand(rng, M*K)

	// Two ops sharing one activation, like a real q‖k‖v / gate‖up fused
	// call — both N a multiple of 4, so the batch's shard boundaries (8-
	// aligned) land on quad boundaries for every op and the row4 path
	// covers the whole batch.
	shapes := []int{256, 512}
	var canonOps, repackedOps []W4A8Op
	var wantDst, gotDst [][]float32
	for _, n := range shapes {
		q4, q4s := quantizeInt4Random(rng, n, K, group)
		wD := make([]float32, n)
		gD := make([]float32, n)
		wantDst = append(wantDst, wD)
		gotDst = append(gotDst, gD)
		canonOps = append(canonOps, W4A8Op{W4: q4, Scales: q4s, Dst: wD, N: n})

		row4, row4s := RepackW4A8Row4(q4, n, K, group), RepackW4A8Row4Scales(q4s, n, K, group)
		repackedOps = append(repackedOps, W4A8Op{Row4: row4, Row4Scales: row4s, Dst: gD, N: n})
	}

	var wsCanon Workspace
	MatmulBTW4A8Batch(&wsCanon, a, M, K, group, canonOps)
	var wsRepacked Workspace
	MatmulBTW4A8Batch(&wsRepacked, a, M, K, group, repackedOps)

	for opIdx := range wantDst {
		for i := range wantDst[opIdx] {
			if g, w := gotDst[opIdx][i], wantDst[opIdx][i]; g != w {
				t.Fatalf("op %d, col %d: repacked-only batch = %v, canonical batch = %v", opIdx, i, g, w)
			}
		}
	}
}

// TestMatmulBTW4A8Batch_repackedOnlyUnalignedEdgePanics is the failure-mode
// half of the M-22 batch fix: a repacked-only op whose requested column
// range straddles a non-quad boundary must panic, naming the op and range,
// rather than dereference W4A8Op.W4 == nil three calls into w4a8Span.
func TestMatmulBTW4A8Batch_repackedOnlyUnalignedEdgePanics(t *testing.T) {
	const group, K = 32, 1536
	if !row4Usable() {
		t.Skip("row4Usable() is false on this core")
	}
	rng := rand.New(rand.NewPCG(22, 6))
	const n = 8 // row4-shaped (N%4==0), but the call below asks for a non-quad-aligned slice
	q4, q4s := quantizeInt4Random(rng, n, K, group)
	row4, row4s := RepackW4A8Row4(q4, n, K, group), RepackW4A8Row4Scales(q4s, n, K, group)

	aq := make([]int8, K)
	aScales := []float32{1}
	dst := make([]float32, n)
	op := W4A8Op{Row4: row4, Row4Scales: row4s, Dst: dst, N: n}
	nGroups, bpr := groupsFor(K, group)

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected a panic: columns [1,3) straddle a quad boundary with no canonical fallback")
		}
		t.Logf("panicked as expected: %v", r)
	}()
	// [1,3) is not quad-aligned (quads are [0,4), [4,8)) and op.W4 is nil —
	// this must panic, not silently read op.W4[...] as a zero-length slice.
	w4a8BatchOp(aq, aScales, op, 1, K, group, nGroups, bpr, 1, 3)
}
