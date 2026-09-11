//go:build amd64

package linalg

import (
	"math/rand/v2"
	"testing"
)

// TestMatmulBTW4A8Batch_splitHalfOnlyMatchesCanonical is the batch-path
// twin of TestMatmulBTW4A8Batch_repackedOnlyMatchesCanonical (audit M-22
// follow-up): a W4A8Op with only SplitHalf set (W4/Row4 both nil) must
// match the canonical batch result bit-for-bit at M=1 — split-half has no
// quad-alignment carve-out (unlike row4), so unlike that test there is no
// separate "aligned interior vs edge" case to construct: the whole
// requested range routes through w4a8BatchSplitHalfSpan or none of it does.
func TestMatmulBTW4A8Batch_splitHalfOnlyMatchesCanonical(t *testing.T) {
	if !vnniDeclineCheck(t) {
		return
	}
	assertSplitHalfTileEngages(t, 4, 1536) // capability sanity check, even though this test is M=1
	const group, K = 32, 1536
	rng := rand.New(rand.NewPCG(22, 15))
	const M = 1
	a := mconsistentRand(rng, M*K)

	shapes := []int{256, 512}
	var canonOps, splitOps []W4A8Op
	var wantDst, gotDst [][]float32
	for _, n := range shapes {
		q4, q4s := quantizeInt4Random(rng, n, K, group)
		wD := make([]float32, n)
		gD := make([]float32, n)
		wantDst = append(wantDst, wD)
		gotDst = append(gotDst, gD)
		canonOps = append(canonOps, W4A8Op{W4: q4, Scales: q4s, Dst: wD, N: n})

		splitHalf := RepackW4A8SplitHalf(cloneBytes(q4), n, K, group)
		splitOps = append(splitOps, W4A8Op{SplitHalf: splitHalf, Scales: q4s, Dst: gD, N: n})
	}

	var wsCanon Workspace
	MatmulBTW4A8Batch(&wsCanon, a, M, K, group, canonOps)
	var wsSplit Workspace
	MatmulBTW4A8Batch(&wsSplit, a, M, K, group, splitOps)

	for opIdx := range wantDst {
		for i := range wantDst[opIdx] {
			if g, w := gotDst[opIdx][i], wantDst[opIdx][i]; g != w {
				t.Fatalf("op %d, col %d: split-half-only batch = %v, canonical batch = %v", opIdx, i, g, w)
			}
		}
	}
}

// TestMatmulBTW4A8Batch_splitHalfOnlyM2Panics is the failure-mode half of
// the batch fix: a split-half-only op used at M>1 must panic, naming the
// op and range, rather than silently produce a wrong answer or dereference
// a nil W4/Scales three calls into w4a8Span. There is no batch-path tile
// for split-half (or row4) at M>1 — WeightMat.MatmulBTW4A8Into's own M>1
// tile is a method-level dispatch, not something MatmulBTW4A8Batch reaches.
func TestMatmulBTW4A8Batch_splitHalfOnlyM2Panics(t *testing.T) {
	if !vnniDeclineCheck(t) {
		return
	}
	const group, K = 32, 1536
	rng := rand.New(rand.NewPCG(22, 16))
	const n = 8
	q4, q4s := quantizeInt4Random(rng, n, K, group)
	splitHalf := RepackW4A8SplitHalf(cloneBytes(q4), n, K, group)

	const M = 2
	a := mconsistentRand(rng, M*K)
	dst := make([]float32, M*n)
	op := W4A8Op{SplitHalf: splitHalf, Scales: q4s, Dst: dst, N: n}

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected a panic: split-half-only op at M=2 has no batch-path fallback")
		}
		t.Logf("panicked as expected: %v", r)
	}()
	var ws Workspace
	MatmulBTW4A8Batch(&ws, a, M, K, group, []W4A8Op{op})
}
