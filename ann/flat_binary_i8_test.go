package ann

import (
	"github.com/townsendmerino/aikit/linalg/bitvec"
	"math"
	"testing"
)

// TestFlatBinaryI8_exactWhenNothingToPrune pins the boundary where
// FlatBinaryI8 stops being approximate. If the candidate set is the whole
// corpus — k <= 0, k >= Len, or overquery·k >= Len — a prefilter can discard
// nothing, so the result must be FlatI8's own exact int8 scan, hit for hit AND
// score bit for bit (not Flat's float32 one — that is a second, independent
// source of difference FlatBinaryI8 does not remove).
func TestFlatBinaryI8_exactWhenNothingToPrune(t *testing.T) {
	vecs := unitVecs(300, 64, 38)
	q := unitVecs(1, 64, 99)[0]
	i8 := NewFlatI8(vecs)

	for _, tc := range []struct {
		name string
		over int
		k    int
	}{
		{"k=0 returns all", 4, 0},
		{"k=-1 returns all", 4, -1},
		{"k>=Len", 4, 300},
		{"overquery covers the corpus", 400, 5},
		{"overquery exactly covers", 60, 5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fb := NewFlatBinaryI8Overquery(vecs, tc.over)
			want := i8.Query(q, tc.k)
			got := fb.Query(q, tc.k)
			if len(got) != len(want) {
				t.Fatalf("got %d hits, FlatI8 returned %d", len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("hit %d: got %+v, FlatI8 returned %+v", i, got[i], want[i])
				}
			}
		})
	}
}

// TestFlatBinaryI8_rerankScoresMatchFlatI8 checks the part of the contract
// that survives approximation: whatever comes back is scored exactly the way
// FlatI8 would score that same document, even though membership may not be
// the true top-k. The rerank gathers each candidate's int8 code + scale out of
// the SAME FlatI8 storage FlatBinaryI8.exact holds and runs the same
// MatmulBTW8A8Into kernel, so this must be bit-identical, not merely close —
// an off-by-one in the gather (wrong id's code attached to a score) would
// still often pass a pure recall test and would fail this one.
func TestFlatBinaryI8_rerankScoresMatchFlatI8(t *testing.T) {
	vecs := unitVecs(5000, 128, 380)
	fb := NewFlatBinaryI8(vecs)
	i8 := NewFlatI8(vecs)
	for qi := range 20 {
		q := unitVecs(1, 128, int64(1000+qi))[0]
		hits := fb.Query(q, 10)
		if len(hits) != 10 {
			t.Fatalf("query %d: got %d hits, want 10", qi, len(hits))
		}
		// FlatI8's own full-corpus score for this exact document, for
		// comparison — k >= Len so this is FlatI8's exact scan.
		full := i8.Query(q, i8.Len())
		byID := make(map[int]float64, len(full))
		for _, h := range full {
			byID[h.Index] = h.Score
		}
		for i, h := range hits {
			want, ok := byID[h.Index]
			if !ok {
				t.Fatalf("query %d hit %d: doc %d not found in FlatI8's own scan", qi, i, h.Index)
			}
			if h.Score != want {
				t.Fatalf("query %d hit %d (doc %d): rerank score %v != FlatI8 score %v", qi, i, h.Index, h.Score, want)
			}
			if i > 0 && hits[i-1].Score < h.Score {
				t.Fatalf("query %d: hits out of descending order at %d", qi, i)
			}
		}
	}
}

// TestFlatBinaryI8_recallClustered is the non-skipping recall floor, on a
// corpus that HAS neighbors to find — see TestFlatBinary_recallClustered for
// why unstructured data cannot measure this. int8 quantization is a second,
// independent source of approximation stacked on the same prefilter, but it
// costs less than expected: 0.964 measured here against FlatBinary's 1.0000
// on its own (differently-seeded) clustered corpus in the same file — close
// enough that 0.85 leaves real margin rather than sitting at the noise floor.
func TestFlatBinaryI8_recallClustered(t *testing.T) {
	const dim, k = 256, 10
	vecs, qs := binClusteredCorpus(200, 100, dim, 50, 0.5, 3800)
	flat := New(vecs)
	fb := NewFlatBinaryI8(vecs)

	recall := binRecallAt(flat, fb, qs, k)
	t.Logf("recall@%d over %d queries, %d clustered vectors, overquery %d: %.4f",
		k, len(qs), len(vecs), fb.Overquery(), recall)
	if recall < 0.85 {
		t.Errorf("recall@%d = %.4f, want >= 0.85", k, recall)
	}
}

// TestFlatBinaryI8_overqueryRaisesRecall is the mutation-resistant shape
// check FlatBinary's own version is — recall must be MONOTONE in the
// candidate multiplier.
func TestFlatBinaryI8_overqueryRaisesRecall(t *testing.T) {
	const dim, k = 256, 10
	vecs, qs := binClusteredCorpus(200, 100, dim, 40, 0.5, 3801)
	flat := New(vecs)

	var prev float64
	for _, over := range []int{1, 2, 4, 8, 16} {
		r := binRecallAt(flat, NewFlatBinaryI8Overquery(vecs, over), qs, k)
		t.Logf("overquery %2d: recall@%d = %.4f", over, k, r)
		if r < prev-1e-9 {
			t.Errorf("recall fell from %.4f to %.4f when overquery rose to %d", prev, r, over)
		}
		prev = r
	}
	if prev < 0.90 {
		t.Errorf("recall@%d at overquery 16 = %.4f, want >= 0.90", k, prev)
	}
}

// TestFlatBinaryI8_filterAppliedInPrefilter mirrors
// TestFlatBinary_filterAppliedInPrefilter: keep runs during the prefilter,
// not after it, regardless of which rerank stage follows — the prefilter
// code is shared (binaryPrefilter), so this is really the same behavior
// under test, gated a second time because FlatBinaryI8 is a second public
// entry point into it.
func TestFlatBinaryI8_filterAppliedInPrefilter(t *testing.T) {
	const dim, k = 128, 10
	vecs, qs := binClusteredCorpus(200, 100, dim, 1, 0.5, 3802)
	fb := NewFlatBinaryI8(vecs)
	flat := New(vecs)
	keep := func(id int) bool { return id%50 == 0 }

	q := qs[0]
	got := fb.QueryFilter(q, k, keep)
	if len(got) != k {
		t.Fatalf("got %d hits, want %d — the filter is being applied after candidate selection", len(got), k)
	}
	for _, h := range got {
		if !keep(h.Index) {
			t.Fatalf("hit %d is filtered out but was returned", h.Index)
		}
	}
	truth := map[int]bool{}
	for _, h := range flat.QueryFilter(q, k, keep) {
		truth[h.Index] = true
	}
	hits := 0
	for _, h := range got {
		if truth[h.Index] {
			hits++
		}
	}
	if hits < 7 {
		t.Errorf("only %d/%d of the filtered top-%d recovered", hits, k, k)
	}
}

// TestFlatBinaryI8_poolIsInert poisons every pooled scratch buffer before a
// query and checks the answer is unchanged — mirrors
// TestFlatBinary_poolIsInert, extended to the int8 gather buffers and the
// W8A8 Workspace.
func TestFlatBinaryI8_poolIsInert(t *testing.T) {
	const dim, k = 128, 10
	vecs := unitVecs(6000, dim, 3804)
	fb := NewFlatBinaryI8(vecs)
	q := unitVecs(1, dim, 555)[0]
	want := fb.Query(q, k)

	// Run an unrelated query to leave real state in the pool, then poison it.
	_ = fb.Query(unitVecs(1, dim, 556)[0], k)
	sc := binScratchI8Pool.Get().(*binScratchI8)
	for i := range sc.dists {
		sc.dists[i] = 0
	}
	for i := range sc.merged {
		sc.merged[i].Item, sc.merged[i].Score = -1, math.Inf(1)
	}
	for i := range sc.ids {
		sc.ids[i] = -1
	}
	for i := range sc.gatherCodes {
		sc.gatherCodes[i] = 0
	}
	for i := range sc.gatherScales {
		sc.gatherScales[i] = float32(math.NaN())
	}
	for i := range sc.dst {
		sc.dst[i] = float32(math.NaN())
	}
	binScratchI8Pool.Put(sc)

	got := fb.Query(q, k)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("hit %d after a poisoned pool: %+v, want %+v", i, got[i], want[i])
		}
	}
}

// TestFlatBinaryI8_degenerateInputs mirrors TestFlatBinary_degenerateInputs.
func TestFlatBinaryI8_degenerateInputs(t *testing.T) {
	if got := NewFlatBinaryI8(nil).Query([]float32{1, 0}, 5); got != nil {
		t.Errorf("empty index returned %v", got)
	}
	if got := NewFlatBinaryI8([][]float32{}).Len(); got != 0 {
		t.Errorf("Len on an empty index = %d", got)
	}
	vecs := unitVecs(100, 32, 3805)
	fb := NewFlatBinaryI8(vecs)
	if got := fb.Query(make([]float32, 31), 5); got != nil {
		t.Errorf("wrong-dimension query returned %d hits, want nil", len(got))
	}
	// A single vector: cand >= n on the first try, so this is the exact path.
	one := NewFlatBinaryI8(vecs[:1])
	if got := one.Query(vecs[0], 1); len(got) != 1 || got[0].Index != 0 {
		t.Errorf("single-vector index returned %+v", got)
	}
}

// TestFlatBinaryI8_prefilterMatchesFlatBinary is the extraction's own
// regression gate: FlatBinary and FlatBinaryI8 build the SAME binaryPrefilter
// from the same vectors, so for the same query and overquery they must select
// the EXACT SAME candidate set — the two types differ only in what happens to
// that set afterward. This is the test that would catch the shared-prefilter
// refactor silently diverging into two prefilters instead of one.
func TestFlatBinaryI8_prefilterMatchesFlatBinary(t *testing.T) {
	const dim, k, over = 128, 10, 8
	vecs, qs := binClusteredCorpus(150, 80, dim, 20, 0.5, 3900)
	fbFloat := NewFlatBinaryOverquery(vecs, over)
	fbI8 := NewFlatBinaryI8Overquery(vecs, over)

	for qi, q := range qs {
		qc := make([]uint64, fbFloat.pf.words)
		qf := make([]float32, fbFloat.pf.dim)
		bitvec.PackSignBitsRow(qc, fbFloat.pf.centered(qf, q))

		sc1 := &binPrefilterScratch{}
		ids1 := fbFloat.pf.prefilter(qc, sc1, k*over, nil)
		sc2 := &binPrefilterScratch{}
		ids2 := fbI8.pf.prefilter(qc, sc2, k*over, nil)

		if len(ids1) != len(ids2) {
			t.Fatalf("query %d: FlatBinary prefilter picked %d candidates, FlatBinaryI8 picked %d", qi, len(ids1), len(ids2))
		}
		for i := range ids1 {
			if ids1[i] != ids2[i] {
				t.Fatalf("query %d: candidate %d differs: FlatBinary %d, FlatBinaryI8 %d", qi, i, ids1[i], ids2[i])
			}
		}
	}
}
