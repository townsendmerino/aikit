package ann

import (
	"github.com/townsendmerino/aikit/linalg/bitvec"
	"math"
	"math/rand"
	"testing"
)

// unitVecs builds n L2-normalized vectors — the package invariant every index
// here is built for.
func unitVecs(n, dim int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		var s float64
		for j := range v {
			v[j] = float32(rng.NormFloat64())
			s += float64(v[j]) * float64(v[j])
		}
		inv := float32(1 / math.Sqrt(s))
		for j := range v {
			v[j] *= inv
		}
		out[i] = v
	}
	return out
}

// TestFlatBinary_exactWhenNothingToPrune pins the boundary where FlatBinary
// stops being approximate. If the candidate set is the whole corpus — k <= 0,
// k >= Len, or overquery·k >= Len — a prefilter can discard nothing, so the
// result must be Flat's, hit for hit AND score bit for bit.
//
// This is the half of the contract that IS testable exactly, and it is the half
// that catches a broken tie-break or a mis-scattered candidate id. The
// approximate half needs the recall tests below.
func TestFlatBinary_exactWhenNothingToPrune(t *testing.T) {
	vecs := unitVecs(300, 64, 38)
	q := unitVecs(1, 64, 99)[0]
	flat := New(vecs)

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
			fb := NewFlatBinaryOverquery(vecs, tc.over)
			want := flat.Query(q, tc.k)
			got := fb.Query(q, tc.k)
			if len(got) != len(want) {
				t.Fatalf("got %d hits, Flat returned %d", len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("hit %d: got %+v, Flat returned %+v", i, got[i], want[i])
				}
			}
		})
	}
}

// TestFlatBinary_scoresAndOrderAreExact checks the part of the contract that
// survives approximation: whatever comes back is correctly SCORED and correctly
// ORDERED, even though membership may not be the true top-k. A prefilter bug
// that returns the right ids with the wrong scores attached — an off-by-one in
// the gather, say — passes a pure recall test and fails this one.
func TestFlatBinary_scoresAndOrderAreExact(t *testing.T) {
	vecs := unitVecs(5000, 128, 380)
	fb := NewFlatBinary(vecs)
	for qi := range 20 {
		q := unitVecs(1, 128, int64(1000+qi))[0]
		hits := fb.Query(q, 10)
		if len(hits) != 10 {
			t.Fatalf("query %d: got %d hits, want 10", qi, len(hits))
		}
		for i, h := range hits {
			var want float64
			for j := range q {
				want += float64(q[j]) * float64(vecs[h.Index][j])
			}
			if math.Abs(h.Score-want) > 1e-5 {
				t.Fatalf("query %d hit %d (doc %d): score %v, exact dot %v", qi, i, h.Index, h.Score, want)
			}
			if i > 0 && hits[i-1].Score < h.Score {
				t.Fatalf("query %d: hits out of descending order at %d", qi, i)
			}
		}
	}
}

// binClusteredCorpus builds a corpus with actual neighborhood structure — nc
// cluster centers, per members each a center plus noise of the given spread —
// together with nq queries drawn around THE SAME centers.
//
// Returning both from one function is the point. A first version generated
// queries from a second, independent set of centers, which made every query a
// random direction with no near neighbors in the corpus: recall came out at
// 0.28, indistinguishable from unstructured data, and the corpus's structure
// was entirely wasted. A "clustered" corpus queried at random is not a
// clustered benchmark.
func binClusteredCorpus(nc, per, dim, nq int, spread float64, seed int64) (vecs, qs [][]float32) {
	rng := rand.New(rand.NewSource(seed))
	norm := func(acc []float64) []float32 {
		v := make([]float32, dim)
		var s float64
		for j, x := range acc {
			v[j] = float32(x)
			s += x * x
		}
		inv := float32(1 / math.Sqrt(s))
		for j := range v {
			v[j] *= inv
		}
		return v
	}
	centers := make([][]float64, nc)
	for i := range centers {
		c := make([]float64, dim)
		for j := range c {
			c[j] = rng.NormFloat64()
		}
		centers[i] = c
	}
	acc := make([]float64, dim)
	draw := func(c []float64, sp float64) []float32 {
		for j := range acc {
			acc[j] = c[j] + sp*rng.NormFloat64()
		}
		return norm(acc)
	}
	vecs = make([][]float32, 0, nc*per)
	for _, c := range centers {
		for range per {
			vecs = append(vecs, draw(c, spread))
		}
	}
	qs = make([][]float32, nq)
	for i := range qs {
		// A slightly tighter spread than the members: a query lands inside its
		// cluster, which is what a real near-duplicate lookup looks like.
		qs[i] = draw(centers[i%nc], spread*0.7)
	}
	return vecs, qs
}

// binRecallAt measures recall@k of an approximate index against the exact scan
// over the given queries. Takes queryFilterer rather than *FlatBinary
// specifically so FlatBinaryI8's own recall gates (flat_binary_i8_test.go)
// reuse it too — same measurement, different index under test.
func binRecallAt(flat *Flat, fb queryFilterer, qs [][]float32, k int) float64 {
	hit, total := 0, 0
	for _, q := range qs {
		truth := map[int]bool{}
		for _, h := range flat.Query(q, k) {
			truth[h.Index] = true
		}
		for _, h := range fb.Query(q, k) {
			if truth[h.Index] {
				hit++
			}
			total++
		}
	}
	if total == 0 {
		return 0
	}
	return float64(hit) / float64(total)
}

// TestFlatBinary_recallClustered is the non-skipping recall floor, on a corpus
// that HAS neighbors to find.
//
// The first version of this test used random unit vectors and measured 0.31,
// which reads as a broken prefilter and is not: at dim 256 and n = 20k the
// angular gap between the 10th and 160th nearest of a random corpus is
// ~0.051 rad, while the Hamming estimator's own standard deviation is
// ~0.097 rad. There is nothing there to resolve. An approximate index measured
// on structureless data is measuring its own noise floor.
func TestFlatBinary_recallClustered(t *testing.T) {
	const dim, k = 256, 10
	vecs, qs := binClusteredCorpus(200, 100, dim, 50, 0.5, 3800)
	flat := New(vecs)
	fb := NewFlatBinary(vecs)

	recall := binRecallAt(flat, fb, qs, k)
	t.Logf("recall@%d over %d queries, %d clustered vectors, overquery %d: %.4f",
		k, len(qs), len(vecs), fb.Overquery(), recall)
	if recall < 0.90 {
		t.Errorf("recall@%d = %.4f, want >= 0.90", k, recall)
	}
}

// TestFlatBinary_structureIsWhatMakesItWork keeps the finding above executable.
// The prefilter must do materially better on a corpus with neighborhood
// structure than on one without — that is the claim the whole design rests on,
// and it is the reason the recall gates use clustered data.
//
// It compares the two rather than asserting a number on the unstructured case,
// so an improvement to the prefilter (a random rotation, say) raises the floor
// without turning this red.
func TestFlatBinary_structureIsWhatMakesItWork(t *testing.T) {
	const dim, k, n = 256, 10, 20_000
	clustered, qs := binClusteredCorpus(200, n/200, dim, 30, 0.5, 3806)
	rStruct := binRecallAt(New(clustered), NewFlatBinary(clustered), qs, k)

	unstructured := unitVecs(n, dim, 3806)
	uq := unitVecs(30, dim, 52_000)
	rFlat := binRecallAt(New(unstructured), NewFlatBinary(unstructured), uq, k)

	t.Logf("recall@%d: clustered %.4f, unstructured %.4f", k, rStruct, rFlat)
	if rStruct <= rFlat {
		t.Errorf("recall on clustered data (%.4f) did not beat unstructured (%.4f)", rStruct, rFlat)
	}
}

// TestFlatBinary_overqueryRaisesRecall is the mutation-resistant shape check:
// recall must be MONOTONE in the candidate multiplier, and overquery=1 must be
// visibly worse than the default. A prefilter that ignored its distances (say,
// returned the first cand ids) would give a flat curve, and a rerank that
// ignored the candidate list would give a perfect one.
func TestFlatBinary_overqueryRaisesRecall(t *testing.T) {
	const dim, k = 256, 10
	vecs, qs := binClusteredCorpus(200, 100, dim, 40, 0.5, 3801)
	flat := New(vecs)

	var prev float64
	for _, over := range []int{1, 2, 4, 8, 16} {
		r := binRecallAt(flat, NewFlatBinaryOverquery(vecs, over), qs, k)
		t.Logf("overquery %2d: recall@%d = %.4f", over, k, r)
		if r < prev-1e-9 {
			t.Errorf("recall fell from %.4f to %.4f when overquery rose to %d", prev, r, over)
		}
		prev = r
	}
	if prev < 0.95 {
		t.Errorf("recall@%d at overquery 16 = %.4f, want >= 0.95", k, prev)
	}
}

// TestFlatBinary_filterAppliedInPrefilter checks the documented behaviour that
// keep runs during the prefilter rather than after it. The distinction is
// invisible on a permissive filter and decisive on a selective one: filtering
// AFTER a fixed candidate set would return far fewer than k hits here, since
// only 1 document in 50 is live.
func TestFlatBinary_filterAppliedInPrefilter(t *testing.T) {
	const dim, k = 128, 10
	vecs, qs := binClusteredCorpus(200, 100, dim, 1, 0.5, 3802)
	fb := NewFlatBinary(vecs)
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
	// And it should mostly agree with the exact filtered scan.
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
	if hits < 8 {
		t.Errorf("only %d/%d of the filtered top-%d recovered", hits, k, k)
	}
}

// TestFlatBinary_shardedMatchesSerial gates the parallel prefilter against the
// serial one on the same corpus. Both must return the SAME hits: sharding
// changes how candidates are found, never which.
//
// The corpus is large enough to cross binParallelThreshold and the comparison
// index is built small enough not to — so this is genuinely two code paths,
// which a single-size test would not be.
func TestFlatBinary_shardedMatchesSerial(t *testing.T) {
	const dim, k = 256, 10
	// n·words above binParallelThreshold (1<<17) at dim 256 (words 4) needs
	// n >= 32768.
	vecs := unitVecs(40_000, dim, 3803)
	fb := NewFlatBinary(vecs)
	if binQueryWorkers(fb.pf.n, fb.pf.words) <= 1 {
		t.Fatalf("corpus does not reach the parallel path; this test proves nothing")
	}
	for qi := range 10 {
		q := unitVecs(1, dim, int64(90_000+qi))[0]
		got := fb.Query(q, k)

		// Serial reference: one shard over the whole corpus.
		sc2 := &binScratch{}
		sc2.qc = ensure(sc2.qc, fb.pf.words)
		sc2.qf = ensure(sc2.qf, fb.pf.dim)
		packQueryForTest(fb, sc2, q)
		sc2.parts = append(sc2.parts[:0], nil)
		fb.pf.scanShard(sc2.qc, &sc2.binPrefilterScratch, 0, fb.pf.n, k*fb.pf.over, nil, &sc2.parts[0])
		ids := finishCandidates(sc2.parts, k*fb.pf.over, &sc2.binPrefilterScratch)
		want := fb.rerank(sc2, q, ids, k)

		if len(got) != len(want) {
			t.Fatalf("query %d: sharded returned %d hits, serial %d", qi, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("query %d hit %d: sharded %+v, serial %+v", qi, i, got[i], want[i])
			}
		}
	}
}

func packQueryForTest(f *FlatBinary, sc *binScratch, q []float32) {
	bitvec.PackSignBitsRow(sc.qc, f.pf.centered(sc.qf, q))
}

// TestFlatBinary_poolIsInert poisons every pooled scratch buffer before a query
// and checks the answer is unchanged. The scratch carries a distance block, a
// merged candidate list and a gather buffer, all reused across queries — a
// missing reslice would read a previous query's candidates and be invisible on
// a cold pool.
func TestFlatBinary_poolIsInert(t *testing.T) {
	const dim, k = 128, 10
	vecs := unitVecs(6000, dim, 3804)
	fb := NewFlatBinary(vecs)
	q := unitVecs(1, dim, 555)[0]
	want := fb.Query(q, k)

	// Run an unrelated query to leave real state in the pool, then poison it.
	_ = fb.Query(unitVecs(1, dim, 556)[0], k)
	sc := binScratchPool.Get().(*binScratch)
	for i := range sc.dists {
		sc.dists[i] = 0
	}
	for i := range sc.merged {
		sc.merged[i].Item, sc.merged[i].Score = -1, math.Inf(1)
	}
	for i := range sc.ids {
		sc.ids[i] = -1
	}
	for i := range sc.gather {
		sc.gather[i] = nil
	}
	binScratchPool.Put(sc)

	got := fb.Query(q, k)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("hit %d after a poisoned pool: %+v, want %+v", i, got[i], want[i])
		}
	}
}

// TestFlatBinary_degenerateInputs covers the shapes a retrieval path meets in
// production and never in a benchmark.
func TestFlatBinary_degenerateInputs(t *testing.T) {
	if got := NewFlatBinary(nil).Query([]float32{1, 0}, 5); got != nil {
		t.Errorf("empty index returned %v", got)
	}
	if got := NewFlatBinary([][]float32{}).Len(); got != 0 {
		t.Errorf("Len on an empty index = %d", got)
	}
	vecs := unitVecs(100, 32, 3805)
	fb := NewFlatBinary(vecs)
	if got := fb.Query(make([]float32, 31), 5); got != nil {
		t.Errorf("wrong-dimension query returned %d hits, want nil", len(got))
	}
	// A single vector: cand >= n on the first try, so this is the exact path.
	one := NewFlatBinary(vecs[:1])
	if got := one.Query(vecs[0], 1); len(got) != 1 || got[0].Index != 0 {
		t.Errorf("single-vector index returned %+v", got)
	}
}

// TestFlatBinary_histAndHeapAgree gates the two prefilter implementations
// against each other. The histogram path (unfiltered) and the heap path
// (filtered) must select the SAME candidates, so an always-true filter must
// give exactly the results of no filter at all — hit for hit, score for score.
//
// This is the only thing keeping the fast path honest: it is the one that runs
// for essentially every real query, and without this it would be gated only by
// a recall number that a subtly-wrong candidate set would still satisfy.
func TestFlatBinary_histAndHeapAgree(t *testing.T) {
	alwaysLive := func(int) bool { return true }
	for _, dim := range []int{64, 256} {
		// Two corpus sizes so both the serial and the sharded form of each
		// implementation are exercised.
		for _, n := range []int{3_000, 40_000} {
			vecs, qs := binClusteredCorpus(n/50, 50, dim, 12, 0.5, int64(n+dim))
			for _, over := range []int{2, 8, 32} {
				fb := NewFlatBinaryOverquery(vecs, over)
				for qi, q := range qs {
					for _, k := range []int{1, 10} {
						hist := fb.Query(q, k)
						heap := fb.QueryFilter(q, k, alwaysLive)
						if len(hist) != len(heap) {
							t.Fatalf("d%d n%d over%d q%d k%d: histogram %d hits, heap %d",
								dim, n, over, qi, k, len(hist), len(heap))
						}
						for i := range hist {
							if hist[i] != heap[i] {
								t.Fatalf("d%d n%d over%d q%d k%d hit %d: histogram %+v, heap %+v",
									dim, n, over, qi, k, i, hist[i], heap[i])
							}
						}
					}
				}
			}
		}
	}
}

// TestFlatBinary_histTieBreakIsByID pins the half of the candidate rule that
// only shows up on ties: at the threshold distance, candidates are taken in
// ASCENDING ID order until the budget runs out. Every vector here is identical,
// so every distance is identical and the entire candidate set is decided by the
// tie-break — which makes the top-k exactly ids 0..k-1.
func TestFlatBinary_histTieBreakIsByID(t *testing.T) {
	const dim, n, k = 64, 1000, 10
	v := unitVecs(1, dim, 7)[0]
	vecs := make([][]float32, n)
	for i := range vecs {
		vecs[i] = v
	}
	fb := NewFlatBinaryOverquery(vecs, 4)
	hits := fb.Query(v, k)
	if len(hits) != k {
		t.Fatalf("got %d hits, want %d", len(hits), k)
	}
	for i, h := range hits {
		if h.Index != i {
			t.Fatalf("hit %d is doc %d; an all-ties corpus must return ids 0..%d in order", i, h.Index, k-1)
		}
	}
}
