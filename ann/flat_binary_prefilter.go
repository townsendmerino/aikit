package ann

import (
	"runtime"
	"slices"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/topk"
)

// binaryPrefilter is the SimHash/sign-bit prefilter FlatBinary and FlatBinaryI8
// both build on: pack each vector's (v - center) sign bits into codes, then rank
// candidates by Hamming distance to the query's packed code. Extracted because
// the prefilter — centering, packing, the histogram/heap dispatch, the
// parallel-sharding threshold, all of it — is IDENTICAL between the two
// siblings; the only thing that differs is what backs the rerank stage after it
// (float32 exact for FlatBinary, int8-quantized for FlatBinaryI8). Splitting it
// out once means a future prefilter change (a new selection strategy, a
// different centering rule) is a one-place edit for both, not two kept in sync
// by hand.
//
// See FlatBinary's package doc for what this buys and what it does not.
type binaryPrefilter struct {
	codes []uint64 // [n*words] packed sign bits of (vec - center)
	// center is the corpus mean, subtracted before the sign is taken. See
	// FlatBinary's package doc on the codes field for why (measured, not
	// assumed: TestFlatBinary_centeringIsWhatMakesRealCorporaWork).
	center []float32

	n, dim, words int
	over          int
}

// DefaultOverquery is the prefilter's candidate multiplier: Query(q, k) reranks
// the DefaultOverquery·k nearest by Hamming distance. Higher trades scan time
// for recall.
//
// 16 is where the recall curve reaches 1.0 — on the real Model2Vec corpus
// (0.5625 / 0.7250 / 0.8375 / 0.9625 / 1.0000 at overquery 1 / 2 / 4 / 8 / 16,
// logged by TestFlatBinary_recallReal_Model2Vec) and on the synthetic clusters
// alike. It is set there rather than at 8 because the prefilter's counting-sort
// selection is FLAT in this knob: at d=768, N=200k, k=10 the query costs 877 µs
// at overquery 4 and 901 µs at 16, so the last 4 points of recall are worth
// about 3% of the query. That was not true of the first implementation — see
// prefilterHist — and it is the measurement to redo before changing this.
//
// It is a starting point, not a universal constant. A corpus with tighter
// clusters wants more; measure on your own data with the same recall-versus-
// overquery sweep those two tests run.
const DefaultOverquery = 16

// newBinaryPrefilter builds the prefilter over vecs (each assumed
// L2-normalized, the package invariant). center is a parameter rather than
// always true so the recall test can measure what centering is worth instead
// of asserting it; there is no reason for a caller to turn it off.
func newBinaryPrefilter(vecs [][]float32, overquery int, center bool) binaryPrefilter {
	pf := binaryPrefilter{n: len(vecs), over: max(overquery, 1)}
	if pf.n == 0 {
		return pf
	}
	pf.dim = len(vecs[0])
	pf.words = linalg.PackedWords(pf.dim)
	if center {
		// float64 accumulation: at n in the millions a float32 running sum loses
		// the low bits of the mean, which is exactly the quantity the sign test
		// is sensitive to.
		acc := make([]float64, pf.dim)
		used := 0
		for _, v := range vecs {
			if len(v) != pf.dim {
				continue // ragged input; the code for it is left all-zero below
			}
			used++
			for i, x := range v {
				acc[i] += float64(x)
			}
		}
		if used > 0 {
			pf.center = make([]float32, pf.dim)
			for i, s := range acc {
				pf.center[i] = float32(s / float64(used))
			}
		}
	}

	pf.codes = make([]uint64, pf.n*pf.words)
	row := make([]float32, pf.dim)
	for i, v := range vecs {
		if len(v) != pf.dim {
			continue
		}
		linalg.PackSignBitsRow(pf.codes[i*pf.words:(i+1)*pf.words], pf.centered(row, v))
	}
	return pf
}

// centered writes v - center into dst and returns it (or v itself when the
// prefilter is uncentered, avoiding the copy).
func (pf *binaryPrefilter) centered(dst, v []float32) []float32 {
	if pf.center == nil {
		return v
	}
	for i, x := range v {
		dst[i] = x - pf.center[i]
	}
	return dst
}

// prefilter returns the cand nearest ids by Hamming distance, ascending by id.
//
// Ascending is not incidental: the caller's rerank pushes in that order, and
// topk's first-seen-wins tie rule then reproduces Flat's "ties broken by
// ascending index" exactly. Handing the rerank a distance-ordered list would
// break that tie-break for documents with equal scores.
//
// Two implementations, and which one runs depends only on whether there is a
// filter. See prefilterHist for why the unfiltered case gets its own.
func (pf *binaryPrefilter) prefilter(qc []uint64, sc *binPrefilterScratch, cand int, keep func(int) bool) []int32 {
	if keep == nil {
		return pf.prefilterHist(qc, sc, cand)
	}
	return pf.prefilterHeap(qc, sc, cand, keep)
}

// prefilterHist selects the cand nearest by COUNTING SORT over the distance
// histogram, which is available here and almost nowhere else: a Hamming
// distance is an integer in [0, dim], so dim+1 counters describe the whole
// corpus exactly.
//
// It replaced a per-shard top-cand heap, and the reason was measured. The heap
// made the candidate multiplier expensive — at d=768, N=200k, k=10 a query cost
// 594 µs at overquery 4 and 2042 µs at overquery 32, a 3.4× swing on a knob
// that only controls RECALL. The scan does not change at all across that sweep;
// what changed was 16 shards each maintaining, allocating and merging a
// cand-sized heap. The histogram is O(n) with no per-shard structure
// proportional to cand, so overquery costs only the extra exact reranks it
// actually buys — which is what let DefaultOverquery be set where the recall
// curve flattens instead of where the heap stopped hurting.
//
// The result is exactly the same candidate set the heap produced: all documents
// strictly nearer than the threshold distance, then documents AT it in
// ascending id order until the budget is spent — which is "(distance asc, id
// asc), take cand" written out.
func (pf *binaryPrefilter) prefilterHist(qc []uint64, sc *binPrefilterScratch, cand int) []int32 {
	nb := pf.dim + 1
	workers := binQueryWorkers(pf.n, pf.words)
	sc.dists = ensure(sc.dists, pf.n)
	sc.hist = ensure(sc.hist, workers*nb)
	clear(sc.hist)

	if workers <= 1 {
		pf.scanHist(qc, sc.dists, sc.hist[:nb], 0, pf.n)
	} else {
		per := (pf.n + workers - 1) / workers
		var wg sync.WaitGroup
		for w := range workers {
			lo := w * per
			if lo >= pf.n {
				break
			}
			hi := min(lo+per, pf.n)
			wg.Add(1)
			// Each worker owns a disjoint span of dists AND its own histogram
			// slice, so there is nothing to synchronize and nothing to allocate
			// per worker.
			go func(w, lo, hi int) {
				defer wg.Done()
				pf.scanHist(qc, sc.dists, sc.hist[w*nb:(w+1)*nb], lo, hi)
			}(w, lo, hi)
		}
		wg.Wait()
		for w := 1; w < workers; w++ {
			for b := range nb {
				sc.hist[b] += sc.hist[w*nb+b]
			}
		}
	}

	// The threshold distance: the smallest t whose cumulative count reaches
	// cand. Everything nearer is taken outright; t itself is taken partially.
	cum, t := 0, 0
	for ; t < nb; t++ {
		if cum+int(sc.hist[t]) >= cand {
			break
		}
		cum += int(sc.hist[t])
	}
	budget := cand - cum

	sc.ids = ensure(sc.ids, cand)
	ids := sc.ids
	k := 0
	tU16 := uint16(t)
	for i, d := range sc.dists[:pf.n] {
		if d < tU16 {
			ids[k] = int32(i)
			k++
		} else if d == tU16 && budget > 0 {
			ids[k] = int32(i)
			k++
			budget--
		}
	}
	ids = ids[:k]
	sc.ids = ids
	return ids
}

// scanHist fills dists[lo:hi] and tallies hist, a block at a time so the
// distances are still in L1 when the histogram reads them back.
func (pf *binaryPrefilter) scanHist(qc []uint64, dists []uint16, hist []int32, lo, hi int) {
	for base := lo; base < hi; base += binBlockRows {
		end := min(base+binBlockRows, hi)
		blk := dists[base:end]
		linalg.HammingRows(qc, pf.codes[base*pf.words:end*pf.words], pf.words, end-base, blk)
		for _, d := range blk {
			hist[d]++
		}
	}
}

// prefilterHeap is the filtered path: a per-shard top-cand selector, which
// calls keep only for candidates that could actually be retained.
//
// The histogram cannot serve this case without either calling keep for every
// document in the corpus — turning a cheap live-set lookup into n
// non-inlinable calls — or iterating thresholds upward until enough survivors
// appear. The heap already has the property that matters (keep is consulted
// O(cand·log n) times, not O(n)), so the filtered path keeps it and the two are
// gated against each other by TestFlatBinary_histAndHeapAgree.
func (pf *binaryPrefilter) prefilterHeap(qc []uint64, sc *binPrefilterScratch, cand int, keep func(int) bool) []int32 {
	workers := binQueryWorkers(pf.n, pf.words)
	if workers <= 1 {
		sc.parts = append(sc.parts[:0], nil)
		pf.scanShard(qc, sc, 0, pf.n, cand, keep, &sc.parts[0])
		return finishCandidates(sc.parts, cand, sc)
	}

	per := (pf.n + workers - 1) / workers
	parts := make([][]topk.ItemWithScore[int32], workers)
	var wg sync.WaitGroup
	for w := range workers {
		lo := w * per
		if lo >= pf.n {
			break
		}
		hi := min(lo+per, pf.n)
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			// Each shard needs its OWN block buffer, so it takes its own scratch
			// from the pool. The packed query is passed by value rather than
			// stored on that scratch: a borrowed scratch goes back to the pool at
			// return, and stashing the parent's buffer in it would leave two
			// scratches sharing one array on some later query.
			s := binPrefilterScratchPool.Get().(*binPrefilterScratch)
			defer binPrefilterScratchPool.Put(s)
			pf.scanShard(qc, s, lo, hi, cand, keep, &parts[w])
		}(w, lo, hi)
	}
	wg.Wait()
	// Merging per-shard top-cand lists is exact under the same argument
	// Flat.queryShards makes: any element of the global top-cand under
	// (distance asc, id asc) is also in its own shard's top-cand, since a shard
	// is a subset and the order is total.
	return finishCandidates(parts, cand, sc)
}

// scanShard computes Hamming distances for rows [lo, hi) a block at a time and
// keeps the cand nearest.
//
// Blocked rather than one big distance array because the block buffer then
// stays in L1 between the kernel writing it and the selection loop reading it,
// and because a whole-corpus []uint16 would be a 2 MB per-query allocation at
// n = 10⁶ for no benefit.
func (pf *binaryPrefilter) scanShard(qc []uint64, sc *binPrefilterScratch, lo, hi, cand int, keep func(int) bool, out *[]topk.ItemWithScore[int32]) {
	sel := topk.New[int32](cand)
	// Score is NEGATED distance: topk keeps the highest, and nearest means
	// smallest. Distances are small integers exactly representable in float64,
	// so this is a relabeling, not an approximation.
	th := sel.Threshold()
	sc.dists = ensure(sc.dists, binBlockRows)
	for base := lo; base < hi; base += binBlockRows {
		end := min(base+binBlockRows, hi)
		rows := end - base
		linalg.HammingRows(qc, pf.codes[base*pf.words:end*pf.words], pf.words, rows, sc.dists[:rows])
		for j, d := range sc.dists[:rows] {
			s := -float64(d)
			if s > th && (keep == nil || keep(base+j)) {
				sel.Push(int32(base+j), s)
				th = sel.Threshold()
			}
		}
	}
	*out = sel.Result()
}

// finishCandidates merges the shards' selections, truncates to cand under
// (distance asc, id asc), and returns the ids in ASCENDING ID order.
func finishCandidates(parts [][]topk.ItemWithScore[int32], cand int, sc *binPrefilterScratch) []int32 {
	merged := sc.merged[:0]
	for _, p := range parts {
		merged = append(merged, p...)
	}
	sc.merged = merged
	slices.SortFunc(merged, topk.ItemCmp[int32])
	if len(merged) > cand {
		merged = merged[:cand]
	}
	ids := sc.ids[:0]
	for _, m := range merged {
		ids = append(ids, m.Item)
	}
	sc.ids = ids
	slices.Sort(ids)
	return ids
}

// binBlockRows is the Hamming scan's blocking factor: 1024 rows of uint16 is a
// 2 KB distance buffer, comfortably L1-resident alongside the codes being
// streamed through it.
const binBlockRows = 1024

// binParallelThreshold is the scanned-word count at or above which sharding the
// prefilter pays for the goroutine spawn.
//
// It is NOT flatParallelThreshold converted to words. That constant fans out
// when the exact scan reaches roughly 80 µs on this box; matching that TIME —
// which is what the goroutine spawn has to be amortized against — takes ~14×
// more vectors here, because that is how much cheaper this scan is per vector.
// Time-matched, not swept, and amd64-measured.
const binParallelThreshold = 1 << 17

func binQueryWorkers(n, words int) int {
	if n < 2 || int64(n)*int64(words) < binParallelThreshold {
		return 1
	}
	return min(runtime.NumCPU(), n)
}

// binPrefilterScratch is one query's reusable buffers for binaryPrefilter's
// scan — shared by FlatBinary and FlatBinaryI8 since the prefilter stage is
// identical for both; only the rerank stage (each type's own scratch, which
// embeds this) differs. Pooled for the same reason flatI8Scratch is: the
// per-query allocation is otherwise proportional to the corpus, and a
// retrieval path is called in a loop.
type binPrefilterScratch struct {
	qc     []uint64
	qf     []float32
	dists  []uint16
	hist   []int32
	merged []topk.ItemWithScore[int32]
	ids    []int32
	parts  [][]topk.ItemWithScore[int32]
}

var binPrefilterScratchPool = sync.Pool{New: func() any { return new(binPrefilterScratch) }}

// ensure returns b resized to length n, reusing its existing backing array
// when it has enough capacity and allocating fresh otherwise — the scratch-
// buffer growth pattern used by every scoring-cache field in this file
// (qc/qf/dists/hist), previously reimplemented once per element type.
func ensure[T any](b []T, n int) []T {
	if cap(b) >= n {
		return b[:n]
	}
	return make([]T, n)
}
