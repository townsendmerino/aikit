package ann

import (
	"github.com/townsendmerino/aikit/linalg/bitvec"
	"github.com/townsendmerino/aikit/topk"
	"slices"
	"sync"
)

// FlatBinary is Flat with a two-stage scan: a 1-bit sign-quantized Hamming
// PREFILTER over the whole corpus, then an EXACT float32 rerank of the
// survivors. It exposes the same Hit / Query(q, k) shape as Flat and FlatI8.
//
// What it trades. This is the first index in the package whose result set is
// APPROXIMATE — Flat and FlatI8 both score every vector and are exact over
// their respective score functions. Here the true top-k can be missed outright
// if the prefilter does not rank it in the top Overquery·k by Hamming distance.
// The scores that come back are exact (same kernel Flat uses); the membership
// is not.
//
// What it buys. Only the SCAN shrinks, not the index: the float32 vectors are
// still held for the rerank, and the codes add dim/8 bytes per vector on top —
// about 3% at dim 768. If it is memory you need, FlatI8 is the sibling that
// gives you a quarter of the bytes — or FlatBinaryI8, which composes the same
// binary prefilter with FlatI8's int8 rerank instead of this type's float32
// one, for dim/8 + dim bytes total rather than this type's dim/8 + 4·dim. What
// this gives you is a first stage that reads 96 B per candidate instead of
// 3072 B, measured at 14× the throughput of the exact scan on this corpus
// shape (perf-campaign item 38).
//
// It composes with embed.Truncate rather than competing with it: truncating to
// dim d shrinks both stages, and the prefilter's cost is linear in d.
//
// One shape where it does NOT pay: a very small candidate set. The prefilter
// materializes a distance per document and makes a second pass over them, which
// at overquery·k below roughly 100 on a 200k corpus costs more than the
// heap-based selection the filtered path uses — measured at 877 µs against
// 614 µs for k=10, overquery 4 (BenchmarkFlatBinaryPrefilterPaths). Above that
// the counting sort wins by 1.4× to 6×, and it is flat in overquery where the
// heap is not. The dispatch is on the presence of a filter alone rather than on
// a fitted candidate-count threshold, because that crossover was measured at
// one corpus size on one machine and scales with n.
//
// Like Flat, a built *FlatBinary is read-only and Query is safe to call
// concurrently; the constructors are the single-threaded builders.
type FlatBinary struct {
	pf binaryPrefilter
	// exact backs the rerank and the k<=0 / k>=Len passthrough. Sharing it means
	// the exact stage is literally Flat's scan, not a second copy of it.
	exact *Flat
}

// NewFlatBinary builds a binary-prefiltered index over vecs (each assumed
// L2-normalized, the package invariant) with DefaultOverquery. Vectors are used
// by reference, not copied.
func NewFlatBinary(vecs [][]float32) *FlatBinary {
	return newFlatBinary(vecs, DefaultOverquery, true)
}

// NewFlatBinaryOverquery is NewFlatBinary with an explicit candidate
// multiplier. overquery <= 1 means "rerank exactly k candidates", which makes
// the result the prefilter's own ranking and is almost never what you want;
// raise it until recall@k stops improving on your data.
func NewFlatBinaryOverquery(vecs [][]float32, overquery int) *FlatBinary {
	return newFlatBinary(vecs, overquery, true)
}

// newFlatBinary is the shared builder. center is a parameter rather than
// always true so the recall test can measure what centering is worth instead
// of asserting it; there is no reason for a caller to turn it off.
func newFlatBinary(vecs [][]float32, overquery int, center bool) *FlatBinary {
	return &FlatBinary{
		pf:    newBinaryPrefilter(vecs, overquery, center),
		exact: New(vecs),
	}
}

// Len is the number of indexed vectors.
func (f *FlatBinary) Len() int { return f.pf.n }

// Overquery is the candidate multiplier this index was built with.
func (f *FlatBinary) Overquery() int { return f.pf.over }

// Query returns approximately the k highest cosine-similarity vectors to q,
// descending, ties broken by ascending index. k <= 0 or k >= Len returns all,
// sorted — and is EXACT, since there is nothing for a prefilter to discard.
//
// The scores are exact float32 dot products, so a hit's Score means the same
// thing it does for Flat. Which hits come back can differ: see the type's doc.
func (f *FlatBinary) Query(q []float32, k int) []Hit {
	return f.query(q, k, nil)
}

// QueryFilter is Query restricted to documents for which keep(id) is true. A
// nil keep is exactly Query.
//
// keep is applied in the PREFILTER, not after it, so the k results are the best
// live documents rather than whatever survives filtering a fixed candidate set
// — the difference matters when the filter is selective. As in
// Flat.QueryFilter, keep must be a pure predicate safe for concurrent use: the
// prefilter is sharded, and each shard's running threshold means the set of ids
// it asks about, and their order, differ from a serial scan.
func (f *FlatBinary) QueryFilter(q []float32, k int, keep func(id int) bool) []Hit {
	return f.query(q, k, keep)
}

func (f *FlatBinary) query(q []float32, k int, keep func(int) bool) []Hit {
	if f.pf.n == 0 || len(q) != f.pf.dim {
		return nil
	}
	cand := k * f.pf.over
	// Two ways the prefilter has nothing to do: the caller wants everything, or
	// the candidate set is the whole corpus. Both defer to the exact scan, which
	// is faster than prefiltering and then reranking all of it.
	if k <= 0 || k >= f.pf.n || cand >= f.pf.n || cand < 0 /* overflow */ {
		return f.exact.query(q, k, keep)
	}

	sc := binScratchPool.Get().(*binScratch)
	defer binScratchPool.Put(sc)
	sc.qc = ensure(sc.qc, f.pf.words)
	sc.qf = ensure(sc.qf, f.pf.dim)
	bitvec.PackSignBitsRow(sc.qc, f.pf.centered(sc.qf, q))

	ids := f.pf.prefilter(sc.qc, &sc.binPrefilterScratch, cand, keep)
	return f.rerank(sc, q, ids, k)
}

// rerank scores the candidates exactly and returns the k best.
//
// It gathers the candidate vectors and runs them through scanFlat — the same
// kernel Flat.Query uses — rather than a Dot per candidate, so a document's
// score here is computed the way Flat computes it. (A document that lands in
// the ragged tail of one grouping and the 8-wide body of the other can still
// differ in the last bits; that is the sub-ULP variation Flat's own package doc
// already describes, not a second source of error.)
//
// The gather itself is not worth optimizing: at k=10, overquery 8 the rerank is
// 80 candidates against a prefilter that just read the whole corpus.
func (f *FlatBinary) rerank(sc *binScratch, q []float32, ids []int32, k int) []Hit {
	gath := sc.gather[:0]
	for _, id := range ids {
		gath = append(gath, f.exact.vecs[id])
	}
	sc.gather = gath

	sel := topk.New[int](k)
	th := sel.Threshold()
	scanFlat(q, gath, func(i int, score float64) {
		if score > th {
			sel.Push(int(ids[i]), score)
			th = sel.Threshold()
		}
	})
	items := sel.Result()
	slices.SortFunc(items, topk.ItemCmp[int])
	hits := make([]Hit, len(items))
	for j, s := range items {
		hits[j] = Hit{Index: s.Item, Score: s.Score}
	}
	return hits
}

// binScratch is one query's reusable buffers: the shared prefilter scratch
// plus FlatBinary's own exact-rerank gather buffer.
type binScratch struct {
	binPrefilterScratch
	gather [][]float32
}

var binScratchPool = sync.Pool{New: func() any { return new(binScratch) }}
