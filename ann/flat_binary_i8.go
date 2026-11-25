package ann

import (
	"slices"
	"sync"

	"github.com/townsendmerino/aikit/linalg"

	"github.com/townsendmerino/aikit/linalg/bitvec"
	"github.com/townsendmerino/aikit/topk"
)

// FlatBinaryI8 is FlatBinary with FlatI8 standing in for the rerank stage
// instead of Flat: the same 1-bit sign-quantized Hamming PREFILTER, followed by
// an int8 W8A8 rerank of the survivors rather than an exact float32 one. It
// exposes the same Hit / Query(q, k) shape as Flat, FlatI8, and FlatBinary.
//
// What it buys over FlatBinary. FlatBinary's own doc explains it composes with
// embed.Truncate (shrinking both stages) but keeps the full float32 vectors
// for its rerank — dim/8 bytes of codes ON TOP of 4·dim bytes of vectors. This
// type composes the SAME prefilter with FlatI8's quantized storage instead:
// dim/8 + dim bytes total, a further ~3.9× on the rerank-stage memory (and,
// compounding with FlatBinary's own already-measured 13–26× throughput gain
// over FlatI8's full scan, the same win on the query side). The cost is
// FlatI8's own recall hit from int8 quantization, layered on top of the
// prefilter's approximation — see FlatI8's package doc for that tradeoff in
// isolation, and TestFlatBinaryI8_recallReal_Model2Vec for the two combined.
//
// What it does NOT change. The prefilter — centering, packing, histogram vs.
// heap selection, the parallel-sharding threshold — is the literal same code
// FlatBinary runs (binaryPrefilter); nothing here has its own copy to drift.
//
// Like FlatBinary, a built *FlatBinaryI8 is read-only and Query is safe to
// call concurrently; the constructors are the single-threaded builders.
type FlatBinaryI8 struct {
	pf binaryPrefilter
	// exact backs the rerank and the k<=0 / k>=Len passthrough — FlatI8's own
	// int8 storage and W8A8 scan, not a second copy of it. Named to match
	// FlatBinary's `exact *Flat` field; here "exact" means exact int8 scoring
	// (FlatI8's own contract), not exact float32.
	exact *FlatI8
}

// NewFlatBinaryI8 builds a binary-prefiltered, int8-reranked index over vecs
// (each assumed L2-normalized, the package invariant) with DefaultOverquery.
func NewFlatBinaryI8(vecs [][]float32) *FlatBinaryI8 {
	return newFlatBinaryI8(vecs, DefaultOverquery, true)
}

// NewFlatBinaryI8Overquery is NewFlatBinaryI8 with an explicit candidate
// multiplier — see NewFlatBinaryOverquery; the same guidance applies here.
func NewFlatBinaryI8Overquery(vecs [][]float32, overquery int) *FlatBinaryI8 {
	return newFlatBinaryI8(vecs, overquery, true)
}

func newFlatBinaryI8(vecs [][]float32, overquery int, center bool) *FlatBinaryI8 {
	return &FlatBinaryI8{
		pf:    newBinaryPrefilter(vecs, overquery, center),
		exact: NewFlatI8(vecs),
	}
}

// Len is the number of indexed vectors.
func (f *FlatBinaryI8) Len() int { return f.pf.n }

// Overquery is the candidate multiplier this index was built with.
func (f *FlatBinaryI8) Overquery() int { return f.pf.over }

// Query returns approximately the k highest int8-cosine-similarity vectors to
// q, descending, ties broken by ascending index. k <= 0 or k >= Len returns
// all, sorted — and is FlatI8's own exact int8 scan, since there is nothing
// for a prefilter to discard.
//
// The scores are FlatI8's int8 W8A8 scores, so a hit's Score means the same
// thing it does for FlatI8. Which hits come back can differ: see the type's
// doc.
func (f *FlatBinaryI8) Query(q []float32, k int) []Hit {
	return f.query(q, k, nil)
}

// QueryFilter is Query restricted to documents for which keep(id) is true. A
// nil keep is exactly Query. Same semantics as FlatBinary.QueryFilter: keep is
// applied in the PREFILTER, not after it.
func (f *FlatBinaryI8) QueryFilter(q []float32, k int, keep func(id int) bool) []Hit {
	return f.query(q, k, keep)
}

func (f *FlatBinaryI8) query(q []float32, k int, keep func(int) bool) []Hit {
	if f.pf.n == 0 || len(q) != f.pf.dim {
		return nil
	}
	cand := k * f.pf.over
	// Two ways the prefilter has nothing to do: the caller wants everything, or
	// the candidate set is the whole corpus. Both defer to FlatI8's own full
	// scan, which is faster than prefiltering and then reranking all of it —
	// same reasoning as FlatBinary.query.
	if k <= 0 || k >= f.pf.n || cand >= f.pf.n || cand < 0 /* overflow */ {
		return f.exact.query(q, k, keep)
	}

	sc := binScratchI8Pool.Get().(*binScratchI8)
	defer binScratchI8Pool.Put(sc)
	sc.qc = ensure(sc.qc, f.pf.words)
	sc.qf = ensure(sc.qf, f.pf.dim)
	bitvec.PackSignBitsRow(sc.qc, f.pf.centered(sc.qf, q))

	ids := f.pf.prefilter(sc.qc, &sc.binPrefilterScratch, cand, keep)
	return f.rerank(sc, q, ids, k)
}

// rerank scores the candidates with FlatI8's own W8A8 kernel and returns the k
// best.
//
// It gathers the candidates' int8 codes and scales into a contiguous buffer
// and runs ONE MatmulBTW8A8Into call over them — the same kernel FlatI8.Query
// uses for its full-corpus scan (M=1: one query against N=len(ids) stored
// rows) — rather than a DotI8 per candidate, so a document's score here is
// computed the way FlatI8 computes it. Mirrors FlatBinary.rerank's gather-
// then-batch-score shape exactly, one storage precision down.
//
// The gather itself is not worth optimizing, same reasoning as FlatBinary's:
// at k=10, overquery 8 the rerank is 80 candidates against a prefilter that
// just read the whole corpus.
func (f *FlatBinaryI8) rerank(sc *binScratchI8, q []float32, ids []int32, k int) []Hit {
	dim := f.pf.dim
	codes := sc.gatherCodes[:0]
	scales := sc.gatherScales[:0]
	for _, id := range ids {
		row := f.exact.bq[int(id)*dim : int(id)*dim+dim]
		codes = append(codes, row...)
		scales = append(scales, f.exact.scales[id])
	}
	sc.gatherCodes, sc.gatherScales = codes, scales

	dst := ensure(sc.dst, len(ids))
	scoreW8A8(&sc.wm, &sc.ws, q, codes, scales, dst, 1, dim, len(ids))
	sc.dst = dst

	sel := topk.New[int](k)
	th := sel.Threshold()
	for i, score := range dst {
		s := float64(score)
		if s > th {
			sel.Push(int(ids[i]), s)
			th = sel.Threshold()
		}
	}
	items := sel.Result()
	slices.SortFunc(items, topk.ItemCmp[int])
	hits := make([]Hit, len(items))
	for j, s := range items {
		hits[j] = Hit{Index: s.Item, Score: s.Score}
	}
	return hits
}

// binScratchI8 is one query's reusable buffers: the shared prefilter scratch
// plus FlatBinaryI8's own int8-rerank gather buffers and W8A8 Workspace.
type binScratchI8 struct {
	binPrefilterScratch
	gatherCodes  []int8
	gatherScales []float32
	ws           linalg.Workspace
	wm           linalg.WeightMat // scoreW8A8's view of the gathered rows
	dst          []float32
}

var binScratchI8Pool = sync.Pool{New: func() any { return new(binScratchI8) }}
