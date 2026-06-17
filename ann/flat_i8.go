package ann

import (
	"sort"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/mmap"
	"github.com/townsendmerino/aikit/topk"
)

// FlatI8 is the int8-quantized sibling of Flat: it stores each indexed vector as
// int8 codes plus a per-vector float32 scale — a quarter of Flat's float32
// footprint — and scores a query by int8×int8 dot product. It exposes the same
// Hit / Query(q, k) shape as Flat, so it drops into the same fuse.RRF flow and is
// a swap-in where memory matters more than the last fraction of recall.
//
// The win is exactly aikit's niche: embedded, single-binary, RAM-constrained
// retrieval (and a 4× smaller blob to //go:embed). The cost is a small recall hit
// from quantization — tiny for the L2-normalized embeddings this index is built
// for, since every component lands in a bounded range and gets ~1/127 resolution.
//
// Scoring reuses linalg's W8A8 kernel: the query is dynamically quantized to int8,
// dotted against the stored int8 vectors in the integer domain (SIMD, parallel
// over the corpus), and rescaled by the query and per-vector scales. So this is
// W8A8 at M=1 — the same math the quantized decode path uses.
//
// Like Flat, a built *FlatI8 is read-only and Query is safe to call concurrently;
// NewFlatI8 is the single-threaded builder.
type FlatI8 struct {
	bq     []int8    // [n*dim] row-major int8 vectors
	scales []float32 // [n] per-vector reconstruction scales
	n, dim int

	// mmap is the backing mapping when built by LoadFlatI8Mmap (bq aliases it);
	// nil for an in-memory index (NewFlatI8 / LoadFlatI8). closed is set by Close.
	mmap   []byte
	closed bool

	// pager is non-nil only for a LoadFlatI8MmapPaged index: it bounds resident code
	// bytes to a budget by paging fixed-size blocks of rows in and out of the mapping
	// (idea lifted from goinfer's expert pager). nil ⇒ the whole code block stays
	// resident, the default path with full query parallelism. blockRows is the paging
	// granularity. pagerMu guards the pager, which is stateful (an LRU it mutates per
	// scan): concurrent Query calls stay correct but serialize through it, so a paged
	// index keeps the RAM cap at the cost of cross-query parallelism.
	pager     *mmap.SpanCache[int]
	blockRows int
	pagerMu   sync.Mutex
}

// NewFlatI8 builds an int8 index by quantizing vecs (each assumed L2-normalized,
// the package invariant) to int8 + per-vector scales. The float32 inputs are not
// retained — only the int8 codes and scales — so the index holds ~¼ the bytes.
// All vectors are treated as dimension len(vecs[0]); a shorter vector is
// zero-padded and a longer one truncated (the embed pipeline yields a uniform
// dimension, so this is just defensive).
func NewFlatI8(vecs [][]float32) *FlatI8 {
	n := len(vecs)
	d := 0
	if n > 0 {
		d = len(vecs[0])
	}
	flat := make([]float32, n*d)
	for i, v := range vecs {
		copy(flat[i*d:i*d+d], v)
	}
	bq, scales := linalg.QuantizeRowsInt8(flat, n, d)
	return &FlatI8{bq: bq, scales: scales, n: n, dim: d}
}

// Len is the number of indexed vectors.
func (f *FlatI8) Len() int { return f.n }

// Query returns the k highest int8-cosine vectors to q, descending, ties broken
// by ascending index for determinism — the same contract as Flat.Query. k <= 0 or
// k >= Len returns all, sorted. A query of the wrong dimension returns nil.
func (f *FlatI8) Query(q []float32, k int) []Hit {
	return f.query(q, k, nil)
}

// QueryFilter is Query restricted to documents for which keep(id) is true — a
// logical-delete / live-set filter applied at query time, so the index stays
// immutable. Exact over the int8 scores. A nil keep is exactly Query.
func (f *FlatI8) QueryFilter(q []float32, k int, keep func(id int) bool) []Hit {
	return f.query(q, k, keep)
}

func (f *FlatI8) query(q []float32, k int, keep func(int) bool) []Hit {
	if f.closed {
		// A mmap-backed index whose mapping has been released: querying would read
		// unmapped memory. Fail loudly (programmer error) rather than segfault.
		panic("ann: Query on a closed FlatI8 (mmap released by Close)")
	}
	if f.n == 0 || len(q) != f.dim {
		return nil
	}
	// W8A8 at M=1: dynamically quantize q, int8-dot it against every stored
	// vector, rescale by the query and per-vector scales. SIMD + parallel.
	dst := make([]float32, f.n)
	if f.pager == nil {
		linalg.MatmulBTW8A8(q, f.bq, f.scales, dst, 1, f.dim, f.n)
	} else {
		f.scorePaged(q, dst)
	}

	if k <= 0 || k >= f.n {
		hits := make([]Hit, 0, f.n)
		for i, s := range dst {
			if keep == nil || keep(i) {
				hits = append(hits, Hit{Index: i, Score: float64(s)})
			}
		}
		sort.Slice(hits, func(a, b int) bool {
			if hits[a].Score != hits[b].Score {
				return hits[a].Score > hits[b].Score
			}
			return hits[a].Index < hits[b].Index
		})
		return hits
	}

	sel := topk.New[int](k)
	for i, s := range dst {
		if keep == nil || keep(i) {
			sel.Push(i, float64(s))
		}
	}
	items := sel.Result()
	sort.SliceStable(items, func(a, b int) bool {
		if items[a].Score != items[b].Score {
			return items[a].Score > items[b].Score
		}
		return items[a].Item < items[b].Item
	})
	hits := make([]Hit, len(items))
	for j, s := range items {
		hits[j] = Hit{Index: s.Item, Score: s.Score}
	}
	return hits
}
