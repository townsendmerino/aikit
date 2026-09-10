package ann

import (
	"runtime"
	"slices"
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
	// gpu is non-nil after EnableGPU: a device-resident copy of the int8 codes
	// (see backend.go). When set, query scores on the device instead of the CPU
	// W8A8 kernel, falling back to CPU if a device Score errors.
	gpu I8Index
	// gpuShard is non-nil after EnableGPUShardSplit: a device-resident copy of
	// ONLY rows [0, gpuShardRows) — the complement [gpuShardRows, n) scores on
	// the CPU concurrently with it. Independent of gpu/EnableGPU (see backend.go);
	// QueryBatch-only, does not affect Query/QueryFilter.
	gpuShard     I8Index
	gpuShardRows int
	// ws is a persistent W8A8 scratch for scorePaged, reused across the per-block
	// matmuls instead of MatmulBTW8A8 allocating a fresh Workspace per block
	// (~9766 blocks per query on a 10M-vector index — audit #13). Guarded by
	// pagerMu, which scorePaged already holds for the whole scan.
	ws linalg.Workspace
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
	bq := make([]int8, n*d)
	scales := make([]float32, n)
	// One d-sized scratch row, reused, instead of an n*d f32 staging matrix
	// (perf-campaign A6). The old shape copied every input vector into a flat
	// [n,d] float32 block, quantized it, and dropped it: at 378k vectors of
	// dim 256 that block is 387 MB allocated and discarded to produce a 97 MB
	// index. linalg.QuantizeRowsInt8 is a loop over QuantizeRowInt8, and
	// quant.go says so — "exposed so a loader can quantize each row as it is
	// dequantized, without buffering the whole f32 matrix" — so streaming is
	// bit-identical, not merely equivalent.
	//
	// The scratch exists for the ragged-row contract below, not for speed: a
	// short vector must be zero-padded and a long one truncated, and doing that
	// in place would mutate the caller's slice. A row that is already exactly d
	// long skips it and quantizes straight from the input.
	var scratch []float32
	for i, v := range vecs {
		row := v
		if len(v) != d {
			if scratch == nil {
				scratch = make([]float32, d)
			}
			clear(scratch)
			copy(scratch, v)
			row = scratch
		}
		scales[i] = linalg.QuantizeRowInt8(row, bq[i*d:(i+1)*d])
	}
	return &FlatI8{bq: bq, scales: scales, n: n, dim: d}
}

// Len is the number of indexed vectors.
func (f *FlatI8) Len() int { return f.n }

// Query returns the k highest int8-cosine vectors to q, descending, ties broken
// by ascending index for determinism — the same contract as Flat.Query. k <= 0 or
// k >= Len returns all, sorted. A query of the wrong dimension returns nil.
//
// The scan is one W8A8 SIMD matmul (q against every stored vector) and it
// parallelizes across the corpus only above linalg's parallel threshold — a MAC
// count of dim×Len (see linalg.SetParallelThreshold). Below it the scan runs
// SERIAL on the calling goroutine, so a small index, or a batch of many small
// queries you are already fanning out yourself, stays single-threaded here by
// design — don't expect Query to saturate cores until dim×Len clears the threshold.
func (f *FlatI8) Query(q []float32, k int) []Hit {
	return f.query(q, k, nil)
}

// QueryFilter is Query restricted to documents for which keep(id) is true — a
// logical-delete / live-set filter applied at query time, so the index stays
// immutable. Exact over the int8 scores. A nil keep is exactly Query.
// keep must be a pure predicate that is safe for concurrent use, per
// Flat.QueryFilter — that sibling shards its scan and calls keep from several
// goroutines. This implementation applies it serially today; the requirement is
// stated uniformly so one filter works with every index and so this path can
// shard later without a second contract change.
func (f *FlatI8) QueryFilter(q []float32, k int, keep func(id int) bool) []Hit {
	return f.query(q, k, keep)
}

func (f *FlatI8) query(q []float32, k int, keep func(int) bool) []Hit {
	// H6: f.bq aliases f's mmap region (a finalizer munmaps it when f becomes
	// unreachable). The scan below reads f.bq; keep f reachable across it so a
	// mid-query finalize can't unmap the codes under the matmul. Safe today only
	// because f.n is read afterwards — this makes it not depend on that.
	defer runtime.KeepAlive(f)
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
	//
	// Both the score buffer and the kernel's Workspace come from a pool
	// (perf-campaign item 4). The in-memory fast path used to allocate a fresh
	// []float32 of f.n per query AND go through the allocating MatmulBTW8A8
	// wrapper, which makes a zero Workspace — so int8Buf(K) + f32Buf(1) were
	// fresh every call too. The PAGED path had already fixed exactly this
	// (scorePaged, with a comment naming the problem); the common path was left
	// behind. It cannot simply reuse f.ws the way scorePaged does, because that
	// is guarded by pagerMu and Query has no lock — hence a pool.
	//
	// Reuse is safe because the kernel ASSIGNS every element of dst[0:n]
	// (w8a8Span writes dst[i,j], it does not accumulate), as do the GPU and
	// paged paths; TestFlatI8_pooledScratchIsInert poisons the pool to check it.
	sc := flatI8ScratchPool.Get().(*flatI8Scratch)
	defer flatI8ScratchPool.Put(sc)
	// The pooled Workspace is a zero value, so left alone it inherits the
	// process-wide parThreshold (1<<24 MACs) — that constant is goinfer's DECODE
	// tuning, chosen so tiny per-token projections stay serial, and it is 32x
	// Flat's flatParallelThreshold (1<<19), which was measured on THIS shape.
	// The effect was that the int8 scan ran single-core up to ~65k vectors at
	// dim 256 while the f32 scan sharded from ~2k (audit M-19). The kernel's
	// MAC count here is M*N*K = 1*n*dim, directly comparable to Flat's N*dim.
	//
	// Width-inert: SetThreshold changes only the fan-out, and w8a8Span is
	// bit-identical across any column split (TestW8A8Batch_widthInert).
	sc.ws.SetThreshold(flatParallelThreshold)
	if cap(sc.dst) < f.n {
		sc.dst = make([]float32, f.n)
	}
	dst := sc.dst[:f.n]
	switch {
	case f.gpu != nil:
		// Device-resident scoring; fall back to the CPU kernel if the device errors
		// (the scores are identical up to the int8 tolerance, so the fallback is
		// transparent to the ranking).
		//
		// Device SELECTION first, when the backend offers it and there is no filter:
		// scoring on the device and selecting on the host means copying all n scores
		// back and scanning them here, which on CUDA at n=200k measured 0.18 ms of
		// readback plus 0.15 ms of host top-k against 0.40 ms of kernel — 43% of the
		// call spent moving and re-reading what the device already had. TopKBatch
		// returns k hits instead of n scores.
		//
		// A keep filter excludes this: the device selects before any filtering, so it
		// could return k hits that the filter then empties. That path needs all n
		// scores and says so by falling through.
		if ti, ok := f.gpu.(I8TopKIndex); ok && keep == nil && k > 0 && k < f.n {
			if hits, err := ti.TopKBatch([][]float32{q}, k); err == nil && len(hits) == 1 {
				return hits[0]
			}
		}
		if err := f.gpu.Score(q, dst); err != nil {
			linalg.MatmulBTW8A8Into(&sc.ws, q, f.bq, f.scales, dst, 1, f.dim, f.n)
		}
	case f.pager == nil:
		linalg.MatmulBTW8A8Into(&sc.ws, q, f.bq, f.scales, dst, 1, f.dim, f.n)
	default:
		f.scorePaged(q, dst)
	}

	return f.topHits(dst, k, keep)
}

// topHits selects the k highest scores from dst (one score per indexed vector,
// or per shard row when dst is a shard-sized sub-score, not always f.n — the
// caller offsets returned indices when dst represents a sub-range), descending,
// ties broken by ascending index, honoring an optional keep filter. k <= 0 or
// k >= len(dst) returns all, sorted. Shared by Query, QueryBatch, and the
// CPU∥GPU shard-split path.
func (f *FlatI8) topHits(dst []float32, k int, keep func(int) bool) []Hit {
	if k <= 0 || k >= len(dst) {
		hits := make([]Hit, 0, f.n)
		for i, s := range dst {
			if keep == nil || keep(i) {
				hits = append(hits, Hit{Index: i, Score: float64(s)})
			}
		}
		slices.SortFunc(hits, hitCmp)
		return hits
	}

	sel := topk.New[int](k)
	// Hoist the retention threshold out of the loop: Push cannot be inlined (it
	// calls siftUp/siftDown), so every rejected candidate otherwise pays a full call
	// to fail one comparison. Once the heap is full, almost every candidate in a
	// large scan is rejected. `>` matches Push's own strict test, so this changes
	// which items are kept: not at all.
	th := sel.Threshold()
	for i, s := range dst {
		if float64(s) > th && (keep == nil || keep(i)) {
			sel.Push(i, float64(s))
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

// flatI8Scratch is the per-query scratch a FlatI8 scan needs: the score buffer
// and the W8A8 kernel's quantization Workspace. Pooled rather than per-query —
// see the note in query (item 4).
type flatI8Scratch struct {
	ws  linalg.Workspace
	dst []float32
}

var flatI8ScratchPool = sync.Pool{New: func() any { return new(flatI8Scratch) }}
