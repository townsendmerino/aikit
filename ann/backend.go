package ann

import (
	"slices"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
)

// Backend is a device accelerator for FlatI8 scoring. It is registered by an
// aikit/gpu build (native Metal today, CUDA later) via RegisterBackend in an
// init(); the default pure-Go build registers none, and FlatI8 scores on the CPU
// (linalg.MatmulBTW8A8). This is the same inversion the encoder/vision packages
// use for their GPU backends — the core ann package never imports the device
// package, so the default build stays cgo-free and pure-Go.
type Backend interface {
	// NewI8Index makes an int8 index resident on the device: bq is [n*dim]
	// row-major int8 codes, scales is [n] per-row reconstruction scales. The
	// returned handle scores queries against those resident codes.
	NewI8Index(bq []int8, scales []float32, n, dim int) (I8Index, error)
	// Name identifies the backend (e.g. "metal") for diagnostics.
	Name() string
}

// I8Index is a device-resident int8 index. Score writes, for each row j, the same
// W8A8 value linalg.MatmulBTW8A8 computes on the CPU (dynamic-quantize q to int8,
// int8-dot against row j, rescale by the query and row scales) — so a GPU-scored
// top-k is rank-identical to the CPU one within the int8 quantization.
type I8Index interface {
	Score(q []float32, dst []float32) error
	Close() error
}

// I8BatchIndex is an optional extension of I8Index: a device index that scores
// MANY queries against the whole corpus in a single dispatch — the batched int8
// GEMM that is the GPU's sweet spot (M queries × N rows). A backend whose I8Index
// also implements this is used by FlatI8.QueryBatch; one that doesn't falls back
// to per-query scoring.
type I8BatchIndex interface {
	// ScoreBatch scores M = len(queries) queries against all N rows into dst
	// (length M*N, row-major): dst[m*N+j] is the W8A8 score of queries[m] against
	// row j — the same values Score would produce per query, laid out as a matrix.
	ScoreBatch(queries [][]float32, dst []float32) error
}

// I8TopKIndex is an optional extension of I8BatchIndex: a device index that returns
// each query's top-k DIRECTLY, doing the selection on the device. That avoids the
// full M*N score matrix entirely on the host side — neither the M*N readback nor the
// M*N host allocation ScoreBatch requires (at N=1e6, batch=256 that matrix is ~1 GB) —
// so only M*k hits cross back. FlatI8.QueryBatch prefers it when the backend
// implements it. The returned hits are rank-identical to ScoreBatch + topHits within
// the int8 quantization (score descending, ties broken by lower index).
type I8TopKIndex interface {
	TopKBatch(queries [][]float32, k int) ([][]Hit, error)
}

// QueryBatch runs a batch of queries, returning each one's k highest hits (the
// same per-query contract as Query). When the index is GPU-enabled and the backend
// supports batched scoring, the whole batch is scored in one int8 GEMM on the
// device — the throughput win over Query-in-a-loop; otherwise it degrades to
// per-query scoring. A query of the wrong dimension yields nil for that row.
func (f *FlatI8) QueryBatch(queries [][]float32, k int) [][]Hit {
	if f.closed {
		panic("ann: QueryBatch on a closed FlatI8 (mmap released by Close)")
	}
	out := make([][]Hit, len(queries))
	allDim := f.n > 0 && len(queries) > 0
	for _, q := range queries {
		if len(q) != f.dim {
			allDim = false
			break
		}
	}
	// CPU∥GPU shard-split path (preferred over all three below, when configured):
	// a device-resident shard and a CPU-scored shard score the whole batch
	// concurrently, then their per-query top-k results merge. See
	// EnableGPUShardSplit. Falls through to the tiers below on any error, exactly
	// like the tiers below fall through to each other.
	if f.gpuShard != nil && allDim && k > 0 {
		if hits, ok := f.queryBatchSharded(queries, k); ok {
			return hits
		}
	}
	// Device top-k path (preferred): the backend selects each query's top-k on the
	// device, so the M*N score matrix is never copied to — or allocated on — the host.
	if ti, ok := f.gpu.(I8TopKIndex); ok && allDim && k > 0 && k < f.n {
		if hits, err := ti.TopKBatch(queries, k); err == nil && len(hits) == len(queries) {
			return hits
		}
	}
	// Device batched path: one GEMM for the whole batch, then top-k on the host.
	if bi, ok := f.gpu.(I8BatchIndex); ok && allDim {
		// The M×N score matrix is POOLED, not allocated per call (lens doc §4.7).
		// It is genuinely needed here — unlike the device top-k path above, which
		// never materializes it — because ScoreBatch's contract is to fill a host
		// buffer, and on a PCIe device that buffer is the staging area. What was
		// wasteful was allocating a fresh one on every call: at 256 queries over a
		// 1M-row index that is a 1 GB allocation per batch, handed straight to the
		// GC. So the fix is a reused scratch rather than removal, exactly as the
		// finding says of the identical code in gpu/enccuda.
		//
		// Reuse is safe because ScoreBatch ASSIGNS every element of dst[0:M*N] —
		// that is its documented contract — and topHits reads only what was just
		// written. TestQueryBatch_pooledScratchIsInert poisons the pool to check it.
		sc := batchScratchPool.Get().(*batchScratch)
		defer batchScratchPool.Put(sc)
		dst := sc.buf(len(queries) * f.n)
		if err := bi.ScoreBatch(queries, dst); err == nil {
			for m := range queries {
				out[m] = f.topHits(dst[m*f.n:(m+1)*f.n], k, nil)
			}
			return out
		}
	}
	// Fallback: per query (CPU, or single-query GPU if enabled).
	for m, q := range queries {
		out[m] = f.Query(q, k)
	}
	return out
}

// batchScratch is the pooled M×N host score matrix for the device batched path.
// Separate from flatI8ScratchPool because that one is sized N per query and this
// one is M·N per batch — sharing would inflate every single-query scratch to a
// batch-sized buffer for the rest of the process.
type batchScratch struct{ dst []float32 }

// buf returns a need-long slice of sc.dst, growing the backing array when it is
// too small. Callers do the Get/Put themselves rather than having this method
// wrap them, and that is deliberate: `defer batchScratchPool.Put(sc)` has to sit
// in the frame that USES the returned slice. A helper that acquired and released
// around its own body would hand back a slice it had already returned to the
// pool, so a concurrent Get could hand the same backing array to another
// goroutine mid-read. Sizing is the only part that is safe to share.
func (sc *batchScratch) buf(need int) []float32 {
	if cap(sc.dst) < need {
		sc.dst = make([]float32, need)
	}
	return sc.dst[:need]
}

var batchScratchPool = sync.Pool{New: func() any { return new(batchScratch) }}

var backend Backend

// RegisterBackend installs the process-wide device backend for FlatI8 scoring.
// Called from an init() in the aikit/gpu build; the default build never calls it.
// Last registration wins (a process links at most one native backend).
func RegisterBackend(b Backend) { backend = b }

// HasBackend reports whether a device backend has been registered.
func HasBackend() bool { return backend != nil }

// BackendName returns the registered backend's name, or "" if none.
func BackendName() string {
	if backend == nil {
		return ""
	}
	return backend.Name()
}

// EnableGPU makes this index's codes resident on the registered device backend so
// subsequent Query calls score on the GPU. It returns an error if no backend is
// registered (default build), if the index is paged (LoadFlatI8MmapPaged — the
// budget-paged path is CPU-only until Phase 2), or if the device upload fails; on
// any error the index keeps scoring on the CPU. Idempotent-ish: a second call
// replaces the prior device index (the old one is closed).
func (f *FlatI8) EnableGPU() error {
	if backend == nil {
		return errNoBackend
	}
	if f.pager != nil {
		return errPagedGPU
	}
	if f.closed {
		panic("ann: EnableGPU on a closed FlatI8")
	}
	idx, err := backend.NewI8Index(f.bq, f.scales, f.n, f.dim)
	if err != nil {
		return err
	}
	if f.gpu != nil {
		_ = f.gpu.Close() //nolint:errcheck // releasing the superseded GPU index; the replacement already built
	}
	f.gpu = idx
	return nil
}

// GPUEnabled reports whether Query scores on the device backend. Only reflects
// EnableGPU (the whole-corpus path): an index with EnableGPUShardSplit but not
// EnableGPU reports false here even though QueryBatch is GPU-accelerated,
// since Query/QueryFilter are unaffected by the shard-split path.
func (f *FlatI8) GPUEnabled() bool { return f.gpu != nil }

// EnableGPUShardSplit makes the FIRST rows = int(gpuShare*n) rows resident on
// the registered device backend, and marks the remaining rows to score on the
// CPU — concurrently with the device shard, on every subsequent QueryBatch
// call (see the shard-split tier at the top of QueryBatch). Independent of
// EnableGPU: Query and QueryFilter are unaffected, only QueryBatch is.
//
// gpuShare is an absolute row-count knob, not an auto-tuned one: how many
// rows are worth putting on the device is backend- and scale-dependent (a
// device backend's own on-device top-k selection declines below its own
// threshold — e.g. gpu/annmetal's topkMinN=100_000, gpu/anncuda's =50_000,
// neither visible to this package — falling back to a score-then-host-select
// path that is slower per row than the fused one). Pick gpuShare by measuring
// QueryBatch throughput at a few candidate shares on the box that matters,
// the same way this repo's crossover benchmarks record a per-box,
// per-backend number rather than compute one.
//
// gpuShare must be in (0, 1) and n >= 2, so neither shard is empty; returns
// errShardRange otherwise. Also returns an error if no backend is registered,
// if the index is paged (CPU-only until Phase 2, same as EnableGPU), or if
// the device upload fails — on any error QueryBatch keeps using its other
// tiers. A second call replaces the prior shard device index (the old one is
// closed).
func (f *FlatI8) EnableGPUShardSplit(gpuShare float64) error {
	if backend == nil {
		return errNoBackend
	}
	if f.pager != nil {
		return errPagedGPU
	}
	if f.closed {
		panic("ann: EnableGPUShardSplit on a closed FlatI8")
	}
	if f.n < 2 || !(gpuShare > 0 && gpuShare < 1) {
		return errShardRange
	}
	// Clamp to [1, f.n-1] — at least one row each side of the split. The bounds
	// cannot invert: the guard above already rejects f.n < 2, so f.n-1 >= 1.
	rows := min(max(int(gpuShare*float64(f.n)), 1), f.n-1)
	idx, err := backend.NewI8Index(f.bq[:rows*f.dim], f.scales[:rows], rows, f.dim)
	if err != nil {
		return err
	}
	if f.gpuShard != nil {
		_ = f.gpuShard.Close() //nolint:errcheck // releasing the superseded shard index; the replacement already built
	}
	f.gpuShard = idx
	f.gpuShardRows = rows
	return nil
}

// queryBatchSharded runs the CPU∥GPU shard-split path: the device-resident
// shard and the CPU-scored shard score the whole batch CONCURRENTLY, then
// each query's two shard-local top-k results merge into one global top-k. ok
// is false if either shard's scoring failed, in which case the caller falls
// through to QueryBatch's other tiers — the same "try, else fall through"
// contract every tier in QueryBatch uses.
func (f *FlatI8) queryBatchSharded(queries [][]float32, k int) ([][]Hit, bool) {
	var gpuHits [][]Hit
	var gpuOK bool
	var wg sync.WaitGroup
	wg.Go(func() {
		gpuHits, gpuOK = f.scoreShardGPU(queries, k)
	})

	cpuHits, cpuOK := f.scoreShardCPU(queries, k)
	wg.Wait()

	if !gpuOK || !cpuOK {
		return nil, false
	}

	out := make([][]Hit, len(queries))
	for m := range queries {
		// gpuHits[m]'s indices are already corpus-global (the shard is rows
		// [0, gpuShardRows)); cpuHits[m]'s were offset in scoreShardCPU. Gather +
		// full sort + truncate is Flat.queryShards' own merge shape (flat.go),
		// just 2-way instead of N-way — correct for the same reason: each
		// shard's local top-k is a superset of its contribution to the global
		// top-k under this package's total order (hitCmp).
		merged := append(append([]Hit(nil), gpuHits[m]...), cpuHits[m]...)
		slices.SortFunc(merged, hitCmp)
		if len(merged) > k {
			merged = merged[:k]
		}
		out[m] = merged
	}
	return out, true
}

// scoreShardGPU scores queries against the device-resident shard [0,
// gpuShardRows), preferring on-device top-k selection and falling back to
// score-then-host-select — the same two-tier fallback QueryBatch's own
// whole-corpus dispatch uses above, just scoped to the shard.
func (f *FlatI8) scoreShardGPU(queries [][]float32, k int) ([][]Hit, bool) {
	if ti, ok := f.gpuShard.(I8TopKIndex); ok && k < f.gpuShardRows {
		if hits, err := ti.TopKBatch(queries, k); err == nil && len(hits) == len(queries) {
			return hits, true
		}
	}
	bi, ok := f.gpuShard.(I8BatchIndex)
	if !ok {
		return nil, false
	}
	sc := batchScratchPool.Get().(*batchScratch)
	defer batchScratchPool.Put(sc)
	dst := sc.buf(len(queries) * f.gpuShardRows)
	if err := bi.ScoreBatch(queries, dst); err != nil {
		return nil, false
	}
	hits := make([][]Hit, len(queries))
	for m := range queries {
		hits[m] = f.topHits(dst[m*f.gpuShardRows:(m+1)*f.gpuShardRows], k, nil)
	}
	return hits, true
}

// scoreShardCPU scores queries against the CPU-scored shard [gpuShardRows,
// n) in one batched W8A8 matmul — the same kernel query() uses at M=1,
// generalized here to M=len(queries) so the CPU side gets the same
// weight-streamed-once win the device batched tier gets — then offsets
// returned indices by gpuShardRows so they are corpus-global.
func (f *FlatI8) scoreShardCPU(queries [][]float32, k int) ([][]Hit, bool) {
	shardN := f.n - f.gpuShardRows
	if shardN <= 0 {
		return nil, false
	}
	M := len(queries)
	sc := shardCPUScratchPool.Get().(*shardCPUScratch)
	defer shardCPUScratchPool.Put(sc)
	if cap(sc.qbuf) < M*f.dim {
		sc.qbuf = make([]float32, M*f.dim)
	}
	qbuf := sc.qbuf[:M*f.dim]
	for m, q := range queries {
		copy(qbuf[m*f.dim:(m+1)*f.dim], q)
	}
	need := M * shardN
	if cap(sc.dst) < need {
		sc.dst = make([]float32, need)
	}
	dst := sc.dst[:need]
	linalg.MatmulBTW8A8Into(&sc.ws, qbuf, f.bq[f.gpuShardRows*f.dim:], f.scales[f.gpuShardRows:], dst, M, f.dim, shardN)
	hits := make([][]Hit, M)
	for m := range queries {
		h := f.topHits(dst[m*shardN:(m+1)*shardN], k, nil)
		for i := range h {
			h[i].Index += f.gpuShardRows
		}
		hits[m] = h
	}
	return hits, true
}

// shardCPUScratch is the pooled per-batch scratch for scoreShardCPU: the
// flattened-queries buffer, the W8A8 kernel's Workspace, and the shard-sized
// score matrix. Separate from flatI8Scratch/batchScratch — a different shape
// (M·dim + M·shardN) from either of those.
type shardCPUScratch struct {
	ws   linalg.Workspace
	qbuf []float32
	dst  []float32
}

var shardCPUScratchPool = sync.Pool{New: func() any { return new(shardCPUScratch) }}

type annError string

func (e annError) Error() string { return string(e) }

const (
	errNoBackend  annError = "ann: no device backend registered (build with the aikit/gpu backend)"
	errPagedGPU   annError = "ann: EnableGPU unsupported on a paged index (CPU-only until Phase 2)"
	errShardRange annError = "ann: EnableGPUShardSplit requires 0 < gpuShare < 1 and n >= 2"
)
