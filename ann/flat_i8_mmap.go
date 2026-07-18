package ann

import (
	"runtime"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/mmap"
)

// LoadFlatI8Mmap loads a FlatI8 index by memory-mapping path and ALIASING the int8
// codes directly from the mapping — zero-copy. So a large embedded index is
// query-ready instantly (no parse-and-copy of the n×dim codes into the heap) and
// its bytes live in the shared OS page cache rather than the Go heap. Only the
// small scales block (n floats) is copied. On non-unix platforms the file is
// heap-read instead (identical result, no page-cache benefit). For a plain
// in-memory copy from a byte slice, use LoadFlatI8.
//
// Lifetime: the returned *FlatI8 aliases the mapping, so keep it reachable for as
// long as you Query it; a finalizer unmaps it once it becomes unreachable. Close
// releases the mapping eagerly — and Query after Close panics, so only Close when
// you are done:
//
//	// WRONG — released while still in use.
//	f, _ := ann.LoadFlatI8Mmap("index.bin")
//	f.Close()
//	f.Query(q, 10) // panics: codes unmapped
//
//	// RIGHT — Close (or just let GC unmap it) only after the last query.
//	f, _ := ann.LoadFlatI8Mmap("index.bin")
//	defer f.Close()
//	hits := f.Query(q, 10)
func LoadFlatI8Mmap(path string) (*FlatI8, error) {
	f, _, err := mapAndAliasFlatI8(path)
	if err != nil {
		return nil, err
	}
	runtime.SetFinalizer(f, (*FlatI8).finalizeMmap)
	return f, nil
}

// pagedBlockTargetBytes is the rough byte size of one paging block: a block is the
// unit SpanCache faults in and evicts, so it wants to be several pages (fine-grained
// enough for the budget to bite, coarse enough to amortize the madvise calls). 1 MiB
// is many pages on any platform while staying small against a larger-than-RAM index.
const pagedBlockTargetBytes = 1 << 20

// pagedBlockRows is how many int8 rows (each dim bytes) make up one paging block,
// targeting pagedBlockTargetBytes — at least one row so a wide vector still pages.
func pagedBlockRows(dim int) int {
	if dim <= 0 {
		return 1
	}
	r := max(pagedBlockTargetBytes/dim, 1)
	return r
}

// LoadFlatI8MmapPaged is LoadFlatI8Mmap with a resident-RAM cap: the int8 code block
// is split into fixed-size blocks of rows, and each Query pages the blocks it scans
// in and out of the read-only mapping through an mmap.SpanCache, keeping resident
// code bytes near budget. This is what lets aikit query an int8 index whose codes
// EXCEED RAM — the cold blocks re-fault from the mapping (lossless: read-only,
// file-backed) and the query streams through them under the cap. A budget <= 0
// auto-selects ~half of available RAM (mmap.AutoBudget).
//
// The firm cap is Linux-only (madvise MADV_DONTNEED); on darwin/BSD/Windows the
// pager's bookkeeping still runs but the OS, not aikit, decides when to reclaim the
// clean pages (see package mmap). Results are byte-identical to a fully-resident
// LoadFlatI8Mmap — paging changes only residency, never the scores.
//
// Tradeoff vs LoadFlatI8Mmap: the pager is stateful, so concurrent Query calls stay
// correct but serialize through it — a paged index keeps the cap at the cost of
// cross-query parallelism (throughput is fault-bound anyway; the point is to fit at
// all rather than to go fast). Lifetime/Close are identical to LoadFlatI8Mmap.
func LoadFlatI8MmapPaged(path string, budget int64) (*FlatI8, error) {
	return loadFlatI8MmapPaged(path, budget, 0)
}

// loadFlatI8MmapPaged is LoadFlatI8MmapPaged with an explicit block size (rows per
// paging block); blockRows <= 0 uses the auto size (pagedBlockRows). The block size
// is a test seam — production always takes the auto path — so a unit test can force
// many small blocks (and real eviction) on a modest corpus.
func loadFlatI8MmapPaged(path string, budget int64, blockRows int) (*FlatI8, error) {
	f, at, err := mapAndAliasFlatI8(path)
	if err != nil {
		return nil, err
	}
	if budget <= 0 {
		budget = mmap.AutoBudget()
	}
	if blockRows <= 0 {
		blockRows = pagedBlockRows(f.dim)
	}
	f.blockRows = blockRows
	f.pager = mmap.NewSpanCache[int](budget)
	// Register each block's page-aligned span within the mapping. Block b covers
	// rows [b*blockRows, …); its bytes are data[at+r0*dim : at+r1*dim].
	for b, r0 := 0, 0; r0 < f.n; b, r0 = b+1, r0+f.blockRows {
		r1 := min(r0+f.blockRows, f.n)
		lo, hi := at+r0*f.dim, at+r1*f.dim
		f.pager.Add(b, [][]byte{mmap.PageAlignedInterior(f.mmap[lo:hi])})
	}
	runtime.SetFinalizer(f, (*FlatI8).finalizeMmap)
	return f, nil
}

// scorePaged scores q against every stored vector block by block, touching each
// block's spans (faulting them in, evicting the LRU tail to stay under budget)
// before scoring its rows. Splitting the scan into per-block MatmulBTW8A8 calls is
// byte-identical to one whole-corpus call: the query quantization is deterministic
// and per-row dots are independent, so dst is the same either way. Serialized via
// pagerMu because the pager is stateful.
func (f *FlatI8) scorePaged(q []float32, dst []float32) {
	f.pagerMu.Lock()
	defer f.pagerMu.Unlock()
	for b, r0 := 0, 0; r0 < f.n; b, r0 = b+1, r0+f.blockRows {
		r1 := min(r0+f.blockRows, f.n)
		f.pager.Touch(b)
		linalg.MatmulBTW8A8(q, f.bq[r0*f.dim:r1*f.dim], f.scales[r0:r1], dst[r0:r1], 1, f.dim, r1-r0)
	}
}

// PageStats reports the cumulative (hits, misses, evictions) of a paged index's span
// cache and whether the index is paged at all. A non-zero eviction count means the
// budget actually bit (the LRU tail was released), not just cold-start misses. For a
// non-paged index (LoadFlatI8Mmap / NewFlatI8 / LoadFlatI8) paged is false and the
// counts are zero.
func (f *FlatI8) PageStats() (hits, misses, evictions int64, paged bool) {
	if f.pager == nil {
		return 0, 0, 0, false
	}
	h, m, e := f.pager.Stats()
	return h, m, e, true
}

// mapAndAliasFlatI8 maps path read-only, validates the FlatI8 layout, copies the
// small scales block, and aliases the int8 code block directly from the mapping
// (zero-copy). It returns the index (mmap/bq/scales/n/dim set, NO finalizer yet — the
// caller sets it) and the codes' byte offset `at` (the paged loader needs it to
// register block spans). On any error the mapping is released.
func mapAndAliasFlatI8(path string) (f *FlatI8, at int, err error) {
	data, err := mmap.MapReadOnly(path)
	if err != nil {
		return nil, 0, err
	}
	dim, n, at, err := flatI8Layout(data)
	if err != nil {
		_ = mmap.Unmap(data)
		return nil, 0, err
	}
	f = &FlatI8{
		scales: readScales(data[at+n*dim:], n),
		n:      n,
		dim:    dim,
		mmap:   data,
	}
	if n*dim > 0 {
		// int8 has no alignment requirement, so aliasing the code block from the
		// (page-aligned) read-only mapping is safe on every architecture.
		f.bq = unsafe.Slice((*int8)(unsafe.Pointer(&data[at])), n*dim)
	}
	return f, at, nil
}

// Close releases the mapping of a LoadFlatI8Mmap index. It is a no-op (and leaves
// the index queryable) for an in-memory index from NewFlatI8 / LoadFlatI8, and is
// idempotent. After Close on a mapped index the codes are unmapped, so Query
// panics — Close only once you are done querying. Not safe to call concurrently
// with Query (coordinate the handoff, as with any teardown).
func (f *FlatI8) Close() error {
	if f.mmap == nil || f.closed {
		return nil // in-memory (nothing to release) or already closed
	}
	f.closed = true
	f.bq = nil
	runtime.SetFinalizer(f, nil)
	return mmap.Unmap(f.mmap)
}

// finalizeMmap unmaps a still-open mapping when the index becomes unreachable
// without an explicit Close — the safety net for callers who forget.
func (f *FlatI8) finalizeMmap() {
	if !f.closed && f.mmap != nil {
		_ = mmap.Unmap(f.mmap)
	}
}
