package ann

import (
	"runtime"

	"github.com/townsendmerino/aikit/mmap"
)

// LoadHNSWMmap loads an HNSW index by memory-mapping path and ALIASING the
// vector block directly from the mapping — zero-copy, mirroring
// LoadFlatI8Mmap's int8 aliasing (ann/flat_i8_mmap.go). So a large embedded
// index is query-ready instantly (no parse-and-copy of the ndocs×dim vectors
// into the heap) and its bytes live in the shared OS page cache rather than
// the Go heap. From a v5 blob the graph (offsets and neighbor ids) and the
// int8 scales alias the mapping too, so a mapped load costs almost no heap
// (memory campaign item 7); a v4 blob's graph is still copied, into the same
// compact form. On non-unix platforms the file is heap-read instead
// (identical result, no page-cache benefit; see package mmap). For a plain
// in-memory copy from a byte slice, use Load.
//
// Lifetime: the returned *HNSW aliases the mapping, so keep it reachable for
// as long as you Query it; a finalizer unmaps it once it becomes unreachable.
// Close releases the mapping eagerly — and Query or Add after Close panics, so
// only Close when you are done:
//
//	// WRONG — released while still in use.
//	h, _ := ann.LoadHNSWMmap("index.bin")
//	h.Close()
//	h.Query(q, 10) // panics: vectors unmapped
//
//	// RIGHT — Close (or just let GC unmap it) only after the last query.
//	h, _ := ann.LoadHNSWMmap("index.bin")
//	defer h.Close()
//	hits := h.Query(q, 10)
func LoadHNSWMmap(path string) (*HNSW, error) {
	data, err := mmap.MapReadOnly(path)
	if err != nil {
		return nil, err
	}
	h, err := loadHNSW(data, true)
	if err != nil {
		_ = mmap.Unmap(data)
		return nil, err
	}
	h.mmap = data
	runtime.SetFinalizer(h, (*HNSW).finalizeMmap)
	return h, nil
}

// Close releases the mapping of a LoadHNSWMmap index. It is a no-op (and
// leaves the index queryable) for an in-memory index from NewHNSW / Add /
// Load, and is idempotent. After Close on a mapped index the vectors are
// unmapped, so Query and Add panic — Close only once you are done. Not safe
// to call concurrently with Query or Add (coordinate the handoff, as with any
// teardown).
func (h *HNSW) Close() error {
	if h.mmap == nil || h.closed {
		return nil // in-memory (nothing to release) or already closed
	}
	h.closed = true
	h.vecs = nil
	h.bq = nil
	h.scales = nil
	h.csr = nil
	runtime.SetFinalizer(h, nil)
	return mmap.Unmap(h.mmap)
}

// finalizeMmap unmaps a still-open mapping when the index becomes unreachable
// without an explicit Close — the safety net for callers who forget.
func (h *HNSW) finalizeMmap() {
	if !h.closed && h.mmap != nil {
		_ = mmap.Unmap(h.mmap)
	}
}
