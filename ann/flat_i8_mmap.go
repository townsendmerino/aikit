package ann

import (
	"runtime"
	"unsafe"
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
	data, err := mmapReadOnly(path)
	if err != nil {
		return nil, err
	}
	dim, n, at, err := flatI8Layout(data)
	if err != nil {
		_ = munmap(data)
		return nil, err
	}
	f := &FlatI8{
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
	runtime.SetFinalizer(f, (*FlatI8).finalizeMmap)
	return f, nil
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
	return munmap(f.mmap)
}

// finalizeMmap unmaps a still-open mapping when the index becomes unreachable
// without an explicit Close — the safety net for callers who forget.
func (f *FlatI8) finalizeMmap() {
	if !f.closed && f.mmap != nil {
		_ = munmap(f.mmap)
	}
}
