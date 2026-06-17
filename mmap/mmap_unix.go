//go:build unix

package mmap

import (
	"fmt"
	"os"
	"syscall"
)

// MapReadOnly returns a read-only MAP_PRIVATE mapping of path's whole contents.
// The fd is closed before returning — the mapping survives it. Pure Go (stdlib
// syscall, no cgo).
//
// This is the seam that makes aliased bytes pageable: the contents live in the OS
// page cache, faulted in lazily and evictable via Advise, instead of being eagerly
// copied onto the Go heap. The //go:build !unix sibling in mmap_other.go falls back
// to a heap read so the package still compiles + works on Windows et al.; callers
// are platform-agnostic against this pair (they just lose page-cache sharing and
// residency control there).
//
// The returned slice must be released with Unmap, not left to the GC.
func MapReadOnly(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close() // fd no longer needed after mmap

	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}
	sz := st.Size()
	if sz < 8 {
		return nil, fmt.Errorf("mmap %s: file too small (%d bytes)", path, sz)
	}
	if sz > int64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("mmap %s: file too large for this platform (%d bytes)", path, sz)
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(sz), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}
	return data, nil
}

// Unmap releases a mapping returned by MapReadOnly. Safe on a nil/empty slice.
func Unmap(b []byte) error {
	if len(b) == 0 {
		return nil
	}
	return syscall.Munmap(b)
}
