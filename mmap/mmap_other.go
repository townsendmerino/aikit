//go:build !unix

package mmap

import (
	"fmt"
	"os"
)

// MapReadOnly is the non-unix fallback for platforms without syscall.Mmap
// (notably Windows): it reads the whole file into the Go heap and returns it.
// Same interface as the unix mmap path, so callers work unchanged — the bytes just
// live in the heap instead of the OS page cache, so there is no zero-copy benefit
// and the residency hints (Advise / SpanCache) are inert: nothing to page. Unmap is
// then a no-op; the GC reclaims the slice. Files smaller than 8 bytes are rejected
// (like the unix path): every format this reads has an ≥8-byte prefix.
func MapReadOnly(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	if len(data) < 8 {
		return nil, fmt.Errorf("mmap %s: file too small (%d bytes)", path, len(data))
	}
	return data, nil
}

// Unmap is a no-op where MapReadOnly returns a heap slice (the GC reclaims it).
func Unmap([]byte) error { return nil }
