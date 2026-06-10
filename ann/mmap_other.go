//go:build !unix

package ann

import (
	"fmt"
	"os"
)

// mmapReadOnly is the non-unix fallback (notably Windows): it reads the whole file
// into the Go heap. Same interface as the unix mmap path, so LoadFlatI8Mmap works
// unchanged — the bytes just live in the heap instead of the OS page cache, so on
// these platforms it costs the same RAM as LoadFlatI8 (no zero-copy benefit, but
// correct). munmap is then a no-op; the GC reclaims the slice.
func mmapReadOnly(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	if len(data) < 8 {
		return nil, fmt.Errorf("mmap %s: file too small (%d bytes)", path, len(data))
	}
	return data, nil
}

// munmap is a no-op where mmapReadOnly returns a heap slice.
func munmap([]byte) error { return nil }
