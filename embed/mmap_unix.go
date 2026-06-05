//go:build unix

package embed

import (
	"fmt"
	"os"
	"syscall"
)

// mmapReadOnly returns a read-only MAP_PRIVATE mapping of path's whole
// contents. The fd is closed before returning — the mapping survives it.
//
// This is the unix implementation (darwin/linux/bsd). The //go:build !unix
// sibling in mmap_other.go falls back to a heap read so the package still
// compiles on Windows et al.; callers (OpenSafetensorsMmap / OpenGGUFMmap)
// are platform-agnostic against this pair.
func mmapReadOnly(path string) ([]byte, error) {
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

// munmap releases a mapping returned by mmapReadOnly.
func munmap(b []byte) error { return syscall.Munmap(b) }
