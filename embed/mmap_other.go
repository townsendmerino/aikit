//go:build !unix

package embed

import (
	"fmt"
	"os"
)

// mmapReadOnly is the non-unix fallback for platforms without syscall.Mmap
// (notably Windows): it reads the whole file into the Go heap and returns it.
// Same interface as the unix mmap path, so OpenSafetensorsMmap / OpenGGUFMmap
// work unchanged — the only difference is the bytes live in the heap instead
// of the OS page cache, so on these platforms the "mmap" loaders cost the same
// RAM as the plain heap loaders (OpenSafetensors / OpenGGUF). munmap is then a
// no-op; the GC reclaims the slice.
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
