//go:build linux || darwin

package guardmem

import (
	"fmt"
	"runtime"
	"syscall"
)

// Supported is true where New maps guard pages.
const Supported = true

func pageSize() int { return syscall.Getpagesize() }

func platform() string { return runtime.GOOS + "/" + runtime.GOARCH }

// mapGuarded maps data bytes (a whole number of pages) between two pages with no access
// and returns the whole mapping, guards included.
func mapGuarded(data int) ([]byte, error) {
	page := pageSize()
	m, err := syscall.Mmap(-1, 0, data+2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}
	if err := syscall.Mprotect(m[:page], syscall.PROT_NONE); err != nil {
		_ = syscall.Munmap(m)
		return nil, fmt.Errorf("mprotect low guard: %w", err)
	}
	if err := syscall.Mprotect(m[page+data:], syscall.PROT_NONE); err != nil {
		_ = syscall.Munmap(m)
		return nil, fmt.Errorf("mprotect high guard: %w", err)
	}
	return m, nil
}

func unmap(m []byte) { _ = syscall.Munmap(m) }
