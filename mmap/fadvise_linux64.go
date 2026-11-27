//go:build linux && (amd64 || arm64)

package mmap

import (
	"os"
	"syscall"
)

// fadvDontNeed is POSIX_FADV_DONTNEED on amd64 and arm64.
const fadvDontNeed = 4

// DropFileCache tells the kernel the n bytes of f at off will not be read again (POSIX_FADV_DONTNEED), so it drops their
// clean page-cache pages now instead of holding them until memory runs short. A later read re-reads the file: the bytes are
// identical, the cost is a disk read where the cache would have served it. Pages still mapped by a process stay cached; drop
// the mapping's own pages first (Advise). Advisory. On linux/amd64 and linux/arm64, whose fadvise64 takes the four arguments
// plainly; a no-op on every other platform (fadvise_other.go).
func DropFileCache(f *os.File, off, n int64) error {
	if f == nil || n <= 0 {
		return nil
	}
	if _, _, e := syscall.Syscall6(syscall.SYS_FADVISE64, f.Fd(), uintptr(off), uintptr(n), fadvDontNeed, 0, 0); e != 0 {
		return e
	}
	return nil
}
