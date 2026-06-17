//go:build linux

package mmap

import "syscall"

// Advise hints the kernel on the residency of a page-aligned span of a read-only
// mapping. willNeed (MADV_WILLNEED) hints to fault the span in; otherwise
// (MADV_DONTNEED) it releases the span's resident pages.
//
// Releasing is safe regardless of timing: a MapReadOnly mapping is read-only and
// file-backed (MAP_PRIVATE, clean pages), so a released span simply re-faults from
// the file on the next access — the bytes are identical. That's what lets SpanCache
// cap resident RAM without any correctness risk (an evicted-then-reused span costs a
// re-fault, never wrong data).
//
// This is the Linux path: MADV_DONTNEED frees the resident pages immediately, so the
// RAM cap is firm here. The stdlib's syscall.Madvise wrapper exists only on Linux,
// so darwin (which also lacks MADV_DONTNEED semantics for this mapping type) goes
// through golang.org/x/sys/unix in madvise_darwin.go, and every other platform
// (the BSDs, Windows) falls to the best-effort no-op in madvise_other.go. span must
// be page-aligned (use PageAlignedInterior).
func Advise(span []byte, willNeed bool) error {
	if len(span) == 0 {
		return nil
	}
	adv := syscall.MADV_DONTNEED
	if willNeed {
		adv = syscall.MADV_WILLNEED
	}
	return syscall.Madvise(span, adv)
}
