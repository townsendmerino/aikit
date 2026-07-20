//go:build darwin

package mmap

import "golang.org/x/sys/unix"

// Advise on darwin honors the willNeed (prefetch) hint but CANNOT enforce an
// on-demand RAM cap — unlike the Linux path in madvise_linux.go.
//
// The willNeed=true case issues MADV_WILLNEED, a real, working fault-ahead hint
// (syscall.Madvise doesn't exist on darwin, so this goes through golang.org/x/sys/unix —
// the one external import the leaf takes, see the package doc).
//
// The eviction case (willNeed=false) is a deliberate no-op. macOS keeps clean,
// file-backed pages of a read-only MAP_PRIVATE mapping in the Unified Buffer Cache
// and reclaims them only under memory pressure; there is no syscall that forces an
// immediate resident drop for this mapping type. Empirically (verified on
// darwin/arm64): MADV_DONTNEED and MADV_FREE leave RSS unchanged, MADV_FREE_REUSABLE
// returns EPERM (it is a malloc-zone flag, illegal on a file-backed mapping), and the
// msync MS_INVALIDATE/MS_KILLPAGES/MS_DEACTIVATE variants are no-ops on RSS too.
//
// So on darwin SpanCache's bookkeeping still runs and WILLNEED prefetch works; the
// resident pages are clean and freely evictable by the OS when RAM is tight (no
// writeback, no OOM risk), but there is NO firm, self-enforced cap here — that
// guarantee is Linux-only (the BSDs route through madvise_other.go's no-op Advise,
// so they have no firm cap either). Correctness is unaffected regardless: the mapping is
// read-only and file-backed, so any page the OS reclaims simply re-faults identical
// bytes. span must be page-aligned (use PageAlignedInterior).
func Advise(span []byte, willNeed bool) error {
	if len(span) == 0 || !willNeed {
		return nil // no-op eviction: see above — no darwin syscall forces a resident drop here
	}
	return unix.Madvise(span, unix.MADV_WILLNEED)
}
