//go:build !linux && !darwin

package mmap

// Advise is a best-effort no-op on every platform but Linux and darwin: the BSDs
// (whose stdlib syscall package has no Madvise wrapper) and Windows. SpanCache's
// bookkeeping still runs, but it can't release resident pages, so there is no
// enforced RAM cap — "firm on Linux, best-effort elsewhere." Correctness is
// unaffected: a read-only file-backed mapping the OS reclaims under pressure simply
// re-faults identical bytes. On !unix the MapReadOnly fallback keeps everything on
// the heap anyway, so there is nothing to page there.
func Advise([]byte, bool) error { return nil }
