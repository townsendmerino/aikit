// Package mmap is a zero-dependency-on-aikit leaf providing the OS memory
// substrate that aikit's loaders sit on: a read-only file mapping, residency
// hints (madvise), and a demand-signal-agnostic span-residency cache for paging
// a mapping in and out under a byte budget.
//
// # Why a leaf
//
// The read-only mmap primitive was duplicated byte-for-byte across aikit (ann and
// embed each kept a private copy to avoid an ann→embed edge) and goinfer. This
// package is the single home for it: ann and embed both import it, so the copies
// collapse to one without any package growing a dependency on another aikit
// package. The leaf invariant is that this package imports only the standard
// library (plus golang.org/x/sys/unix on darwin, where the stdlib lacks a madvise
// wrapper) and never imports another aikit package — so any aikit package can
// import it with zero cycle risk. It is cgo-free; the //go:build !unix builds
// fall back to a heap read and inert hints, exactly as aikit's existing loaders
// do.
//
// # What it provides
//
//   - MapReadOnly / Unmap — a read-only MAP_PRIVATE mapping of a whole file.
//   - Advise — MADV_WILLNEED / MADV_DONTNEED over a page-aligned span. The RAM cap
//     is firm on Linux/BSD; on darwin eviction is a no-op (clean file-backed pages
//     are reclaimed only under pressure) and WILLNEED is an advisory prefetch.
//   - SpanCache — an LRU of page-aligned spans within a mapping, bounded by a byte
//     budget, faulting a member in on Touch and releasing the LRU tail. It is
//     demand-signal-agnostic: the caller registers each member's spans and decides
//     when to Touch. No model-specific logic lives here.
//   - PageAlignedInterior, AvailableRAM, AutoBudget — page-rounding and RAM-budget
//     helpers the cache and its callers need.
//
// Releasing a span is safe at any time: a MapReadOnly mapping is read-only and
// file-backed, so a released (or OS-reclaimed) page simply re-faults identical
// bytes on the next access. Eviction therefore costs a fault, never wrong data.
//
// This is Experimental-tier surface — it may change in a minor release.
package mmap
