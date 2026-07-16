package mmap

import (
	"os"
	"path/filepath"
	"testing"
)

// pattern is a deterministic byte at offset i — distinct enough that an off-by-one
// re-fault or a stale page would change the bytes we compare.
func pattern(i int) byte { return byte(i*131 + 7) }

func writePatternFile(t *testing.T, nBytes int) string {
	t.Helper()
	buf := make([]byte, nBytes)
	for i := range buf {
		buf[i] = pattern(i)
	}
	p := filepath.Join(t.TempDir(), "data.bin")
	if err := os.WriteFile(p, buf, 0o600); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	return p
}

// TestMapReadOnly_roundTrip checks the mapping aliases the file's bytes and Unmap
// is clean. (Ported intent from ann/embed's existing mmap coverage, now at the leaf.)
func TestMapReadOnly_roundTrip(t *testing.T) {
	const n = 64 << 10
	p := writePatternFile(t, n)
	b, err := MapReadOnly(p)
	if err != nil {
		t.Fatalf("MapReadOnly: %v", err)
	}
	if len(b) != n {
		t.Fatalf("len = %d, want %d", len(b), n)
	}
	for i := range b {
		if b[i] != pattern(i) {
			t.Fatalf("byte %d = %d, want %d", i, b[i], pattern(i))
		}
	}
	if err := Unmap(b); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
}

func TestMapReadOnly_tooSmall(t *testing.T) {
	p := filepath.Join(t.TempDir(), "tiny")
	if err := os.WriteFile(p, []byte{1, 2, 3}, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := MapReadOnly(p); err == nil {
		t.Fatal("expected error mapping a <8-byte file")
	}
}

// TestMadvise_dontneedRefaultsIntact is the correctness keystone for eviction
// (ported from goinfer's always-on test of the same name): a read-only file-backed
// mapping that is released with MADV_DONTNEED re-faults BYTE-IDENTICAL bytes on the
// next read. This is what makes SpanCache eviction lossless — dropping a span never
// costs correctness, only a re-fault. On Linux DONTNEED really frees the pages; on
// darwin/BSD/Windows Advise is a (best-effort) no-op, so the read is trivially
// identical there too — the property holds on every platform.
func TestMadvise_dontneedRefaultsIntact(t *testing.T) {
	const pages = 16
	pg := os.Getpagesize()
	p := writePatternFile(t, pages*pg)
	b, err := MapReadOnly(p)
	if err != nil {
		t.Fatalf("MapReadOnly: %v", err)
	}
	defer func() { _ = Unmap(b) }()

	span := PageAlignedInterior(b)
	if len(span) == 0 {
		t.Fatal("PageAlignedInterior returned nothing on a multi-page mapping")
	}

	// Fault it in, then release it.
	if err := Advise(span, true); err != nil { // WILLNEED
		t.Fatalf("Advise WILLNEED: %v", err)
	}
	touchAll(span)
	if err := Advise(span, false); err != nil { // DONTNEED — release resident pages
		t.Fatalf("Advise DONTNEED: %v", err)
	}

	// Re-read after release: every byte must match the file, re-faulting as needed.
	base := offsetOf(t, b, span)
	for i := range span {
		if got, want := span[i], pattern(base+i); got != want {
			t.Fatalf("after DONTNEED, byte %d = %d, want %d (re-fault not byte-identical)", base+i, got, want)
		}
	}

	// And again after an explicit WILLNEED prefetch.
	if err := Advise(span, true); err != nil {
		t.Fatalf("Advise WILLNEED (2): %v", err)
	}
	for i := range span {
		if got, want := span[i], pattern(base+i); got != want {
			t.Fatalf("after WILLNEED, byte %d = %d, want %d", base+i, got, want)
		}
	}
}

// touchAll reads every byte so the pages are actually resident before we release.
func touchAll(b []byte) {
	var sink byte
	for i := range b {
		sink ^= b[i]
	}
	_ = sink
}

// offsetOf returns span's start offset within the parent mapping b (span is a
// sub-slice of b produced by PageAlignedInterior).
func offsetOf(t *testing.T, b, span []byte) int {
	t.Helper()
	for off := 0; off+len(span) <= len(b); off++ {
		if &b[off] == &span[0] {
			return off
		}
	}
	t.Fatal("span is not a sub-slice of the mapping")
	return 0
}

func TestPageAlignedInterior(t *testing.T) {
	pg := os.Getpagesize()
	b, err := MapReadOnly(writePatternFile(t, 8*pg))
	if err != nil {
		t.Fatalf("MapReadOnly: %v", err)
	}
	defer func() { _ = Unmap(b) }()

	in := PageAlignedInterior(b)
	if len(in)%pg != 0 {
		t.Fatalf("interior length %d is not a multiple of page size %d", len(in), pg)
	}
	if len(in) > len(b) {
		t.Fatalf("interior %d larger than mapping %d", len(in), len(b))
	}

	// A sub-page slice has no full-page interior.
	if got := PageAlignedInterior(make([]byte, 1)); got != nil {
		t.Fatalf("sub-page slice should have nil interior, got len %d", len(got))
	}
	if got := PageAlignedInterior(nil); got != nil {
		t.Fatalf("nil slice should have nil interior")
	}
}
