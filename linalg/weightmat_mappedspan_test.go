package linalg

import (
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/mmap"
)

// TestWeightMat_MappedSpan checks the three cases the pager relies on: a quantized
// weight whose codes alias a mapping yields a page-aligned in-bounds span; a
// heap-backed quantized weight yields nil; an f32 weight yields nil.
func TestWeightMat_MappedSpan(t *testing.T) {
	pg := os.Getpagesize()
	// A file big enough that a tensor's codes have a non-empty page-aligned interior.
	const pages = 8
	buf := make([]byte, pages*pg)
	for i := range buf {
		buf[i] = byte(i)
	}
	p := filepath.Join(t.TempDir(), "codes.bin")
	if err := os.WriteFile(p, buf, 0o600); err != nil {
		t.Fatal(err)
	}
	mapping, err := mmap.MapReadOnly(p)
	if err != nil {
		t.Fatalf("MapReadOnly: %v", err)
	}
	defer mmap.Unmap(mapping)

	base := uintptr(unsafe.Pointer(&mapping[0]))
	end := base + uintptr(len(mapping))

	// Treat the whole mapping as a [rows, cols] int8 tensor aliased from it (no copy).
	rows, cols := 8, pages*pg/8
	codes := unsafe.Slice((*int8)(unsafe.Pointer(&mapping[0])), rows*cols)
	scales := make([]float32, rows)
	mapped := WrapInt8(codes, scales, rows, cols, false)

	span := mapped.MappedSpan(base, end)
	if len(span) == 0 {
		t.Fatal("mapping-backed int8 weight: MappedSpan returned nil")
	}
	if len(span)%pg != 0 {
		t.Fatalf("span length %d is not page-aligned (%d)", len(span), pg)
	}
	sStart := uintptr(unsafe.Pointer(&span[0]))
	if sStart < base || sStart+uintptr(len(span)) > end {
		t.Fatalf("span [%#x,%#x) escapes mapping [%#x,%#x)", sStart, sStart+uintptr(len(span)), base, end)
	}

	// Heap-backed int8 weight of the same shape: its codes are NOT in the mapping.
	heapCodes := make([]int8, rows*cols)
	heap := WrapInt8(heapCodes, scales, rows, cols, false)
	if got := heap.MappedSpan(base, end); got != nil {
		t.Fatalf("heap-backed weight should yield nil, got len %d", len(got))
	}

	// f32 weight is never pageable here.
	f32 := WrapF32(make([]float32, rows*cols), rows, cols)
	if got := f32.MappedSpan(base, end); got != nil {
		t.Fatalf("f32 weight should yield nil, got len %d", len(got))
	}

	// Zero value: no storage, nil span.
	var zero WeightMat
	if got := zero.MappedSpan(base, end); got != nil {
		t.Fatalf("zero-value weight should yield nil, got len %d", len(got))
	}
}
