package embed

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"
	"unsafe"
)

// R-11 (goinfer docs/tasks/task-recompute-audit.md §5), embed's half. The dequantizers write into a caller-owned dst, and a loader reuses that scratch row after
// row, so the contract that matters is "the result does not depend on what dst held". Each writer is run with dst filled three ways (0xFF, zeros, 0x5A) and the
// results must be byte-identical, with the guard words past the region untouched. (This package's input is arbitrary bytes, so an f16 scale can legitimately decode
// to NaN; "no NaN left" is therefore replaced by the stronger "all three fills agree", which a surviving poison value cannot satisfy.)
func runDstIndependent(t *testing.T, name string, n int, call func(dst []float32)) {
	t.Helper()
	const guard = 16
	var outs [3][]byte
	for i, fill := range []byte{0xFF, 0x00, 0x5A} {
		back := make([]float32, n+guard)
		raw := unsafe.Slice((*byte)(unsafe.Pointer(&back[0])), len(back)*4)
		for j := range raw {
			raw[j] = fill
		}
		call(back[:n:n])
		if !bytes.Equal(raw[n*4:], bytes.Repeat([]byte{fill}, guard*4)) {
			t.Errorf("%s: wrote past dst[:%d] (fill %#x)", name, n, fill)
		}
		outs[i] = append([]byte(nil), raw[:n*4]...)
	}
	if !bytes.Equal(outs[0], outs[1]) || !bytes.Equal(outs[0], outs[2]) {
		t.Errorf("%s: the result depends on what dst held before the call — it does not overwrite every element it covers", name)
	}
}

func TestOutputContract_dequantWritersOverwrite(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	rnd := func(n int) []byte {
		b := make([]byte, n)
		rng.Read(b)
		return b
	}
	// Every type the reader admits, so a newly added dequantizer is covered without a list to update.
	covered := 0
	for typ := range uint32(256) {
		be, ok := ggmlBlockElems(typ)
		if !ok {
			continue
		}
		covered++
		name := fmt.Sprintf("type%d", typ)
		n := 3 * max(be, 1) * 2 // several blocks, and a non-zero start below
		raw := rnd(64 * 1024)
		start := 0
		if be > 1 {
			start = be // block-aligned, not zero
		}
		runDstIndependent(t, "dequantRange/"+name, n, func(dst []float32) { dequantRange(typ, raw, start, dst, be) })
	}
	if covered < 17 {
		t.Fatalf("only %d ggml types admitted; the reader knows at least 17", covered)
	}
	for _, nb := range []int{1, 3} {
		raw := rnd(nb * MXFP4BlockBytes)
		runDstIndependent(t, fmt.Sprintf("DequantMXFP4Blocks/nBlocks=%d", nb), nb*MXFP4BlockElems, func(dst []float32) {
			if err := DequantMXFP4Blocks(raw, nb, dst); err != nil {
				t.Fatal(err)
			}
		})
		blocks, scales := rnd(nb*mxfp4PackedBytes), rnd(nb)
		runDstIndependent(t, fmt.Sprintf("DequantMXFP4Split/nBlocks=%d", nb), nb*MXFP4BlockElems, func(dst []float32) {
			if err := DequantMXFP4Split(blocks, scales, nb, dst); err != nil {
				t.Fatal(err)
			}
		})
	}
}
