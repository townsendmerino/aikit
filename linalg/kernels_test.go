package linalg

import (
	"slices"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/internal/kernreg"
)

// kernelRows is the inventory of this package's assembly entry points, one row per
// TEXT symbol on each architecture. Each kernel family declares its rows beside its own
// test file; this list gathers them.
var kernelRows = slices.Concat(
	dotRows,
	kquantRows,
	attnRows,
	expRows,
	widenRows,
	f16scaleRows,
	hammingKernelRows,
	peakRows,
	doti8Rows,
	dequanti8Rows,
	dotmicroRows,
	quantactRows,
	quantint4Rows,
	w4a8Rows,
)

// TestKernelRegistry_complete fails on a TEXT symbol with no row, a row whose symbol,
// dispatcher or oracle test is gone, and, on this architecture, a row that takes slices
// with no over-read adapter.
func TestKernelRegistry_complete(t *testing.T) {
	kernreg.Complete(t, ".", kernelRows, kernelAdapters())
}

// TestOverRead_everyKernel places every input of every assembly kernel against a guard
// page, at its end and then at its start, at each tail and block boundary, and holds
// the result equal to the call on heap slices.
func TestOverRead_everyKernel(t *testing.T) {
	kernreg.Sweep(t, kernelAdapters())
}

// ptr is the address of x's first element, valid for an empty x as well: a kernel
// called with a zero length gets the address beside its input's guard page.
func ptr[T any](x []T) *T { return unsafe.SliceData(x) }

// upTo returns 0, step, 2·step, … up to and including max, then each of extra; the
// lengths an adapter sweeps.
func upTo(step, max int, extra ...int) []int {
	var out []int
	for n := 0; n <= max; n += step {
		out = append(out, n)
	}
	return append(out, extra...)
}

// seqF32 is n small exact floats, different for each seed, so a sum of products is the
// same in any order and a mixed-up element changes it.
func seqF32(n, seed int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32((i*7+seed*13)%17 - 8)
	}
	return x
}

// seqI8 is n int8s spread over the whole range, different for each seed.
func seqI8(n, seed int) []int8 {
	x := make([]int8, n)
	for i := range x {
		x[i] = int8((i*37 + seed*101) % 256)
	}
	return x
}

// seqU8 is n bytes spread over the whole range, different for each seed.
func seqU8(n, seed int) []uint8 {
	x := make([]uint8, n)
	for i := range x {
		x[i] = uint8((i*53 + seed*29) % 256)
	}
	return x
}
