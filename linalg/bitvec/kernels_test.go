package bitvec

import (
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/internal/kernreg"
)

// TestKernelRegistry_complete fails on a TEXT symbol in this package with no row, a row whose symbol, dispatcher or
// oracle test is gone, and, on this architecture, a row that takes slices with no over-read adapter.
func TestKernelRegistry_complete(t *testing.T) {
	kernreg.Complete(t, ".", hammingKernelRows, kernelAdapters())
}

// TestOverRead_everyKernel places every input of every bitvec assembly kernel against a guard page, at its end and then
// at its start, at each tail and block boundary, and holds the result equal to the call on heap slices.
func TestOverRead_everyKernel(t *testing.T) {
	kernreg.Sweep(t, kernelAdapters())
}

// ptr is the address of x's first element, valid for an empty x as well: a kernel called with a zero length gets the
// address beside its input's guard page.
func ptr[T any](x []T) *T { return unsafe.SliceData(x) }

// shapeIndices is 0..n-1: an adapter's Lengths when n indexes a list of shapes.
func shapeIndices(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}
