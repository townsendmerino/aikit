package elem

import (
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/internal/kernreg"
)

// TestKernelRegistry_complete fails on a TEXT symbol in this package with no row, a row whose symbol, dispatcher or
// oracle test is gone, and, on this architecture, a row that takes slices with no over-read adapter.
func TestKernelRegistry_complete(t *testing.T) {
	kernreg.Complete(t, ".", expRows, kernelAdapters())
}

// TestOverRead_everyKernel places every input of every elem assembly kernel against a guard page, at its end and then
// at its start, at each tail and block boundary, and holds the result equal to the call on heap slices.
func TestOverRead_everyKernel(t *testing.T) {
	kernreg.Sweep(t, kernelAdapters())
}

// ptr is the address of x's first element, valid for an empty x as well: a kernel called with a zero length gets the
// address beside its input's guard page.
func ptr[T any](x []T) *T { return unsafe.SliceData(x) }

// upTo returns 0, step, 2·step, … up to and including max, then each of extra; the lengths an adapter sweeps.
func upTo(step, max int, extra ...int) []int {
	var out []int
	for n := 0; n <= max; n += step {
		out = append(out, n)
	}
	return append(out, extra...)
}

// seqF32 is n small exact floats, different for each seed, so a sum of products is the same in any order and a
// mixed-up element changes it.
func seqF32(n, seed int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32((i*7+seed*13)%17 - 8)
	}
	return x
}
