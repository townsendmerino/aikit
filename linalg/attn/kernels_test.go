package attn

import (
	"math/rand"
	randv2 "math/rand/v2"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/internal/kernreg"
)

// TestKernelRegistry_complete fails on a TEXT symbol in this package with no row, a row whose symbol, dispatcher or
// oracle test is gone, and, on this architecture, a row that takes slices with no over-read adapter.
func TestKernelRegistry_complete(t *testing.T) {
	kernreg.Complete(t, ".", attnRows, kernelAdapters())
}

// TestOverRead_everyKernel places every input of every attn assembly kernel against a guard page, at its end and then
// at its start, at each tail and block boundary, and holds the result equal to the call on heap slices.
func TestOverRead_everyKernel(t *testing.T) {
	kernreg.Sweep(t, kernelAdapters())
}

// ptr is the address of x's first element, valid for an empty x as well: a kernel called with a zero length gets the
// address beside its input's guard page.
func ptr[T any](x []T) *T { return unsafe.SliceData(x) }

// seqF32 is n small exact floats, different for each seed, so a sum of products is the same in any order and a
// mixed-up element changes it.
func seqF32(n, seed int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32((i*7+seed*13)%17 - 8)
	}
	return x
}

// randF is n normal floats, seeded by n.
func randF(n int) []float32 {
	r := rand.New(rand.NewSource(int64(n) + 7))
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

// randVec is n uniform floats in [-1, 1) from rng.
func randVec(rng *randv2.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}
