package cpufeat

import (
	"testing"

	"github.com/townsendmerino/aikit/internal/kernreg"
)

// kernelRows is the inventory of this package's assembly: the CPUID and XGETBV probes,
// which take no slice.
var kernelRows = []kernreg.Row{
	{Symbol: "CPUID", Arch: "amd64", NoSlices: "a CPU feature probe"},
	{Symbol: "XGETBV", Arch: "amd64", NoSlices: "a CPU feature probe"},
}

// TestKernelRegistry_complete fails on a TEXT symbol with no row, or a row whose symbol
// is gone.
func TestKernelRegistry_complete(t *testing.T) {
	kernreg.Complete(t, ".", kernelRows, nil)
}
