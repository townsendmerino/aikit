//go:build arm64

package linalg

import (
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// Repair on arm64: a kernel that is wrong only while DotProd is on makes SelfCheck turn DotProd off, after which the base NEON kernels agree.
func TestSelfCheck_repairStepsDownDotProd(t *testing.T) {
	if !cpufeat.HasDotProd {
		t.Skip("DotProd is already off here (a forced build or a core without it): there is no tier to step down")
	}
	t.Cleanup(func() { cpufeat.HasDotProd = true; scDotI8 = dotI8 })
	scDotI8 = func(a, b []int8) int32 {
		s := dotI8(a, b)
		if cpufeat.HasDotProd && len(a) == 17 {
			s += 3
		}
		return s
	}
	rep := SelfCheck(true)
	if rep.OK() || !slices.Equal(rep.Disabled, []string{"dotprod"}) || len(rep.Remaining) != 0 || cpufeat.HasDotProd {
		t.Fatalf("want dotprod disabled and the kernels agreeing afterwards: %+v cpufeat.HasDotProd=%v", rep, cpufeat.HasDotProd)
	}
}
