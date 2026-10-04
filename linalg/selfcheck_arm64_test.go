//go:build arm64

package linalg

import (
	"slices"
	"testing"
)

// Repair on arm64: a kernel that is wrong only while DotProd is on makes SelfCheck turn DotProd off, after which the base NEON kernels agree.
func TestSelfCheck_repairStepsDownDotProd(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd is already off here (a forced build or a core without it): there is no tier to step down")
	}
	t.Cleanup(func() { hasDotProd = true; scDotI8 = dotI8 })
	scDotI8 = func(a, b []int8) int32 {
		s := dotI8(a, b)
		if hasDotProd && len(a) == 17 {
			s += 3
		}
		return s
	}
	rep := SelfCheck(true)
	if rep.OK() || !slices.Equal(rep.Disabled, []string{"dotprod"}) || len(rep.Remaining) != 0 || hasDotProd {
		t.Fatalf("want dotprod disabled and the kernels agreeing afterwards: %+v hasDotProd=%v", rep, hasDotProd)
	}
}
