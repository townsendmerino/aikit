//go:build arm64

package linalg

import (
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// Under aikit_nodotprod the SDOT flag reads false; a normal build asserts nothing about the hardware.
func TestForcedFallbacks_flagsFollowTheTags(t *testing.T) {
	forced := ForcedFallbacks()
	if len(forced) == 0 {
		t.Logf("no fallback forced: cpufeat.HasDotProd=%v as detected", cpufeat.HasDotProd)
		return
	}
	for _, f := range forced {
		if f != "nodotprod" {
			t.Errorf("unexpected forced fallback %q on arm64", f)
			continue
		}
		if cpufeat.HasDotProd {
			t.Error("nodotprod: cpufeat.HasDotProd is still true")
		}
	}
}
