//go:build arm64

package linalg

import "testing"

// Under aikit_nodotprod the SDOT flag reads false; a normal build asserts nothing about the hardware.
func TestForcedFallbacks_flagsFollowTheTags(t *testing.T) {
	forced := ForcedFallbacks()
	if len(forced) == 0 {
		t.Logf("no fallback forced: hasDotProd=%v as detected", hasDotProd)
		return
	}
	for _, f := range forced {
		if f != "nodotprod" {
			t.Errorf("unexpected forced fallback %q on arm64", f)
			continue
		}
		if hasDotProd {
			t.Error("nodotprod: hasDotProd is still true")
		}
	}
}
