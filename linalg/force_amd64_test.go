//go:build amd64

package linalg

import "testing"

// Under a forced-fallback tag the flags behind the dispatchers read false, including the ones derived from another flag at variable-initialisation time
// (hasQ4KAVX2, hasAVX512VNNIVL), which an override of only the root flag would leave true. A normal build asserts nothing about the hardware.
func TestForcedFallbacks_flagsFollowTheTags(t *testing.T) {
	forced := ForcedFallbacks()
	if len(forced) == 0 {
		t.Logf("no fallback forced: flags are as detected (avx2=%v f16c=%v vnni=%v vnniVL=%v popcnt=%v)", hasAVX2, hasF16C, hasAVX512VNNI, hasAVX512VNNIVL, hasPOPCNT)
		return
	}
	for _, f := range forced {
		switch f {
		case "noavx2":
			if hasAVX2 || hasQ4KAVX2 || hasAVX512VNNI || hasAVX512VNNIVL {
				t.Errorf("noavx2: hasAVX2=%v hasQ4KAVX2=%v hasAVX512VNNI=%v hasAVX512VNNIVL=%v, all must be false", hasAVX2, hasQ4KAVX2, hasAVX512VNNI, hasAVX512VNNIVL)
			}
		case "noavx512":
			if hasAVX512VNNI || hasAVX512VNNIVL {
				t.Errorf("noavx512: hasAVX512VNNI=%v hasAVX512VNNIVL=%v, both must be false", hasAVX512VNNI, hasAVX512VNNIVL)
			}
		case "nopopcnt":
			if hasPOPCNT {
				t.Error("nopopcnt: hasPOPCNT is still true")
			}
		default:
			t.Errorf("unexpected forced fallback %q on amd64", f)
		}
	}
}
