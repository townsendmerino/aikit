//go:build amd64

package linalg

import (
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// Under a forced-fallback tag the flags behind the dispatchers read false, including the ones derived from another flag at variable-initialisation time
// (cpufeat.HasQ4KAVX2, cpufeat.HasAVX512VNNIVL), which an override of only the root flag would leave true. A normal build asserts nothing about the hardware.
func TestForcedFallbacks_flagsFollowTheTags(t *testing.T) {
	forced := ForcedFallbacks()
	if len(forced) == 0 {
		t.Logf("no fallback forced: flags are as detected (avx2=%v f16c=%v vnni=%v vnniVL=%v popcnt=%v)", cpufeat.HasAVX2, cpufeat.HasF16C, cpufeat.HasAVX512VNNI, cpufeat.HasAVX512VNNIVL, cpufeat.HasPOPCNT)
		return
	}
	for _, f := range forced {
		switch f {
		case "noavx2":
			if cpufeat.HasAVX2 || cpufeat.HasQ4KAVX2 || cpufeat.HasAVX512VNNI || cpufeat.HasAVX512VNNIVL {
				t.Errorf("noavx2: cpufeat.HasAVX2=%v cpufeat.HasQ4KAVX2=%v cpufeat.HasAVX512VNNI=%v cpufeat.HasAVX512VNNIVL=%v, all must be false", cpufeat.HasAVX2, cpufeat.HasQ4KAVX2, cpufeat.HasAVX512VNNI, cpufeat.HasAVX512VNNIVL)
			}
		case "noavx512":
			if cpufeat.HasAVX512VNNI || cpufeat.HasAVX512VNNIVL {
				t.Errorf("noavx512: cpufeat.HasAVX512VNNI=%v cpufeat.HasAVX512VNNIVL=%v, both must be false", cpufeat.HasAVX512VNNI, cpufeat.HasAVX512VNNIVL)
			}
		case "nopopcnt":
			if cpufeat.HasPOPCNT {
				t.Error("nopopcnt: cpufeat.HasPOPCNT is still true")
			}
		default:
			t.Errorf("unexpected forced fallback %q on amd64", f)
		}
	}
}
