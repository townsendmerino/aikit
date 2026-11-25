//go:build arm64 && windows

package linalg

import (
	"os"
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// TestDetectDotProd_windowsMatchesTheExpectation: the probe is an OS call, so the only test of it is on a Windows ARM machine, and on a runner it must be told what to expect or it checks nothing. CI's
// windows-11-arm job (Azure Cobalt 100, Neoverse N2, which has DotProd) sets AIKIT_EXPECT_DOTPROD=yes; a machine known to lack it can set "no". Unset, it logs what it found and says it asserted nothing.
func TestDetectDotProd_windowsMatchesTheExpectation(t *testing.T) {
	got := cpufeat.DetectDotProd()
	t.Logf("IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) = %v; cpufeat.HasDotProd = %v; ActiveKernels = %+v", got, cpufeat.HasDotProd, ActiveKernels())
	if got != cpufeat.HasDotProd {
		t.Errorf("cpufeat.DetectDotProd() = %v but cpufeat.HasDotProd = %v: the package-level flag was not set from the probe", got, cpufeat.HasDotProd)
	}
	want, set := os.LookupEnv("AIKIT_EXPECT_DOTPROD")
	if !set {
		t.Skip("AIKIT_EXPECT_DOTPROD unset: not asserting whether this Windows ARM core has DotProd")
	}
	switch want {
	case "yes":
		if !got {
			t.Fatal("this core was expected to have DotProd (AIKIT_EXPECT_DOTPROD=yes) but Windows reports it absent: the probe, or the runner, is not what the job assumed")
		}
	case "no":
		if got {
			t.Fatal("this core was expected to lack DotProd (AIKIT_EXPECT_DOTPROD=no) but Windows reports it present")
		}
	default:
		t.Fatalf("AIKIT_EXPECT_DOTPROD=%q: want yes or no", want)
	}
}
