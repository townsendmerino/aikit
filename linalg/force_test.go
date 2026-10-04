package linalg

import (
	"os"
	"slices"
	"strings"
	"testing"
)

// TestForcedFallbacks_expected is the guard against a CI job that passes -tags and forces nothing: a tag that matches no file compiles and runs green. A job
// sets AIKIT_EXPECT_FORCED to the fallbacks it asked for ("noavx2", "noavx512,nopopcnt", ...) or to "none", and this fails unless ForcedFallbacks says the same.
// Unset, it checks nothing (an ordinary developer run), and says so.
func TestForcedFallbacks_expected(t *testing.T) {
	want, set := os.LookupEnv("AIKIT_EXPECT_FORCED")
	if !set {
		t.Skip("AIKIT_EXPECT_FORCED unset: not asserting which fallbacks this build forces")
	}
	var wantList []string
	if want != "none" && want != "" {
		wantList = strings.Split(want, ",")
	}
	got := ForcedFallbacks()
	slices.Sort(wantList)
	slices.Sort(got)
	if !slices.Equal(got, wantList) {
		t.Fatalf("ForcedFallbacks() = %v, the job expected %v: a -tags value that matched no file forces nothing and would pass silently", got, wantList)
	}
}

func TestForcedFallbacks_returnsACopy(t *testing.T) {
	a := ForcedFallbacks()
	if len(a) == 0 {
		return
	}
	a[0] = "mutated"
	if ForcedFallbacks()[0] == "mutated" {
		t.Fatal("ForcedFallbacks returned its backing slice")
	}
}
