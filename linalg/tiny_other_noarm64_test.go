//go:build !arm64

package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

// toArchExempt names the arm64-only row4 repack entry points, which do not exist on this
// architecture; tiny_other_arm64_test.go sweeps them on arm64.
var toArchExempt = map[string]string{
	"RepackW4A8Row4":          toArm64Only,
	"RepackW4A8Row4ScalesF16": toArm64Only,
}

const toArm64Only = "arm64 only; swept by the arm64 build"

func toArchAdapters() []kernreg.Adapter { return nil }
