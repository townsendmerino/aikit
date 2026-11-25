//go:build !arm64

package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

// toArchExempt names the arm64-only row4 repack entry points, which do not exist on this
// architecture; tiny_other_arm64_test.go sweeps them on arm64.
var toArchExempt = map[string]string{
	"RepackInt4Row4Quad":           toArm64Only,
	"RepackInt4Row4ScalesQuad":     toArm64Only,
	"RepackInt4Row4ScalesQuadF16":  toArm64Only,
	"RepackW4A8Row4":               toArm64Only,
	"RepackW4A8Row4Deshared":       toArm64Only,
	"RepackW4A8Row4DesharedScales": toArm64Only,
	"RepackW4A8Row4Scales":         toArm64Only,
	"RepackW4A8Row4ScalesF16":      toArm64Only,
}

const toArm64Only = "arm64 only; swept by the arm64 build"

func toArchAdapters() []kernreg.Adapter { return nil }
