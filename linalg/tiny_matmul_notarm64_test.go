//go:build !arm64

package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

// tmArchExempt names the arm64-only entry points, which this architecture's test binary
// cannot call; tiny_matmul_arm64_test.go sweeps them on arm64.
var tmArchExempt = map[string]string{
	"MatmulBTW4A8Row4Into":         "arm64 only; swept on arm64",
	"MatmulBTW4A8Row4TileInto":     "arm64 only; swept on arm64",
	"MatmulBTW4A8Row4PrefetchInto": "arm64 only; swept on arm64",
	"MatmulBTW4A8Row4DesharedInto": "arm64 only; swept on arm64",
}

func tmArchAdapters() []kernreg.Adapter { return nil }
