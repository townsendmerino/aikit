//go:build !arm64

package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

// tmArchExempt is empty: the arm64-only matmuls are unexported (LS6), so the completeness
// check does not count them; tiny_matmul_arm64_test.go still sweeps them on arm64.
var tmArchExempt = map[string]string{}

func tmArchAdapters() []kernreg.Adapter { return nil }
