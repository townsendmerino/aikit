//go:build !amd64 && !arm64

package bitvec

import "github.com/townsendmerino/aikit/internal/kernreg"

// kernelAdapters is empty where bitvec has no assembly.
func kernelAdapters() []kernreg.Adapter { return nil }
