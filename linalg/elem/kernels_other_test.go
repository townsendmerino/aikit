//go:build !amd64 && !arm64

package elem

import "github.com/townsendmerino/aikit/internal/kernreg"

// kernelAdapters is empty where elem has no assembly.
func kernelAdapters() []kernreg.Adapter { return nil }
