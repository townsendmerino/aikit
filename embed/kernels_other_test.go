//go:build !amd64 && !arm64

package embed

import "github.com/townsendmerino/aikit/internal/kernreg"

// kernelAdapters is empty where embed has no assembly.
func kernelAdapters() []kernreg.Adapter { return nil }
