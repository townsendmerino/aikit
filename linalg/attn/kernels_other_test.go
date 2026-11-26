//go:build !amd64 && !arm64

package attn

import "github.com/townsendmerino/aikit/internal/kernreg"

// kernelAdapters is empty where attn has no assembly.
func kernelAdapters() []kernreg.Adapter { return nil }
