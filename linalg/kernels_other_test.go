//go:build !amd64 && !arm64

package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

// kernelAdapters is empty where linalg has no assembly.
func kernelAdapters() []kernreg.Adapter { return nil }
