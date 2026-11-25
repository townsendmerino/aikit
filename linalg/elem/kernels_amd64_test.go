package elem

import "github.com/townsendmerino/aikit/internal/kernreg"

// kernelAdapters is elem's amd64 over-read adapters.
func kernelAdapters() []kernreg.Adapter { return expAdapters() }
