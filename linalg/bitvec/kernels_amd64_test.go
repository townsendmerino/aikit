package bitvec

import "github.com/townsendmerino/aikit/internal/kernreg"

// kernelAdapters is bitvec's amd64 over-read adapters.
func kernelAdapters() []kernreg.Adapter { return hammingAdapters() }
