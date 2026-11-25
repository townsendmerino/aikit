package bitvec

import "github.com/townsendmerino/aikit/internal/kernreg"

// kernelAdapters is bitvec's arm64 over-read adapters.
func kernelAdapters() []kernreg.Adapter { return hammingAdapters() }
