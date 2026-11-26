package attn

import "github.com/townsendmerino/aikit/internal/kernreg"

// kernelAdapters is attn's amd64 over-read adapters.
func kernelAdapters() []kernreg.Adapter { return attnAdapters() }
