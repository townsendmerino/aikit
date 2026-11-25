package linalg

import (
	"slices"

	"github.com/townsendmerino/aikit/internal/kernreg"
)

// kernelAdapters gathers the amd64 over-read adapters each kernel family declares
// beside its own test file.
func kernelAdapters() []kernreg.Adapter {
	return slices.Concat(
		dotAdapters(),
		kquantAdapters(),
		attnAdapters(),
		expAdapters(),
		widenAdapters(),
		f16scaleAdapters(),
		hammingAdapters(),
		doti8Adapters(),
		dequanti8Adapters(),
		dotmicroAdapters(),
		quantactAdapters(),
		quantint4Adapters(),
		w4a8Adapters(),
	)
}
