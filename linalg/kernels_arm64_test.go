package linalg

import (
	"slices"

	"github.com/townsendmerino/aikit/internal/kernreg"
)

// kernelAdapters gathers the arm64 over-read adapters each kernel family declares
// beside its own test file.
func kernelAdapters() []kernreg.Adapter {
	return slices.Concat(
		dotAdapters(),
		kquantAdapters(),
		widenAdapters(),
		f16scaleAdapters(),
		doti8Adapters(),
		dequanti8Adapters(),
		dotmicroAdapters(),
		quantactAdapters(),
		quantint4Adapters(),
		w4a8Adapters(),
	)
}
