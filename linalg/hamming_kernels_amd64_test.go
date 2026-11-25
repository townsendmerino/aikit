package linalg

import (
	"github.com/townsendmerino/aikit/internal/cpufeat"
	"github.com/townsendmerino/aikit/internal/kernreg"
)

func hammingAdapters() []kernreg.Adapter {
	// The POPCNTQ loop takes any word count, four accumulators at a time.
	return []kernreg.Adapter{
		hammingAdapter("hammingPOPCNT", func() bool { return cpufeat.HasPOPCNT }, []int{1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 16, 17}, hammingPOPCNT),
	}
}
