package bitvec

import "github.com/townsendmerino/aikit/internal/kernreg"

func hammingAdapters() []kernreg.Adapter {
	// hammingRows dispatches only the two specialised word counts; the general entry in
	// the same symbol is not reached (hamming_arm64.go:hammingRows).
	return []kernreg.Adapter{
		hammingAdapter("hammingNEON", nil, []int{4, 12}, hammingNEON),
	}
}
