package bitvec

import (
	"fmt"

	"github.com/townsendmerino/aikit/internal/kernreg"
)

var hammingKernelRows = []kernreg.Row{
	{Symbol: "hammingPOPCNT", Arch: "amd64", Dispatch: "hammingRows", Oracle: "TestHammingRows_matchesScalar"},
	{Symbol: "hammingNEON", Arch: "arm64", Dispatch: "hammingRows", Oracle: "TestHammingRows_matchesScalar"},
}

// hammingAdapter sweeps n code rows of each word count in words (n >= 1 and words >= 1,
// as hammingRows passes them).
func hammingAdapter(symbol string, needs func() bool, words []int, kernel func(q, codes *uint64, words, n int, dst *uint16)) kernreg.Adapter {
	type shape struct{ words, n int }
	var shapes []shape
	for _, w := range words {
		for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33} {
			shapes = append(shapes, shape{w, n})
		}
	}
	return kernreg.Adapter{
		Symbol: symbol, Needs: needs, Lengths: shapeIndices(len(shapes)),
		Call: func(p *kernreg.Placer, i int) any {
			s := shapes[i]
			name := fmt.Sprintf("[words=%d n=%d]", s.words, s.n)
			q := kernreg.Put(p, "q"+name, seqU64(s.words, 1))
			codes := kernreg.Put(p, "codes"+name, seqU64(s.words*s.n, 2))
			dst := kernreg.Put(p, "dst"+name, make([]uint16, s.n))
			kernel(ptr(q), ptr(codes), s.words, s.n, ptr(dst))
			return dst
		},
	}
}

// seqU64 is n words with varied bit counts, different for each seed.
func seqU64(n, seed int) []uint64 {
	x := make([]uint64, n)
	for i := range x {
		v := uint64(i*0x9E3779B1+seed*0x85EBCA77) * 0xC2B2AE3D27D4EB4F
		x[i] = v ^ v>>29
	}
	return x
}
