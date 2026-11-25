package embed

import "github.com/townsendmerino/aikit/internal/kernreg"

var ggufDequantRows = []kernreg.Row{
	{Symbol: "dequantQ4KBodyAVX2", Arch: "amd64", Dispatch: "dequantQ4KBlock", Oracle: "TestDequantQ4KBlock_matchesScalar"},
	{Symbol: "dequantQ6KBodyAVX2", Arch: "amd64", Dispatch: "dequantQ6KBlock", Oracle: "TestDequantQ6KBlock_matchesScalar"},
	{Symbol: "dequantQ4KBodyNEON", Arch: "arm64", Dispatch: "dequantQ4KBlock", Oracle: "TestDequantQ4KBlock_matchesScalar"},
	{Symbol: "dequantQ6KBodyNEON", Arch: "arm64", Dispatch: "dequantQ6KBlock", Oracle: "TestDequantQ6KBlock_matchesScalar"},
}

// The bodies each decode one fixed 256-weight super-block, so an adapter's n is not a
// length: it picks one of a few seeds for the block's bytes and multipliers.
var ggufDequantSeeds = []int{0, 1, 2}

// q4kBodyBytes is what a Q4_K body kernel reads: the block's 128 nibble bytes, which end
// where the 144-byte block ends; q6kBodyBytes is a Q6_K body's ql[128] then qh[64], the
// first 192 of the block's 210 bytes.
const (
	q4kBodyBytes = 128
	q6kBodyBytes = 192
)
