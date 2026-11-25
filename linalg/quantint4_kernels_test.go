package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

var quantint4Rows = []kernreg.Row{
	{Symbol: "quantizeInt4F32AVX2", Arch: "amd64", Dispatch: "quantizeInt4Block", Oracle: "TestQuantizeInt4Block_matchesScalar"},
	{Symbol: "quantizeInt4F32NEON", Arch: "arm64", Dispatch: "quantizeInt4Block", Oracle: "TestQuantizeInt4Block_matchesScalar"},
}
