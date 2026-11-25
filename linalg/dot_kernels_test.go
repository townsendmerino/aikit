package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

var dotRows = []kernreg.Row{
	{Symbol: "dotFMA", Arch: "amd64", Dispatch: "dotNEON", Oracle: "TestAVX2_dotFMA_matchesGeneric"},
	{Symbol: "dotFMA4", Arch: "amd64", Dispatch: "dotNEON4x4", Oracle: "TestAVX2_blockedKernels_oddN4"},
	{Symbol: "dotFMA8", Arch: "amd64", Dispatch: "dotNEON8x4", Oracle: "TestAVX2_blockedKernels_oddN4"},
	{Symbol: "dotI8AVX2", Arch: "amd64", Dispatch: "dotI8", Oracle: "TestAVX512VNNI_dotI8_dispatchesThroughEveryTier"},
	{Symbol: "dotNEON4", Arch: "arm64", Dispatch: "dotNEON", Oracle: "TestDotF32_matchesScalar"},
	{Symbol: "dotNEON4x4", Arch: "arm64", Dispatch: "Dot4x4", Oracle: "TestDot4x4_matchesScalar"},
	{Symbol: "dotNEON8x4", Arch: "arm64", Dispatch: "Dot8x4", Oracle: "TestDot8x4_matchesScalar"},
}
