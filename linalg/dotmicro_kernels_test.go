package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

// dotmicroRows are the f32 register microkernels with more than one a-row.
var dotmicroRows = []kernreg.Row{
	{Symbol: "dotFMA3x4", Arch: "amd64", Dispatch: "blockRows3x4", Oracle: "TestDot3x4_matchesDotFMA8"},
	{Symbol: "dotNEON2x8", Arch: "arm64", Dispatch: "Dot2x8", Oracle: "TestDot2x8_equalsDot8x4"},
}
