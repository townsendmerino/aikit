package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

// f16scaleRows holds f16scale_arm64.s; amd64's F16C widen lives in another assembly file
// and is registered with it.
var f16scaleRows = []kernreg.Row{
	{Symbol: "cvtF16ToF32x8NEON", Arch: "arm64", Dispatch: "widenF16", Oracle: "TestWidenF16_exhaustive"},
}
