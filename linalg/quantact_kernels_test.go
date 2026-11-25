package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

// quantactRows are the activation-quantization passes: abs-max, then round and clamp.
var quantactRows = []kernreg.Row{
	{Symbol: "maxAbsF32AVX2", Arch: "amd64", Dispatch: "maxAbsF32", Oracle: "TestQuantActAVX2Kernels_matchScalar"},
	{Symbol: "quantizeF32AVX2", Arch: "amd64", Dispatch: "quantizeRowScaled", Oracle: "TestQuantActAVX2Kernels_matchScalar"},
	{Symbol: "maxAbsF32NEON", Arch: "arm64", Dispatch: "maxAbsF32", Oracle: "TestQuantActNEONKernels_matchScalar"},
	{Symbol: "quantizeF32NEON", Arch: "arm64", Dispatch: "quantizeRowScaled", Oracle: "TestQuantActNEONKernels_matchScalar"},
}

// quantRow is n activations with every sign, a zero, and values past the clamp once
// multiplied by quantInv.
func quantRow(n, seed int) []float32 {
	x := seqF32(n, seed)
	for i := range x {
		x[i] *= 1.37
	}
	return x
}

// quantInv maps quantRow's largest magnitude a little past 127, so the clamp is reached.
const quantInv = float32(127.0 / 9.0)
