package elem

import "github.com/townsendmerino/aikit/internal/kernreg"

var expRows = []kernreg.Row{
	{Symbol: "expF32ContractAVX2", Arch: "amd64", Dispatch: "expContractImpl", Oracle: "TestExpF32AVX2_bitIdenticalToContract"},
	{Symbol: "tanhF32ContractAVX2", Arch: "amd64", Dispatch: "geluTanhContractImpl", Oracle: "TestTanhF32AVX2_bitIdenticalToContract"},
	{Symbol: "erfF32ContractAVX2", Arch: "amd64", Dispatch: "geluContractImpl", Oracle: "TestErfF32AVX2_bitIdenticalToContract"},
	{Symbol: "expF32ContractNEON", Arch: "arm64", Dispatch: "expContractImpl", Oracle: "TestExpF32NEON_bitIdenticalToContract"},
	{Symbol: "expBiasSumF32NEON", Arch: "arm64", Dispatch: "softmaxRowContractNEONWithMax", Oracle: "TestSoftmaxContractNEON_bitIdenticalToScalar"},
	{Symbol: "siluF32ContractNEON", Arch: "arm64", Dispatch: "siluContractImpl", Oracle: "TestSiLUContractNEON_bitIdenticalToScalar"},
	{Symbol: "tanhF32ContractNEON", Arch: "arm64", Dispatch: "geluTanhContractImpl", Oracle: "TestTanhContractNEON_bitIdenticalToScalar"},
	{Symbol: "erfF32ContractNEON", Arch: "arm64", Dispatch: "geluContractImpl", Oracle: "TestErfContractNEON_bitIdenticalToScalar"},
}

// unaryF32 is the adapter of a dst/src elementwise kernel over n elements, n a positive
// multiple of step (its dispatcher never passes 0).
func unaryF32(symbol string, needs func() bool, step int, kernel func(dst, src *float32, n int)) kernreg.Adapter {
	return kernreg.Adapter{
		Symbol: symbol, Needs: needs,
		Lengths: upTo(step, 12*step, 1024-step, 1024, 1024+step)[1:],
		Call: func(p *kernreg.Placer, n int) any {
			src := kernreg.Put(p, "src", seqF32(n, 1))
			dst := kernreg.Put(p, "dst", make([]float32, n))
			kernel(ptr(dst), ptr(src), n)
			return dst
		},
	}
}
