package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

var widenRows = []kernreg.Row{
	{Symbol: "cvtBF16ToF32x8AVX2", Arch: "amd64", Dispatch: "widenBF16", Oracle: "TestWidenBF16_everyLengthAndNoOverrun"},
	{Symbol: "cvtBF16ToF32x8NEON", Arch: "arm64", Dispatch: "widenBF16", Oracle: "TestWidenBF16_everyLengthAndNoOverrun"},
}

// widenAdapter is the adapter of a 16-bit to f32 widen of n8 blocks of eight; its
// dispatcher passes n8 >= 1.
func widenAdapter(symbol string, needs func() bool, kernel func(dst *float32, src *uint16, n8 int)) kernreg.Adapter {
	return kernreg.Adapter{
		Symbol: symbol, Needs: needs,
		Lengths: upTo(1, 20, 63, 64, 65, 127, 128, 129)[1:],
		Call: func(p *kernreg.Placer, n8 int) any {
			src := kernreg.Put(p, "src", seqF16(8*n8, 1))
			dst := kernreg.Put(p, "dst", make([]float32, 8*n8))
			kernel(ptr(dst), ptr(src), n8)
			return dst
		},
	}
}
