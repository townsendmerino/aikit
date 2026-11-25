package linalg

import (
	"github.com/townsendmerino/aikit/internal/cpufeat"
	"github.com/townsendmerino/aikit/internal/kernreg"
)

func dequanti8Adapters() []kernreg.Adapter {
	return []kernreg.Adapter{{
		Symbol: "dequantI8AVX2", Needs: func() bool { return cpufeat.HasAVX2 },
		// n a multiple of 8: 32 per pass, then 8 at a time, each checked first.
		Lengths: upTo(8, 104, 1016, 1024, 1032),
		Call: func(p *kernreg.Placer, n int) any {
			q := kernreg.Put(p, "q", seqI8(n, 1))
			dst := kernreg.Put(p, "dst", make([]float32, n))
			dequantI8AVX2(ptr(q), ptr(dst), n, 0.03125)
			return dst
		},
	}}
}
