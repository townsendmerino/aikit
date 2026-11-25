package linalg

import (
	"github.com/townsendmerino/aikit/internal/cpufeat"
	"github.com/townsendmerino/aikit/internal/kernreg"
)

func quantint4Adapters() []kernreg.Adapter {
	return []kernreg.Adapter{{
		Symbol: "quantizeInt4F32AVX2", Needs: func() bool { return cpufeat.HasAVX2 },
		// n a positive multiple of 8 (int4QuantBlock); the loop runs at least once.
		Lengths: upTo(8, 104, 1016, 1024, 1032)[1:],
		Call: func(p *kernreg.Placer, n int) any {
			row := kernreg.Put(p, "row", quantRow(n, 1))
			packed := kernreg.Put(p, "packed", make([]byte, n/2))
			quantizeInt4F32AVX2(ptr(row), ptr(packed), n, 7.0/9.0)
			return packed
		},
	}}
}
