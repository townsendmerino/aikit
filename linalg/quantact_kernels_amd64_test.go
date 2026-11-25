package linalg

import (
	"github.com/townsendmerino/aikit/internal/cpufeat"
	"github.com/townsendmerino/aikit/internal/kernreg"
)

func quantactAdapters() []kernreg.Adapter {
	avx2 := func() bool { return cpufeat.HasAVX2 }
	// n a multiple of 8 and > 0 (the dispatchers call only with n > 0).
	lengths := upTo(8, 104, 1016, 1024, 1032)[1:]
	return []kernreg.Adapter{
		{
			Symbol: "maxAbsF32AVX2", Needs: avx2,
			// 32 per pass, then 8 at a time.
			Lengths: lengths,
			Call: func(p *kernreg.Placer, n int) any {
				row := kernreg.Put(p, "row", quantRow(n, 1))
				return maxAbsF32AVX2(ptr(row), n)
			},
		},
		{
			Symbol: "quantizeF32AVX2", Needs: avx2,
			Lengths: lengths,
			Call: func(p *kernreg.Placer, n int) any {
				row := kernreg.Put(p, "row", quantRow(n, 1))
				q := kernreg.Put(p, "q", make([]int8, n))
				quantizeF32AVX2(ptr(row), ptr(q), n, quantInv)
				return q
			},
		},
	}
}
