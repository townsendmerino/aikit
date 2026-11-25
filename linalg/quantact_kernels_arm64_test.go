package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

func quantactAdapters() []kernreg.Adapter {
	// n a multiple of 16 and > 0: both loops run at least once.
	lengths := upTo(16, 160, 1008, 1024, 1040)[1:]
	return []kernreg.Adapter{
		{
			Symbol: "maxAbsF32NEON", Lengths: lengths,
			Call: func(p *kernreg.Placer, n int) any {
				row := kernreg.Put(p, "row", quantRow(n, 1))
				return maxAbsF32NEON(ptr(row), n)
			},
		},
		{
			Symbol: "quantizeF32NEON", Lengths: lengths,
			Call: func(p *kernreg.Placer, n int) any {
				row := kernreg.Put(p, "row", quantRow(n, 1))
				q := kernreg.Put(p, "q", make([]int8, n))
				quantizeF32NEON(ptr(row), ptr(q), n, quantInv)
				return q
			},
		},
	}
}
