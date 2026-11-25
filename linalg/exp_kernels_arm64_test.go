package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

func expAdapters() []kernreg.Adapter {
	// n a multiple of 4.
	return []kernreg.Adapter{
		unaryF32("expF32ContractNEON", nil, 4, expF32ContractNEON),
		unaryF32("siluF32ContractNEON", nil, 4, siluF32ContractNEON),
		unaryF32("tanhF32ContractNEON", nil, 4, tanhF32ContractNEON),
		unaryF32("erfF32ContractNEON", nil, 4, erfF32ContractNEON),
		{
			// src's values minus their max, as softmaxRowContractNEONWithMax passes them.
			Symbol: "expBiasSumF32NEON", Lengths: upTo(4, 48, 1020, 1024, 1028)[1:],
			Call: func(p *kernreg.Placer, n int) any {
				src := kernreg.Put(p, "src", seqF32(n, 1))
				dst := kernreg.Put(p, "dst", make([]float32, n))
				partials := kernreg.Put(p, "partials", make([]float64, 4))
				expBiasSumF32NEON(ptr(dst), ptr(src), n, -8, ptr(partials))
				return []any{dst, partials}
			},
		},
	}
}
