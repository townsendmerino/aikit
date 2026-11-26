package elem

import (
	"github.com/townsendmerino/aikit/internal/cpufeat"
	"github.com/townsendmerino/aikit/internal/kernreg"
)

func expAdapters() []kernreg.Adapter {
	avx2 := func() bool { return cpufeat.HasAVX2 }
	// n a multiple of 8.
	return []kernreg.Adapter{
		unaryF32("expF32ContractAVX2", avx2, 8, expF32ContractAVX2),
		unaryF32("tanhF32ContractAVX2", avx2, 8, tanhF32ContractAVX2),
		unaryF32("erfF32ContractAVX2", avx2, 8, erfF32ContractAVX2),
	}
}
