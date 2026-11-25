package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

func widenAdapters() []kernreg.Adapter {
	return []kernreg.Adapter{
		widenAdapter("cvtBF16ToF32x8AVX2", func() bool { return hasAVX2 }, cvtBF16ToF32x8AVX2),
	}
}
