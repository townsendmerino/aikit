package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

func widenAdapters() []kernreg.Adapter {
	return []kernreg.Adapter{
		widenAdapter("cvtBF16ToF32x8NEON", nil, cvtBF16ToF32x8NEON),
	}
}
