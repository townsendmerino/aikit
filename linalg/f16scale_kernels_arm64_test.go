package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

func f16scaleAdapters() []kernreg.Adapter {
	return []kernreg.Adapter{widenAdapter("cvtF16ToF32x8NEON", nil, cvtF16ToF32x8NEON)}
}
