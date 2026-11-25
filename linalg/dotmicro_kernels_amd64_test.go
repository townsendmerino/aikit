package linalg

import (
	"github.com/townsendmerino/aikit/internal/cpufeat"
	"github.com/townsendmerino/aikit/internal/kernreg"
)

func dotmicroAdapters() []kernreg.Adapter {
	return []kernreg.Adapter{{
		Symbol: "dotFMA3x4", Needs: func() bool { return cpufeat.HasAVX2 },
		// n a multiple of 4: 8 floats per pass, then one 4-float step.
		Lengths: upTo(4, 72, 1020, 1024, 1028),
		Call: func(p *kernreg.Placer, n int) any {
			a0, a1, a2 := kernreg.Put(p, "a0", seqF32(n, 1)), kernreg.Put(p, "a1", seqF32(n, 2)), kernreg.Put(p, "a2", seqF32(n, 3))
			b0, b1 := kernreg.Put(p, "b0", seqF32(n, 4)), kernreg.Put(p, "b1", seqF32(n, 5))
			b2, b3 := kernreg.Put(p, "b2", seqF32(n, 6)), kernreg.Put(p, "b3", seqF32(n, 7))
			out := kernreg.Put(p, "out", make([]float32, 12))
			dotFMA3x4(ptr(a0), ptr(a1), ptr(a2), ptr(b0), ptr(b1), ptr(b2), ptr(b3), n, (*[12]float32)(out))
			return out
		},
	}}
}
