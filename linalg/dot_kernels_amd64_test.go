package linalg

import (
	"github.com/townsendmerino/aikit/internal/cpufeat"
	"github.com/townsendmerino/aikit/internal/kernreg"
)

func dotAdapters() []kernreg.Adapter {
	avx2 := func() bool { return cpufeat.HasAVX2 }
	return []kernreg.Adapter{
		{
			Symbol: "dotFMA", Needs: avx2,
			// 32 floats per main-loop pass, then 8, then scalar.
			Lengths: upTo(1, 72, 255, 256, 257, 1023, 1024, 1025),
			Call: func(p *kernreg.Placer, n int) any {
				a, b := kernreg.Put(p, "a", seqF32(n, 1)), kernreg.Put(p, "b", seqF32(n, 2))
				return dotFMA(ptr(a), ptr(b), n)
			},
		},
		{
			Symbol: "dotFMA4", Needs: avx2,
			// n a multiple of 4: 8 floats per pass, then one 4-float step.
			Lengths: upTo(4, 72, 1020, 1024, 1028),
			Call: func(p *kernreg.Placer, n int) any {
				a := kernreg.Put(p, "a", seqF32(n, 1))
				b0, b1 := kernreg.Put(p, "b0", seqF32(n, 2)), kernreg.Put(p, "b1", seqF32(n, 3))
				b2, b3 := kernreg.Put(p, "b2", seqF32(n, 4)), kernreg.Put(p, "b3", seqF32(n, 5))
				out := kernreg.Put(p, "out", make([]float32, 4))
				dotFMA4(ptr(a), ptr(b0), ptr(b1), ptr(b2), ptr(b3), n, (*[4]float32)(out))
				return out
			},
		},
		{
			Symbol: "dotFMA8", Needs: avx2,
			Lengths: upTo(4, 72, 1020, 1024, 1028),
			Call: func(p *kernreg.Placer, n int) any {
				a := kernreg.Put(p, "a", seqF32(n, 1))
				var b [8][]float32
				for r := range b {
					b[r] = kernreg.Put(p, "b"+string(rune('0'+r)), seqF32(n, r+2))
				}
				out := kernreg.Put(p, "out", make([]float32, 8))
				dotFMA8(ptr(a), ptr(b[0]), ptr(b[1]), ptr(b[2]), ptr(b[3]), ptr(b[4]), ptr(b[5]), ptr(b[6]), ptr(b[7]), n, (*[8]float32)(out))
				return out
			},
		},
		{
			Symbol: "dotI8AVX2", Needs: avx2,
			// n a multiple of 16.
			Lengths: upTo(16, 160, 1008, 1024, 1040),
			Call: func(p *kernreg.Placer, n int) any {
				a, b := kernreg.Put(p, "a", seqI8(n, 1)), kernreg.Put(p, "b", seqI8(n, 2))
				return dotI8AVX2(ptr(a), ptr(b), n)
			},
		},
	}
}
