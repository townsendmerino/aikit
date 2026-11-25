package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

func dotAdapters() []kernreg.Adapter {
	// n is n4, the count of 4-float groups.
	lengths := upTo(1, 40, 63, 64, 65, 255, 256, 257)
	return []kernreg.Adapter{
		{
			Symbol: "dotNEON4", Lengths: lengths,
			Call: func(p *kernreg.Placer, n4 int) any {
				a, b := kernreg.Put(p, "a", seqF32(4*n4, 1)), kernreg.Put(p, "b", seqF32(4*n4, 2))
				sums := kernreg.Put(p, "sums", make([]float32, 4))
				dotNEON4(ptr(a), ptr(b), n4, (*[4]float32)(sums))
				return sums
			},
		},
		{
			Symbol: "dotNEON4x4", Lengths: lengths,
			Call: func(p *kernreg.Placer, n4 int) any {
				n := 4 * n4
				a := kernreg.Put(p, "a", seqF32(n, 1))
				b0, b1 := kernreg.Put(p, "b0", seqF32(n, 2)), kernreg.Put(p, "b1", seqF32(n, 3))
				b2, b3 := kernreg.Put(p, "b2", seqF32(n, 4)), kernreg.Put(p, "b3", seqF32(n, 5))
				sums := kernreg.Put(p, "sums", make([]float32, 16))
				dotNEON4x4(ptr(a), ptr(b0), ptr(b1), ptr(b2), ptr(b3), n4, (*[16]float32)(sums))
				return sums
			},
		},
		{
			Symbol: "dotNEON8x4", Lengths: lengths,
			Call: func(p *kernreg.Placer, n4 int) any {
				n := 4 * n4
				a := kernreg.Put(p, "a", seqF32(n, 1))
				var b [8][]float32
				for r := range b {
					b[r] = kernreg.Put(p, "b"+string(rune('0'+r)), seqF32(n, r+2))
				}
				sums := kernreg.Put(p, "sums", make([]float32, 32))
				dotNEON8x4(ptr(a), ptr(b[0]), ptr(b[1]), ptr(b[2]), ptr(b[3]), ptr(b[4]), ptr(b[5]), ptr(b[6]), ptr(b[7]), n4, (*[32]float32)(sums))
				return sums
			},
		},
	}
}
