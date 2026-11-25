package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

func dotmicroAdapters() []kernreg.Adapter {
	return []kernreg.Adapter{{
		Symbol: "dotNEON2x8",
		// n is n4, groups of 4 floats, one per step; 0 skips the loop.
		Lengths: upTo(1, 40, 255, 256, 257),
		Call: func(p *kernreg.Placer, n4 int) any {
			n := 4 * n4
			a0, a1 := kernreg.Put(p, "a0", seqF32(n, 1)), kernreg.Put(p, "a1", seqF32(n, 2))
			var b [8][]float32
			for r := range b {
				b[r] = kernreg.Put(p, "b"+string(rune('0'+r)), seqF32(n, r+3))
			}
			sums := kernreg.Put(p, "sums", make([]float32, 64))
			dotNEON2x8(ptr(a0), ptr(a1), ptr(b[0]), ptr(b[1]), ptr(b[2]), ptr(b[3]), ptr(b[4]), ptr(b[5]), ptr(b[6]), ptr(b[7]), n4, (*[64]float32)(sums))
			return sums
		},
	}}
}
