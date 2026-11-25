package embed

import "github.com/townsendmerino/aikit/internal/kernreg"

func ggufDequantAdapters() []kernreg.Adapter {
	return []kernreg.Adapter{
		{
			Symbol: "dequantQ4KBodyNEON", Lengths: ggufDequantSeeds,
			Call: func(p *kernreg.Placer, seed int) any {
				qs := kernreg.Put(p, "qs", seqU8(q4kBodyBytes, seed))
				out := kernreg.Put(p, "out", make([]float32, 256))
				mul := kernreg.Put(p, "mul", seqF32(8, seed+1))
				mins := kernreg.Put(p, "mins", seqF32(8, seed+2))
				dequantQ4KBodyNEON(ptr(qs), ptr(out), (*[8]float32)(mul), (*[8]float32)(mins), 0.75)
				return out
			},
		},
		{
			Symbol: "dequantQ6KBodyNEON", Lengths: ggufDequantSeeds,
			Call: func(p *kernreg.Placer, seed int) any {
				q := kernreg.Put(p, "q", seqU8(q6kBodyBytes, seed))
				out := kernreg.Put(p, "out", make([]float32, 256))
				ds := kernreg.Put(p, "ds", seqF32(16, seed+1))
				dequantQ6KBodyNEON(ptr(q), ptr(out), (*[16]float32)(ds))
				return out
			},
		},
	}
}
