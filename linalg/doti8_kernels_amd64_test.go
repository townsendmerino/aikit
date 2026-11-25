package linalg

import (
	"github.com/townsendmerino/aikit/internal/cpufeat"
	"github.com/townsendmerino/aikit/internal/kernreg"
)

func doti8Adapters() []kernreg.Adapter {
	avx2 := func() bool { return cpufeat.HasAVX2 }
	tiles := i8TileShapes()
	return []kernreg.Adapter{
		{
			Symbol: "dotI8AVX512VNNI", Needs: func() bool { return cpufeat.HasAVX512VNNI },
			// n a multiple of 64: 128 per pass, then at most one 64-step.
			Lengths: upTo(64, 512, 1984, 2048, 2112),
			Call: func(p *kernreg.Placer, n int) any {
				a, b := kernreg.Put(p, "a", seqI8(n, 1)), kernreg.Put(p, "b", seqI8(n, 2))
				return dotI8AVX512VNNI(ptr(a), ptr(b), n)
			},
		},
		{
			Symbol: "dotI8Scaled32AVX2", Needs: avx2,
			// n is nGroups of 32: four groups per pass, then one at a time.
			Lengths: upTo(1, 13, 31, 32, 33),
			Call: func(p *kernreg.Placer, nG int) any {
				a, b := kernreg.Put(p, "a", seqI8(32*nG, 1)), kernreg.Put(p, "b", seqI8(32*nG, 2))
				aS := kernreg.Put(p, "aS", seqF32(nG, 3))
				return dotI8Scaled32AVX2(ptr(a), ptr(b), ptr(aS), nG)
			},
		},
		{
			Symbol: "dotI8Scaled32x2AVX2", Needs: avx2,
			// nGroups: four per pass, then a pair, then one.
			Lengths: upTo(1, 13, 31, 32, 33),
			Call: func(p *kernreg.Placer, nG int) any {
				a := kernreg.Put(p, "a", seqI8(32*nG, 1))
				b0, b1 := kernreg.Put(p, "b0", seqI8(32*nG, 2)), kernreg.Put(p, "b1", seqI8(32*nG, 3))
				aS := kernreg.Put(p, "aS", seqF32(nG, 4))
				d0, d1 := dotI8Scaled32x2AVX2(ptr(a), ptr(b0), ptr(b1), ptr(aS), nG)
				return [2]float32{d0, d1}
			},
		},
		{
			Symbol: "dotI8Tile4x1AVX2", Needs: avx2,
			// n indexes i8TileShapes; the loop runs at least once, so n >= 16 (w8a8TileRect
			// requires K >= 16).
			Lengths: indexes(len(tiles)),
			Call: func(p *kernreg.Placer, i int) any {
				s := tiles[i]
				act := kernreg.Put(p, "act", seqI8(3*s.stride+s.n, 1))
				w := kernreg.Put(p, "w", seqI8(s.n, 2))
				dst := kernreg.Put(p, "dst", make([]int32, 4))
				dotI8Tile4x1AVX2(ptr(act), s.stride, ptr(w), ptr(dst), s.n)
				return dst
			},
		},
		{
			Symbol: "dotI8x8AVX2", Needs: avx2,
			// n a multiple of 16, checked before each step.
			Lengths: upTo(16, 160, 1008, 1024, 1040),
			Call: func(p *kernreg.Placer, n int) any {
				a := kernreg.Put(p, "a", seqI8(n, 1))
				var b [8][]int8
				for r := range b {
					b[r] = kernreg.Put(p, "b"+string(rune('0'+r)), seqI8(n, r+2))
				}
				out := kernreg.Put(p, "out", make([]int32, 8))
				dotI8x8AVX2(ptr(a), ptr(b[0]), ptr(b[1]), ptr(b[2]), ptr(b[3]), ptr(b[4]), ptr(b[5]), ptr(b[6]), ptr(b[7]), n, (*[8]int32)(out))
				return out
			},
		},
	}
}
