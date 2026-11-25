package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

func doti8Adapters() []kernreg.Adapter {
	dotProd := func() bool { return hasDotProd }
	tiles := i8TileShapes()
	return []kernreg.Adapter{
		{
			Symbol: "dotI8NEON",
			// n a multiple of 16, n/16 steps checked before the loop.
			Lengths: upTo(16, 160, 1008, 1024, 1040),
			Call: func(p *kernreg.Placer, n int) any {
				a, b := kernreg.Put(p, "a", seqI8(n, 1)), kernreg.Put(p, "b", seqI8(n, 2))
				return dotI8NEON(ptr(a), ptr(b), n)
			},
		},
		{
			Symbol: "dotI8SDOT", Needs: dotProd,
			// n a multiple of 16: 64-byte blocks, then 16-byte steps.
			Lengths: upTo(16, 272, 1008, 1024, 1040),
			Call: func(p *kernreg.Placer, n int) any {
				a, b := kernreg.Put(p, "a", seqI8(n, 1)), kernreg.Put(p, "b", seqI8(n, 2))
				return dotI8SDOT(ptr(a), ptr(b), n)
			},
		},
		{
			Symbol: "dotI8Tile4x4", Needs: dotProd,
			// n indexes i8TileShapes, each used as both strides; the loop runs at least
			// once, so n >= 16 (w8a8TileRect requires K >= 16).
			Lengths: indexes(len(tiles)),
			Call: func(p *kernreg.Placer, i int) any {
				s := tiles[i]
				act := kernreg.Put(p, "act", seqI8(3*s.stride+s.n, 1))
				w := kernreg.Put(p, "w", seqI8(3*s.stride+s.n, 2))
				dst := kernreg.Put(p, "dst", make([]int32, 16))
				dotI8Tile4x4(ptr(act), s.stride, ptr(w), s.stride, ptr(dst), s.n)
				return dst
			},
		},
	}
}
