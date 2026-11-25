package linalg

import (
	"github.com/townsendmerino/aikit/internal/cpufeat"
	"github.com/townsendmerino/aikit/internal/kernreg"
)

func w4a8Adapters() []kernreg.Adapter {
	avx2 := func() bool { return cpufeat.HasAVX2 }
	// row is the adapter of a single-row kernel taking (act, packed, scales, nGroups).
	row := func(sym string, needs func() bool, groups []int, k func(act *int8, packed *byte, scales *float32, nGroups int) float32) kernreg.Adapter {
		return kernreg.Adapter{
			Symbol: sym, Needs: needs, Lengths: groups,
			Call: func(p *kernreg.Placer, g int) any {
				act, packed, scales := w4a8Row(p, g)
				return k(ptr(act), ptr(packed), ptr(scales), g)
			},
		}
	}
	// tile is the adapter of a four-activation-row kernel against one weight row.
	tile := func(sym string, k func(act *int8, actStride int, packed *byte, scales *float32, dst *float32, nGroups int)) kernreg.Adapter {
		shapes := w4a8TileShapes(1)
		return kernreg.Adapter{
			Symbol: sym, Needs: avx2, Lengths: w4a8Indices(len(shapes)),
			Call: func(p *kernreg.Placer, i int) any {
				s := shapes[i]
				K := 32 * s.nGroups
				stride := K + s.actPad
				act := kernreg.Put(p, "act", seqI8(3*stride+K, 1))
				packed, scales := kernreg.Put(p, "packed", seqU8(16*s.nGroups, 2)), kernreg.Put(p, "scales", seqF32(s.nGroups, 3))
				dst := kernreg.Put(p, "dst", make([]float32, 4))
				k(ptr(act), stride, ptr(packed), ptr(scales), ptr(dst), s.nGroups)
				return dst
			},
		}
	}
	return []kernreg.Adapter{
		// The loops run at least once: nGroups >= 1 (dotW4A8 calls with K >= 32).
		row("dotW4A8FoldAVX2", avx2, w4a8Groups(1, 1), dotW4A8FoldAVX2),
		// Two groups per pass; nGroups must be even.
		row("dotW4A8Fold2AccAVX2", avx2, w4a8Groups(2, 2), dotW4A8Fold2AccAVX2),
		row("dotW4A8SplitHalfAVX2", avx2, w4a8Groups(1, 1), dotW4A8SplitHalfAVX2),
		// Tests nGroups for zero itself.
		row("dotW4A8FoldAVX512VNNI", func() bool { return cpufeat.HasAVX512VNNIVL }, w4a8Groups(0, 1), dotW4A8FoldAVX512VNNI),
		{
			// Two groups per pass, then a one-group tail; zero is handled.
			Symbol: "dotW4A8SplitHalfScaledAVX2", Needs: avx2, Lengths: w4a8Groups(0, 1),
			Call: func(p *kernreg.Placer, g int) any {
				act, packed, scales := w4a8Row(p, g)
				aScales := kernreg.Put(p, "aScales", seqF32(g, 4))
				return dotW4A8SplitHalfScaledAVX2(ptr(act), ptr(packed), ptr(scales), ptr(aScales), g)
			},
		},
		{
			// n8 blocks of eight halves; zero is handled.
			Symbol: "cvtF16ToF32x8F16C", Needs: func() bool { return cpufeat.HasF16C }, Lengths: upTo(1, 40, 127, 128, 129),
			Call: func(p *kernreg.Placer, n8 int) any {
				src := kernreg.Put(p, "src", w4a8SeqU16(8*n8, 1))
				dst := kernreg.Put(p, "dst", make([]float32, 8*n8))
				cvtF16ToF32x8F16C(ptr(dst), ptr(src), n8)
				return dst
			},
		},
		{
			// Widens the nGroups scales into buf eight at a time, the last block
			// overlapping, so nGroups >= 8 (dotW4A8F16Row's guard); buf holds nGroups.
			Symbol: "dotW4A8FoldF16RowAVX2", Needs: func() bool { return cpufeat.HasAVX2 && cpufeat.HasF16C }, Lengths: w4a8Groups(8, 1),
			Call: func(p *kernreg.Placer, g int) any {
				act, packed := kernreg.Put(p, "act", seqI8(32*g, 1)), kernreg.Put(p, "packed", seqU8(16*g, 2))
				scales := kernreg.Put(p, "scales", w4a8SeqU16(g, 3))
				buf := kernreg.Put(p, "buf", make([]float32, g))
				return []any{dotW4A8FoldF16RowAVX2(ptr(act), ptr(packed), ptr(scales), g, ptr(buf)), buf}
			},
		},
		tile("dotW4A8SplitHalfTile4RowAVX2", dotW4A8SplitHalfTile4RowAVX2),
		tile("dotW4A8Tile4RowAVX2", dotW4A8Tile4RowAVX2),
	}
}
