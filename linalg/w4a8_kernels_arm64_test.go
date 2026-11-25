package linalg

import (
	"github.com/townsendmerino/aikit/internal/cpufeat"
	"github.com/townsendmerino/aikit/internal/kernreg"
)

// w4a8Quad places the interleaved inputs of the row4 kernels for g groups: per group
// the four weight rows' 16 bytes back to back (64), and their four scales.
func w4a8Quad(p *kernreg.Placer, g int) (packed4 []byte, scales4 []float32) {
	return kernreg.Put(p, "packed4", seqU8(64*g, 2)), kernreg.Put(p, "scales4", seqF32(4*g, 3))
}

func w4a8Adapters() []kernreg.Adapter {
	dp := func() bool { return cpufeat.HasDotProd }
	// row is the adapter of a single-row kernel taking (act, packed, scales, nGroups).
	row := func(sym string, groups []int, k func(act *int8, packed *byte, scales *float32, nGroups int) float32) kernreg.Adapter {
		return kernreg.Adapter{
			Symbol: sym, Needs: dp, Lengths: groups,
			Call: func(p *kernreg.Placer, g int) any {
				act, packed, scales := w4a8Row(p, g)
				return k(ptr(act), ptr(packed), ptr(scales), g)
			},
		}
	}
	// quad is the adapter of a row4 kernel taking (act, packed4, scales4, dst, nGroups).
	quad := func(sym string, k func(act *int8, packed4 *byte, scales4 *float32, dst *float32, nGroups int)) kernreg.Adapter {
		return kernreg.Adapter{
			Symbol: sym, Needs: dp, Lengths: w4a8Groups(1, 1),
			Call: func(p *kernreg.Placer, g int) any {
				act := kernreg.Put(p, "act", seqI8(32*g, 1))
				packed4, scales4 := w4a8Quad(p, g)
				dst := kernreg.Put(p, "dst", make([]float32, 4))
				k(ptr(act), ptr(packed4), ptr(scales4), ptr(dst), g)
				return dst
			},
		}
	}
	// corr is w4a8LaneCorrNeg8's output for act, computed by the Go reference so the
	// fold kernels' corr input is placed like any other.
	corr := func(p *kernreg.Placer, act []int8, g int) []int32 {
		c := make([]int32, 4*g)
		w4a8LaneCorrNeg8Scalar(act, c, g)
		return kernreg.Put(p, "corr", c)
	}
	tileShapes := w4a8TileShapes(1)
	tile4Shapes := w4a8TileShapes(0) // dotW4A8Tile4RowSDOT tests nGroups for zero
	return []kernreg.Adapter{
		// The loops run at least once: nGroups >= 1 (dotW4A8 calls with K >= 32).
		row("dotW4A8FoldSDOT", w4a8Groups(1, 1), dotW4A8FoldSDOT),
		row("dotW4A8SplitHalfSDOT", w4a8Groups(1, 1), dotW4A8SplitHalfSDOT),
		// Two or four groups per pass: nGroups a multiple of 2 or 4.
		row("dotW4A8FoldSDOT2Acc", w4a8Groups(2, 2), dotW4A8FoldSDOT2Acc),
		row("dotW4A8FoldSDOT4Acc", w4a8Groups(4, 4), dotW4A8FoldSDOT4Acc),
		row("dotW4A8SplitHalf2Acc", w4a8Groups(2, 2), dotW4A8SplitHalf2Acc),
		row("dotW4A8SplitHalf4Acc", w4a8Groups(4, 4), dotW4A8SplitHalf4Acc),
		{
			Symbol: "dotW4A8FoldSDOTv2", Needs: dp, Lengths: w4a8Groups(1, 1),
			Call: func(p *kernreg.Placer, g int) any {
				act, packed, scales := w4a8Row(p, g)
				sumAct := kernreg.Put(p, "sumAct", make([]int32, g))
				for i := range sumAct {
					sumAct[i] = int32(i*3 - 7)
				}
				return dotW4A8FoldSDOTv2(ptr(act), ptr(packed), ptr(scales), ptr(sumAct), g)
			},
		},
		quad("dotW4A8SplitHalf4Row", dotW4A8SplitHalf4Row),
		quad("dotW4A8SplitHalf4RowPrefetch", func(act *int8, packed4 *byte, scales4 *float32, dst *float32, nGroups int) {
			// A prefetch hint past the block does not fault; the distance is a hint only.
			dotW4A8SplitHalf4RowPrefetch(act, packed4, scales4, dst, nGroups, 256)
		}),
		{
			Symbol: "dotW4A8SplitHalf4RowDeshared", Needs: dp, Lengths: w4a8Groups(1, 1),
			Call: func(p *kernreg.Placer, g int) any {
				act := kernreg.Put(p, "act", seqI8(32*g, 1))
				var pk [4][]byte
				var sc [4][]float32
				for r := range 4 {
					pk[r] = kernreg.Put(p, "packed"+string(rune('0'+r)), seqU8(16*g, r+2))
					sc[r] = kernreg.Put(p, "scales"+string(rune('0'+r)), seqF32(g, r+6))
				}
				dst := kernreg.Put(p, "dst", make([]float32, 4))
				dotW4A8SplitHalf4RowDeshared(ptr(act), ptr(pk[0]), ptr(pk[1]), ptr(pk[2]), ptr(pk[3]), ptr(sc[0]), ptr(sc[1]), ptr(sc[2]), ptr(sc[3]), ptr(dst), g)
				return dst
			},
		},
		{
			Symbol: "w4a8LaneCorrNeg8", Needs: dp, Lengths: w4a8Groups(1, 1),
			Call: func(p *kernreg.Placer, g int) any {
				act := kernreg.Put(p, "act", seqI8(32*g, 1))
				c := kernreg.Put(p, "corr", make([]int32, 4*g))
				w4a8LaneCorrNeg8(ptr(act), ptr(c), g)
				return c
			},
		},
		{
			Symbol: "dotW4A8SplitHalf4RowFold", Needs: dp, Lengths: w4a8Groups(1, 1),
			Call: func(p *kernreg.Placer, g int) any {
				a := seqI8(32*g, 1)
				act, c := kernreg.Put(p, "act", a), corr(p, a, g)
				packed4, scales4 := w4a8Quad(p, g)
				dst := kernreg.Put(p, "dst", make([]float32, 4))
				dotW4A8SplitHalf4RowFold(ptr(act), ptr(c), ptr(packed4), ptr(scales4), ptr(dst), g)
				return dst
			},
		},
		{
			Symbol: "dotW4A8SplitHalf4RowFoldF16", Needs: dp, Lengths: w4a8Groups(1, 1),
			Call: func(p *kernreg.Placer, g int) any {
				a := seqI8(32*g, 1)
				act, c := kernreg.Put(p, "act", a), corr(p, a, g)
				packed4 := kernreg.Put(p, "packed4", seqU8(64*g, 2))
				scales4 := kernreg.Put(p, "scales4", w4a8SeqU16(4*g, 3))
				dst := kernreg.Put(p, "dst", make([]float32, 4))
				dotW4A8SplitHalf4RowFoldF16(ptr(act), ptr(c), ptr(packed4), ptr(scales4), ptr(dst), g)
				return dst
			},
		},
		{
			// Four activation rows against one canonical weight row.
			Symbol: "dotW4A8Tile4RowSDOT", Needs: dp, Lengths: w4a8Indices(len(tile4Shapes)),
			Call: func(p *kernreg.Placer, i int) any {
				s := tile4Shapes[i]
				K := 32 * s.nGroups
				stride := K + s.actPad
				act := kernreg.Put(p, "act", seqI8(3*stride+K, 1))
				packed, scales := kernreg.Put(p, "packed", seqU8(16*s.nGroups, 2)), kernreg.Put(p, "scales", seqF32(s.nGroups, 3))
				dst := kernreg.Put(p, "dst", make([]float32, 4))
				dotW4A8Tile4RowSDOT(ptr(act), stride, ptr(packed), ptr(scales), ptr(dst), s.nGroups)
				return dst
			},
		},
		{
			// Four activation rows against one row4 quad: 16 outputs.
			Symbol: "dotW4A8Row4Tile4x4", Needs: dp, Lengths: w4a8Indices(len(tileShapes)),
			Call: func(p *kernreg.Placer, i int) any {
				s := tileShapes[i]
				K := 32 * s.nGroups
				stride := K + s.actPad
				act := kernreg.Put(p, "act", seqI8(3*stride+K, 1))
				packed4, scales4 := w4a8Quad(p, s.nGroups)
				dst := kernreg.Put(p, "dst", make([]float32, 16))
				dotW4A8Row4Tile4x4(ptr(act), stride, ptr(packed4), ptr(scales4), ptr(dst), s.nGroups)
				return dst
			},
		},
	}
}
