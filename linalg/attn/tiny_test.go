package attn

import (
	"fmt"
	"go/token"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/internal/kernreg"
	"github.com/townsendmerino/aikit/linalg"
)

// The attention entry points over tiny shapes (OG3): every exported one, and the scratch's
// fits, with each buffer against the end guard and the start guard, under Sweep's deadline.
// TestTinyShapes_complete holds the list to attn's exported names.

// tinyDims are the sizes each dimension takes: 0, 1, and one under, at and one over each
// block size the kernels use (4, 8, 16, 32, 64).
var tinyDims = []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65}

// shapeLabels returns a Label for an adapter whose n indexes shapes.
func shapeLabels[S interface{ String() string }](shapes []S) func(int) string {
	return func(i int) string { return shapes[i].String() }
}

// tmResult is what a tiny-shape adapter returns: the refusal ("" when the call returned)
// and every output it wrote.
func tmResult(refused string, outs ...any) any { return append([]any{refused}, outs...) }

func TestTinyShapes_everyExportedEntryPoint(t *testing.T) {
	kernreg.Sweep(t, tmAttnAdapters())
}

func TestTinyShapes_complete(t *testing.T) {
	var swept []string
	for _, ad := range tmAttnAdapters() {
		if i := strings.LastIndexByte(ad.Symbol, '.'); token.IsExported(ad.Symbol[i+1:]) {
			swept = append(swept, ad.Symbol)
		}
	}
	probs, err := kernreg.ExportedProblems(".", swept, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range probs {
		t.Error(p)
	}
}

// tmStrided is an attention shape: rows (M or G), the reduced length (K or nKeys), the
// output count (N or hd), and the offset and row stride into the cache.
type tmStrided struct{ rows, red, out, off, stride int }

func (s tmStrided) String() string {
	return fmt.Sprintf("rows=%d red=%d out=%d off=%d stride=%d", s.rows, s.red, s.out, s.off, s.stride)
}

// tmStridedShapes varies rows, red and out over dims around (1, 8, 8), each with the row
// stride equal to the row width plus 0, 1 and 5, and offsets 0 and 3. width picks which
// dimension a cache row holds (red for QK, out for AV).
func tmStridedShapes(dims []int, width func(s tmStrided) int) []tmStrided {
	var base []tmStrided
	base = append(base, tmStrided{1, 8, 8, 0, 0})
	for _, d := range dims {
		base = append(base, tmStrided{d, 8, 8, 0, 0}, tmStrided{1, d, 8, 0, 0}, tmStrided{1, 8, d, 0, 0})
	}
	base = append(base, tmStrided{0, 0, 0, 0, 0}, tmStrided{1, 1, 1, 0, 0})
	var out []tmStrided
	for _, s := range base {
		for _, pad := range []int{0, 1, 5} {
			for _, off := range []int{0, 3} {
				t := s
				t.off, t.stride = off, width(s)+pad
				out = append(out, t)
			}
		}
	}
	return out
}

func tmAttnAdapters() []kernreg.Adapter {
	qk := tmStridedShapes(tinyDims, func(s tmStrided) int { return s.red })
	av := tmStridedShapes(tinyDims, func(s tmStrided) int { return s.out })
	// cache is the exact cache length the entry points require: off + (n-1)·stride + width.
	cache := func(off, n, stride, width int) int { return off + max(0, n-1)*stride + width }
	qkAd := func(name string, f16 bool, group bool) kernreg.Adapter {
		return kernreg.Adapter{
			Symbol: name, Lengths: shapeIndices(len(qk)), Label: shapeLabels(qk),
			Call: func(p *kernreg.Placer, i int) any {
				s := qk[i]
				a := kernreg.Put(p, "a", seqF32(s.rows*s.red, 1))
				dst := kernreg.Put(p, "dst", make([]float32, s.rows*s.out))
				n := cache(s.off, s.out, s.stride, s.red)
				var r string
				switch {
				case f16:
					b := kernreg.Put(p, "bMat", linalg.F32ToF16Scales(seqF32(n, 2)))
					r = kernreg.Refused(func() { MatmulQKAcc64F16(a, b, dst, s.rows, s.red, s.out, s.off, s.stride) })
				case group:
					b := kernreg.Put(p, "bMat", seqF32(n, 2))
					r = kernreg.Refused(func() { MatmulQKAcc64Group(a, b, dst, s.rows, s.red, s.out, s.off, s.stride) })
				default:
					b := kernreg.Put(p, "bMat", seqF32(n, 2))
					r = kernreg.Refused(func() { MatmulQKAcc64(a, b, dst, s.rows, s.red, s.out, s.off, s.stride) })
				}
				return tmResult(r, dst)
			},
		}
	}
	avAd := func(name string, f16 bool, group bool) kernreg.Adapter {
		return kernreg.Adapter{
			Symbol: name, Lengths: shapeIndices(len(av)), Label: shapeLabels(av),
			Call: func(p *kernreg.Placer, i int) any {
				s := av[i]
				scores := kernreg.Put(p, "scores", seqF32(s.rows*s.red, 1))
				dst := kernreg.Put(p, "dst", make([]float32, s.rows*s.out))
				accN := s.out
				if group {
					accN = s.rows * s.out
				}
				acc := kernreg.Put(p, "acc", make([]float64, accN))
				n := cache(s.off, s.red, s.stride, s.out)
				var r string
				switch {
				case f16:
					v := kernreg.Put(p, "vals", linalg.F32ToF16Scales(seqF32(n, 2)))
					r = kernreg.Refused(func() { MatmulAVAcc64F16(scores, v, dst, acc, s.rows, s.red, s.out, s.off, s.stride) })
				case group:
					v := kernreg.Put(p, "vals", seqF32(n, 2))
					r = kernreg.Refused(func() { MatmulAVAcc64Group(scores, v, dst, acc, s.rows, s.red, s.out, s.off, s.stride) })
				default:
					v := kernreg.Put(p, "vals", seqF32(n, 2))
					r = kernreg.Refused(func() { MatmulAVAcc64(scores, v, dst, acc, s.rows, s.red, s.out, s.off, s.stride) })
				}
				return tmResult(r, dst)
			},
		}
	}

	// The fused tile: kt query rows, head dim hd, nKeys keys, every row allowed every key
	// (full) or keys 0..i (causal). nKeys also crosses the key block.
	type tile struct {
		kt, hd, nKeys int
		causal        bool
	}
	var tiles []tile
	add := func(kt, hd, nk int) {
		tiles = append(tiles, tile{kt, hd, nk, false}, tile{kt, hd, nk, true})
	}
	add(2, 8, 8)
	for _, d := range tinyDims {
		add(d, 8, 8)
		add(2, d, 8)
		add(2, 8, d)
	}
	for _, nk := range []int{FusedAttnKeyBlock - 1, FusedAttnKeyBlock, FusedAttnKeyBlock + 1, 2*FusedAttnKeyBlock + 1} {
		add(2, 4, nk)
	}
	add(0, 0, 0)
	add(1, 1, 1)
	tileLabel := func(i int) string {
		t := tiles[i]
		return fmt.Sprintf("kt=%d hd=%d nKeys=%d causal=%v", t.kt, t.hd, t.nKeys, t.causal)
	}
	tileAd := func(name string, attend func(mm func(a, b, dst []float32, M, K, N int), qh, kh, vBlk, ch []float32, sc *FusedAttnScratch, kt, hd, nKeys int, scale float64, lo, hi []int) bool) kernreg.Adapter {
		return kernreg.Adapter{
			Symbol: name, Lengths: shapeIndices(len(tiles)), Label: tileLabel,
			Call: func(p *kernreg.Placer, i int) any {
				t := tiles[i]
				qh := kernreg.Put(p, "qh", seqF32(t.kt*t.hd, 1))
				kh := kernreg.Put(p, "kh", seqF32(t.nKeys*t.hd, 2))
				vBlk := kernreg.Put(p, "vBlk", seqF32(t.nKeys*t.hd, 3))
				ch := kernreg.Put(p, "ch", make([]float32, t.kt*t.hd))
				lo, hi := make([]int, t.kt), make([]int, t.kt)
				for r := range t.kt {
					hi[r] = t.nKeys - 1
					if t.causal {
						hi[r] = min(r, t.nKeys-1)
					}
				}
				lo, hi = kernreg.Put(p, "lo", lo), kernreg.Put(p, "hi", hi)
				var ok bool
				r := kernreg.Refused(func() {
					sc := NewFusedAttnScratch(t.kt, t.hd)
					sc = &FusedAttnScratch{
						SBlk: kernreg.Put(p, "SBlk", sc.SBlk), Tmp: kernreg.Put(p, "Tmp", sc.Tmp),
						Acc: kernreg.Put(p, "Acc", sc.Acc), MRun: kernreg.Put(p, "MRun", sc.MRun), LRun: kernreg.Put(p, "LRun", sc.LRun),
					}
					ok = attend(linalg.MatmulBT, qh, kh, vBlk, ch, sc, t.kt, t.hd, t.nKeys, 0.125, lo, hi)
				})
				return tmResult(r, ok, ch)
			},
		}
	}

	// GatherVBlockMajor: one kv head kvh of a cache whose rows are kvDim wide.
	type gather struct{ nKeys, hd, kvh, pad int }
	var gs []gather
	for _, d := range tinyDims {
		gs = append(gs, gather{d, 8, 1, 3}, gather{8, d, 1, 3}, gather{8, 8, 0, 0})
	}
	for _, nk := range []int{FusedAttnKeyBlock - 1, FusedAttnKeyBlock, FusedAttnKeyBlock + 1} {
		gs = append(gs, gather{nk, 4, 2, 1})
	}
	gs = append(gs, gather{0, 0, 0, 0}, gather{1, 1, 0, 0})
	gLabel := func(i int) string {
		g := gs[i]
		return fmt.Sprintf("nKeys=%d hd=%d kvh=%d kvDim=%d", g.nKeys, g.hd, g.kvh, (g.kvh+1)*g.hd+g.pad)
	}

	// Fits: a scratch sized for one tile asked about another.
	type fits struct{ kt0, hd0, kt, hd int }
	var fs []fits
	for _, d := range tinyDims {
		fs = append(fs, fits{2, 8, d, 8}, fits{2, 8, 2, d}, fits{d, d, 2, 8})
	}
	fLabel := func(i int) string {
		f := fs[i]
		return fmt.Sprintf("scratch(kt=%d hd=%d).fits(kt=%d hd=%d)", f.kt0, f.hd0, f.kt, f.hd)
	}

	// NewFusedAttnScratch over rows (kt) and cols (hd), as tiny_other's toRC.
	var rc []tmStrided
	for _, c := range tinyDims {
		rc = append(rc, tmStrided{rows: 1, red: c})
	}
	for _, r := range tinyDims {
		rc = append(rc, tmStrided{rows: r, red: 4})
	}
	rc = append(rc, tmStrided{}, tmStrided{rows: 2, red: 3}, tmStrided{rows: 3, red: 1})
	return []kernreg.Adapter{
		qkAd("MatmulQKAcc64", false, false),
		qkAd("MatmulQKAcc64F16", true, false),
		qkAd("MatmulQKAcc64Group", false, true),
		avAd("MatmulAVAcc64", false, false),
		avAd("MatmulAVAcc64F16", true, false),
		avAd("MatmulAVAcc64Group", false, true),
		tileAd("AttendTileFused", AttendTileFused),
		tileAd("AttendTileFusedContractExp", AttendTileFusedContractExp),
		{
			Symbol: "GatherVBlockMajor", Lengths: shapeIndices(len(gs)), Label: gLabel,
			Call: func(p *kernreg.Placer, i int) any {
				g := gs[i]
				kvDim := (g.kvh+1)*g.hd + g.pad
				n := cache(g.kvh*g.hd, g.nKeys, kvDim, g.hd)
				if g.nKeys == 0 {
					n = 0
				}
				keys := kernreg.Put(p, "keys", seqF32(n, 1))
				vals := kernreg.Put(p, "vals", seqF32(n, 2))
				kh := kernreg.Put(p, "kh", make([]float32, g.nKeys*g.hd))
				vBlk := kernreg.Put(p, "vBlk", make([]float32, g.nKeys*g.hd))
				r := kernreg.Refused(func() { GatherVBlockMajor(kh, vBlk, keys, vals, g.kvh, g.hd, kvDim, g.nKeys) })
				return tmResult(r, kh, vBlk)
			},
		},
		{
			Symbol: "NewFusedAttnScratch", Lengths: shapeIndices(len(rc)), Label: shapeLabels(rc),
			Call: func(_ *kernreg.Placer, i int) any {
				s := rc[i]
				var out []any
				r := kernreg.Refused(func() {
					f := NewFusedAttnScratch(s.rows, s.red)
					out = []any{len(f.SBlk), len(f.Tmp), len(f.Acc), len(f.MRun), len(f.LRun), f.fits(s.rows, s.red)}
				})
				return tmResult(r, out...)
			},
		},
		{
			Symbol: "FusedAttnScratch.fits", Lengths: shapeIndices(len(fs)), Label: fLabel,
			Call: func(_ *kernreg.Placer, i int) any {
				f := fs[i]
				var ok bool
				r := kernreg.Refused(func() { ok = NewFusedAttnScratch(f.kt0, f.hd0).fits(f.kt, f.hd) })
				return tmResult(r, ok)
			},
		},
	}
}
