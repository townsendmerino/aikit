package linalg

import (
	"fmt"
	"maps"

	"github.com/townsendmerino/aikit/internal/kernreg"
)

// tinyMatmulAdapters sweep the matmul and attention entry points over tiny shapes.
// Weights are built directly (codes from seqU8/seqI8, small positive scales) rather than
// through the package's quantizers, so a quantizer's own refusal or crash is never
// reported under a matmul's name.
func tinyMatmulAdapters() []kernreg.Adapter {
	var out []kernreg.Adapter
	out = append(out, tmDenseAdapters()...)
	out = append(out, tmQ8Adapters()...)
	out = append(out, tmInt4Adapters()...)
	out = append(out, tmGGUFAdapters()...)
	out = append(out, tmAttnAdapters()...)
	out = append(out, tmArchAdapters()...)
	return out
}

// tinyMatmulExempt is empty on every architecture but the ones whose tiny_matmul_*
// file lists an arch-only name it cannot reach (tmArchExempt).
var tinyMatmulExempt = maps.Clone(tmArchExempt)

// tmMKN is a dense matmul shape.
type tmMKN struct{ M, K, N int }

func (s tmMKN) String() string { return fmt.Sprintf("M=%d K=%d N=%d", s.M, s.K, s.N) }

// tmVary returns base, then base with each dimension in turn set to every value in dims,
// then the corners all-0 and all-1.
func tmVary(base tmMKN, dims []int) []tmMKN {
	out := []tmMKN{base}
	for _, d := range dims {
		out = append(out, tmMKN{d, base.K, base.N}, tmMKN{base.M, d, base.N}, tmMKN{base.M, base.K, d})
	}
	return append(out, tmMKN{0, 0, 0}, tmMKN{1, 1, 1})
}

// tmScales is n small positive scales.
func tmScales(n, seed int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32((i*5+seed*3)%7+1) / 64
	}
	return s
}

// tmResult is what a tiny-shape adapter returns: the refusal ("" when the call returned)
// and every output it wrote.
func tmResult(refused string, outs ...any) any { return append([]any{refused}, outs...) }

func tmMKNAdapter(name string, shapes []tmMKN, call func(p *kernreg.Placer, s tmMKN) any) kernreg.Adapter {
	return kernreg.Adapter{
		Symbol:  name,
		Lengths: shapeIndices(len(shapes)),
		Label:   shapeLabels(shapes),
		Call:    func(p *kernreg.Placer, i int) any { return call(p, shapes[i]) },
	}
}

func tmDenseAdapters() []kernreg.Adapter {
	shapes := tmVary(tmMKN{1, 8, 8}, tinyDims)
	dense := func(name string, mm func(a, b, dst []float32, M, K, N int)) kernreg.Adapter {
		return tmMKNAdapter(name, shapes, func(p *kernreg.Placer, s tmMKN) any {
			a := kernreg.Put(p, "a", seqF32(s.M*s.K, 1))
			b := kernreg.Put(p, "b", seqF32(s.N*s.K, 2))
			dst := kernreg.Put(p, "dst", make([]float32, s.M*s.N))
			return tmResult(kernreg.Refused(func() { mm(a, b, dst, s.M, s.K, s.N) }), dst)
		})
	}
	// MatmulBTAcc64Strided over three layouts of b: packed, padded rows, and transposed
	// (elements a row apart).
	type strided struct {
		tmMKN
		off, row, elem int
	}
	var st []strided
	for _, s := range shapes {
		st = append(st,
			strided{s, 0, s.K, 1},
			strided{s, 3, s.K + 5, 1},
			strided{s, 2, 1, s.N + 1},
		)
	}
	stLabel := func(i int) string {
		s := st[i]
		return fmt.Sprintf("%v bOff=%d bRowStride=%d bElemStride=%d", s.tmMKN, s.off, s.row, s.elem)
	}
	return []kernreg.Adapter{
		dense("MatmulBT", MatmulBT),
		dense("MatmulBTAcc64", MatmulBTAcc64),
		dense("MatmulBTInto", func(a, b, dst []float32, M, K, N int) { MatmulBTInto(dst, a, b, M, K, N) }),
		{
			Symbol: "MatmulBTAcc64Strided", Lengths: shapeIndices(len(st)), Label: stLabel,
			Call: func(p *kernreg.Placer, i int) any {
				s := st[i]
				need := 0
				if s.M > 0 && s.N > 0 && s.K > 0 {
					need = s.off + (s.N-1)*s.row + (s.K-1)*s.elem + 1
				}
				a := kernreg.Put(p, "a", seqF32(s.M*s.K, 1))
				b := kernreg.Put(p, "bMat", seqF32(need, 2))
				dst := kernreg.Put(p, "dst", make([]float32, s.M*s.N))
				return tmResult(kernreg.Refused(func() { MatmulBTAcc64Strided(a, b, dst, s.M, s.K, s.N, s.off, s.row, s.elem) }), dst)
			},
		},
	}
}

func tmQ8Adapters() []kernreg.Adapter {
	shapes := tmVary(tmMKN{1, 16, 8}, tinyDims)
	weights := func(p *kernreg.Placer, s tmMKN) (a []float32, bQ []int8, bS []float32, dst []float32) {
		return kernreg.Put(p, "a", seqF32(s.M*s.K, 1)),
			kernreg.Put(p, "bQ", seqI8(s.N*s.K, 2)),
			kernreg.Put(p, "bScales", tmScales(s.N, 3)),
			kernreg.Put(p, "dst", make([]float32, s.M*s.N))
	}
	q8 := func(name string, mm func(a []float32, bQ []int8, bS, dst []float32, M, K, N int)) kernreg.Adapter {
		return tmMKNAdapter(name, shapes, func(p *kernreg.Placer, s tmMKN) any {
			a, bQ, bS, dst := weights(p, s)
			return tmResult(kernreg.Refused(func() { mm(a, bQ, bS, dst, s.M, s.K, s.N) }), dst)
		})
	}
	return []kernreg.Adapter{
		q8("MatmulBTQ8", MatmulBTQ8),
		q8("MatmulBTQ8Into", func(a []float32, bQ []int8, bS, dst []float32, M, K, N int) {
			MatmulBTQ8Into(new(Workspace), a, bQ, bS, dst, M, K, N)
		}),
		q8("MatmulBTQ8Fused", func(a []float32, bQ []int8, bS, dst []float32, M, K, N int) {
			MatmulBTQ8Fused(dst, a, bQ, bS, M, K, N)
		}),
		q8("MatmulBTQ8FusedInto", func(a []float32, bQ []int8, bS, dst []float32, M, K, N int) {
			MatmulBTQ8FusedInto(dst, a, bQ, bS, M, K, N)
		}),
		q8("MatmulBTW8A8", MatmulBTW8A8),
		q8("MatmulBTW8A8Into", func(a []float32, bQ []int8, bS, dst []float32, M, K, N int) {
			MatmulBTW8A8Into(new(Workspace), a, bQ, bS, dst, M, K, N)
		}),
		tmMKNAdapter("MatmulBTW8A8Pre", shapes, func(p *kernreg.Placer, s tmMKN) any {
			a, bQ, bS, dst := weights(p, s)
			aq := kernreg.Put(p, "aq", make([]int8, s.M*s.K))
			aS := kernreg.Put(p, "aScales", make([]float32, s.M))
			r := kernreg.Refused(func() {
				QuantizeActivationsInto(aq, aS, a, s.M, s.K)
				MatmulBTW8A8Pre(new(Workspace), aq, aS, bQ, bS, dst, s.M, s.K, s.N)
			})
			return tmResult(r, dst)
		}),
		tmMKNAdapter("MatmulBTW8A8Batch", shapes, func(p *kernreg.Placer, s tmMKN) any {
			a, bQ, bS, dst := weights(p, s)
			// A second op of four columns, so the batch's column split crosses an op boundary.
			bQ2 := kernreg.Put(p, "bQ2", seqI8(4*s.K, 4))
			bS2 := kernreg.Put(p, "bScales2", tmScales(4, 5))
			dst2 := kernreg.Put(p, "dst2", make([]float32, s.M*4))
			ops := []W8A8Op{{BQ: bQ, Scales: bS, Dst: dst, N: s.N}, {BQ: bQ2, Scales: bS2, Dst: dst2, N: 4}}
			return tmResult(kernreg.Refused(func() { MatmulBTW8A8Batch(new(Workspace), a, s.M, s.K, ops) }), dst, dst2)
		}),
		tmMKNAdapter("FusedQ8Applies", shapes, func(_ *kernreg.Placer, s tmMKN) any {
			var ok bool
			return tmResult(kernreg.Refused(func() { ok = FusedQ8Applies(s.K, s.N) }), ok)
		}),
	}
}

// tmG is a group-int4 shape, with the workspace's activation group (0: one scale per row).
type tmG struct {
	tmMKN
	group, actGroup int
}

func (s tmG) String() string {
	return fmt.Sprintf("%v group=%d actGroup=%d", s.tmMKN, s.group, s.actGroup)
}

// tmGroupShapes varies M, K, N and the group one at a time around base, over
// tinyDims, extra K values and tinyGroups, each with no activation group and with one of
// 32.
func tmGroupShapes(base tmG, extraK []int) []tmG {
	var out []tmG
	for _, ag := range []int{0, 32} {
		b := base
		b.actGroup = ag
		for _, s := range tmVary(base.tmMKN, append(append([]int{}, tinyDims...), extraK...)) {
			out = append(out, tmG{s, b.group, ag})
		}
		for _, g := range tinyGroups {
			out = append(out, tmG{base.tmMKN, g, ag})
		}
		out = append(out, tmG{tmMKN{0, 0, 0}, 0, ag}, tmG{tmMKN{1, 1, 1}, 1, ag})
	}
	return out
}

// tmInt4Sizes is the packed bytes per row and groups per row of a [*, K] int4 matrix; 0
// groups when group is not positive (the entry point must refuse it).
func tmInt4Sizes(K, group int) (bpr, nGroups int) {
	bpr = (K + 1) / 2
	if group > 0 {
		nGroups = (K + group - 1) / group
	}
	return bpr, nGroups
}

func tmWorkspace(actGroup int) *Workspace {
	ws := new(Workspace)
	ws.SetActQuantGroup(actGroup)
	return ws
}

func tmGroupAdapter(name string, shapes []tmG, call func(p *kernreg.Placer, s tmG) any) kernreg.Adapter {
	return kernreg.Adapter{
		Symbol:  name,
		Lengths: shapeIndices(len(shapes)),
		Label:   shapeLabels(shapes),
		Call:    func(p *kernreg.Placer, i int) any { return call(p, shapes[i]) },
	}
}

func tmInt4Adapters() []kernreg.Adapter {
	shapes := tmGroupShapes(tmG{tmMKN{1, 64, 4}, 32, 0}, []int{127, 128, 129})
	inputs := func(p *kernreg.Placer, s tmG) (a []float32, w4 []byte, wS []float32, dst []float32) {
		bpr, ng := tmInt4Sizes(s.K, s.group)
		return kernreg.Put(p, "a", seqF32(s.M*s.K, 1)),
			kernreg.Put(p, "w4", seqU8(s.N*bpr, 2)),
			kernreg.Put(p, "wScales", tmScales(s.N*ng, 3)),
			kernreg.Put(p, "dst", make([]float32, s.M*s.N))
	}
	plain := func(name string, mm func(ws *Workspace, a []float32, w4 []byte, wS, dst []float32, M, K, N, group int)) kernreg.Adapter {
		return tmGroupAdapter(name, shapes, func(p *kernreg.Placer, s tmG) any {
			a, w4, wS, dst := inputs(p, s)
			ws := tmWorkspace(s.actGroup)
			return tmResult(kernreg.Refused(func() { mm(ws, a, w4, wS, dst, s.M, s.K, s.N, s.group) }), dst)
		})
	}
	f16 := func(p *kernreg.Placer, s tmG) []uint16 {
		_, ng := tmInt4Sizes(s.K, s.group)
		return kernreg.Put(p, "wScales16", F32ToF16Scales(tmScales(s.N*ng, 3)))
	}
	return []kernreg.Adapter{
		plain("MatmulBTQ4", func(_ *Workspace, a []float32, w4 []byte, wS, dst []float32, M, K, N, g int) {
			MatmulBTQ4(a, w4, wS, dst, M, K, N, g)
		}),
		plain("MatmulBTQ4Into", MatmulBTQ4Into),
		plain("MatmulBTW4A8", func(_ *Workspace, a []float32, w4 []byte, wS, dst []float32, M, K, N, g int) {
			MatmulBTW4A8(a, w4, wS, dst, M, K, N, g)
		}),
		plain("MatmulBTW4A8Into", MatmulBTW4A8Into),
		tmGroupAdapter("MatmulBTW4A8F16Into", shapes, func(p *kernreg.Placer, s tmG) any {
			a, w4, _, dst := inputs(p, s)
			wS16 := f16(p, s)
			ws := tmWorkspace(s.actGroup)
			return tmResult(kernreg.Refused(func() { MatmulBTW4A8F16Into(ws, a, w4, wS16, dst, s.M, s.K, s.N, s.group) }), dst)
		}),
		tmGroupAdapter("MatmulBTW4A8F16Pre", shapes, func(p *kernreg.Placer, s tmG) any {
			a, w4, _, dst := inputs(p, s)
			wS16 := f16(p, s)
			ws := tmWorkspace(s.actGroup)
			r := kernreg.Refused(func() {
				var q ActQ
				QuantizeActQ(a, s.M, s.K, actGroupFor(ws), &q)
				MatmulBTW4A8F16Pre(ws, &q, w4, wS16, dst, s.M, s.K, s.N, s.group)
			})
			return tmResult(r, dst)
		}),
		tmGroupAdapter("MatmulBTW4A8Batch", shapes, func(p *kernreg.Placer, s tmG) any {
			a, w4, wS, dst := inputs(p, s)
			bpr, ng := tmInt4Sizes(s.K, s.group)
			w42 := kernreg.Put(p, "w4_2", seqU8(4*bpr, 4))
			wS2 := kernreg.Put(p, "wScales2", tmScales(4*ng, 5))
			dst2 := kernreg.Put(p, "dst2", make([]float32, s.M*4))
			ops := []W4A8Op{{W4: w4, Scales: wS, Dst: dst, N: s.N}, {W4: w42, Scales: wS2, Dst: dst2, N: 4}}
			ws := tmWorkspace(s.actGroup)
			return tmResult(kernreg.Refused(func() { MatmulBTW4A8Batch(ws, a, s.M, s.K, s.group, ops) }), dst, dst2)
		}),
	}
}

// tmGGUFRows is n rows of nb blocks of blockB bytes each, with each block's f16 scale
// fields (at the offsets given) set to a small finite value.
func tmGGUFRows(n, nb, blockB int, scaleOffs ...int) []byte {
	w := seqU8(n*nb*blockB, 7)
	h := F32ToF16(1.0 / 128)
	for b := range n * nb {
		for _, off := range scaleOffs {
			w[b*blockB+off], w[b*blockB+off+1] = byte(h), byte(h>>8)
		}
	}
	return w
}

func tmGGUFAdapters() []kernreg.Adapter {
	ks := []int{0, 1, 255, 256, 257, 511, 512, 513}
	shapes := tmVary(tmMKN{1, 256, 2}, nil)
	for _, d := range tinyDims {
		shapes = append(shapes, tmMKN{d, 256, 2}, tmMKN{1, 256, d})
	}
	for _, k := range ks {
		shapes = append(shapes, tmMKN{1, k, 2})
	}
	gguf := func(name string, blockB int, scaleOffs []int, mm func(a []float32, w []byte, dst []float32, M, K, N int)) kernreg.Adapter {
		return tmMKNAdapter(name, shapes, func(p *kernreg.Placer, s tmMKN) any {
			a := kernreg.Put(p, "a", seqF32(s.M*s.K, 1))
			w := kernreg.Put(p, "w", tmGGUFRows(s.N, s.K/256, blockB, scaleOffs...))
			dst := kernreg.Put(p, "dst", make([]float32, s.M*s.N))
			return tmResult(kernreg.Refused(func() { mm(a, w, dst, s.M, s.K, s.N) }), dst)
		})
	}
	return []kernreg.Adapter{
		gguf("MatmulBTGGUFQ4K", 144, []int{0, 2}, MatmulBTGGUFQ4K),
		gguf("MatmulBTGGUFQ6K", 210, []int{208}, MatmulBTGGUFQ6K),
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
					b := kernreg.Put(p, "bMat", F32ToF16Scales(seqF32(n, 2)))
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
					v := kernreg.Put(p, "vals", F32ToF16Scales(seqF32(n, 2)))
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
					ok = attend(MatmulBT, qh, kh, vBlk, ch, sc, t.kt, t.hd, t.nKeys, 0.125, lo, hi)
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
		return fmt.Sprintf("scratch(kt=%d hd=%d).Fits(kt=%d hd=%d)", f.kt0, f.hd0, f.kt, f.hd)
	}
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
			Symbol: "FusedAttnScratch.Fits", Lengths: shapeIndices(len(fs)), Label: fLabel,
			Call: func(_ *kernreg.Placer, i int) any {
				f := fs[i]
				var ok bool
				r := kernreg.Refused(func() { ok = NewFusedAttnScratch(f.kt0, f.hd0).Fits(f.kt, f.hd) })
				return tmResult(r, ok)
			},
		},
	}
}
