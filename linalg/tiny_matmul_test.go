package linalg

import (
	"fmt"
	"maps"

	"github.com/townsendmerino/aikit/internal/kernreg"
)

// tinyMatmulAdapters sweep the matmul entry points over tiny shapes (attention's are
// linalg/attn's tiny_test.go).
// Weights are built directly (codes from seqU8/seqI8, small positive scales) rather than
// through the package's quantizers, so a quantizer's own refusal or crash is never
// reported under a matmul's name.
func tinyMatmulAdapters() []kernreg.Adapter {
	var out []kernreg.Adapter
	out = append(out, tmDenseAdapters()...)
	out = append(out, tmQ8Adapters()...)
	out = append(out, tmInt4Adapters()...)
	out = append(out, tmGGUFAdapters()...)
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
		q8("matmulBTQ8", matmulBTQ8),
		q8("matmulBTQ8Into", func(a []float32, bQ []int8, bS, dst []float32, M, K, N int) {
			matmulBTQ8Into(new(Workspace), a, bQ, bS, dst, M, K, N)
		}),
		q8("MatmulBTQ8Fused", func(a []float32, bQ []int8, bS, dst []float32, M, K, N int) {
			MatmulBTQ8Fused(dst, a, bQ, bS, M, K, N)
		}),
		q8("MatmulBTQ8FusedInto", func(a []float32, bQ []int8, bS, dst []float32, M, K, N int) {
			MatmulBTQ8FusedInto(dst, a, bQ, bS, M, K, N)
		}),
		q8("matmulBTW8A8", matmulBTW8A8),
		q8("matmulBTW8A8Into", func(a []float32, bQ []int8, bS, dst []float32, M, K, N int) {
			matmulBTW8A8Into(new(Workspace), a, bQ, bS, dst, M, K, N)
		}),
		tmMKNAdapter("MatmulBTW8A8Pre", shapes, func(p *kernreg.Placer, s tmMKN) any {
			a, bQ, bS, dst := weights(p, s)
			aq := kernreg.Put(p, "aq", make([]int8, s.M*s.K))
			aS := kernreg.Put(p, "aScales", make([]float32, s.M))
			r := kernreg.Refused(func() {
				quantizeActivationsInto(aq, aS, a, s.M, s.K)
				MatmulBTW8A8Pre(new(Workspace), aq, aS, bQ, bS, dst, s.M, s.K, s.N)
			})
			return tmResult(r, dst)
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
		plain("matmulBTQ4", func(_ *Workspace, a []float32, w4 []byte, wS, dst []float32, M, K, N, g int) {
			matmulBTQ4(a, w4, wS, dst, M, K, N, g)
		}),
		plain("matmulBTQ4Into", matmulBTQ4Into),
		plain("matmulBTW4A8", func(_ *Workspace, a []float32, w4 []byte, wS, dst []float32, M, K, N, g int) {
			matmulBTW4A8(a, w4, wS, dst, M, K, N, g)
		}),
		plain("matmulBTW4A8Into", matmulBTW4A8Into),
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
		// MatmulBTBatchInto over int4 weights (canonical, binary16 scales) and, at the
		// same M, K and N, over W8A8 int8 weights. A second op of four rows makes the
		// batch's column split cross an op boundary. The WeightMats alias the placed
		// slices, so the weight bytes are what sit against the guards.
		tmGroupAdapter("MatmulBTBatchInto", shapes, func(p *kernreg.Placer, s tmG) any {
			a, w4, _, dst := inputs(p, s)
			bpr, ng := tmInt4Sizes(s.K, s.group)
			wS16 := f16(p, s)
			w42 := kernreg.Put(p, "w4_2", seqU8(4*bpr, 4))
			wS2 := kernreg.Put(p, "wScales2", F32ToF16Scales(tmScales(4*ng, 5)))
			dst2 := kernreg.Put(p, "dst2", make([]float32, s.M*4))
			r4 := kernreg.Refused(func() {
				w1, _ := WrapInt4(w4, wS16, s.N, s.K, s.group, LayoutCanonical)
				w2, _ := WrapInt4(w42, wS2, 4, s.K, s.group, LayoutCanonical)
				MatmulBTBatchInto(tmWorkspace(s.actGroup), a, s.M, []BatchOp{{W: &w1, Dst: dst}, {W: &w2, Dst: dst2}})
			})
			bQ := kernreg.Put(p, "bQ", seqI8(s.N*s.K, 2))
			bS := kernreg.Put(p, "bScales", tmScales(s.N, 3))
			bQ2 := kernreg.Put(p, "bQ2", seqI8(4*s.K, 4))
			bS2 := kernreg.Put(p, "bScales2", tmScales(4, 5))
			d8 := kernreg.Put(p, "dst8", make([]float32, s.M*s.N))
			d82 := kernreg.Put(p, "dst8_2", make([]float32, s.M*4))
			r8 := kernreg.Refused(func() {
				w1, w2 := WrapInt8(bQ, bS, s.N, s.K, true), WrapInt8(bQ2, bS2, 4, s.K, true)
				MatmulBTBatchInto(tmWorkspace(s.actGroup), a, s.M, []BatchOp{{W: &w1, Dst: d8}, {W: &w2, Dst: d82}})
			})
			return tmResult(r4+" | "+r8, dst, dst2, d8, d82)
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
		gguf("matmulBTGGUFQ4K", 144, []int{0, 2}, matmulBTGGUFQ4K),
		gguf("matmulBTGGUFQ6K", 210, []int{208}, matmulBTGGUFQ6K),
	}
}
