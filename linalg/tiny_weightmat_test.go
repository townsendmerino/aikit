package linalg

import (
	"fmt"

	"github.com/townsendmerino/aikit/internal/kernreg"
)

// twShape is one tiny WeightMat call: an M-row input against a rows×cols weight, with an
// int4 group size (ignored by the other kinds) and, for int8, the W8A8 switch.
type twShape struct {
	M, rows, cols, group int
	w8a8                 bool
}

func (s twShape) String() string {
	return fmt.Sprintf("M=%d rows=%d cols=%d group=%d w8a8=%v", s.M, s.rows, s.cols, s.group, s.w8a8)
}

// twShapes varies one dimension at a time over tinyDims (group over tinyGroups) around a
// small valid base, plus the all-zero and all-one corners.
func twShapes(withGroup bool) []twShape {
	base := twShape{M: 1, rows: 4, cols: 64, group: 32}
	var out []twShape
	for _, d := range tinyDims {
		s := base
		s.M = d
		out = append(out, s)
		s = base
		s.rows = d
		out = append(out, s)
		s = base
		s.cols = d
		out = append(out, s)
	}
	if withGroup {
		for _, g := range tinyGroups {
			s := base
			s.group = g
			out = append(out, s)
		}
		// A group that does not divide cols, and cols under one group.
		out = append(out, twShape{M: 2, rows: 3, cols: 48, group: 32}, twShape{M: 2, rows: 3, cols: 17, group: 32})
	}
	return append(out, twShape{}, twShape{M: 1, rows: 1, cols: 1, group: 1})
}

// twKQuantShapes sweeps a K-quant weight: cols is a multiple of the 256-element super-block
// (0, one and two blocks), rows and M over tinyDims around a one-block base.
func twKQuantShapes() []twShape {
	base := twShape{M: 1, rows: 4, cols: 256}
	var out []twShape
	for _, d := range tinyDims {
		s := base
		s.M = d
		out = append(out, s)
		s = base
		s.rows = d
		out = append(out, s)
	}
	for _, c := range []int{0, 512} {
		s := base
		s.cols = c
		out = append(out, s)
	}
	return append(out, twShape{}, twShape{M: 1, rows: 1, cols: 256})
}

// twKind builds one storage kind's WeightMat over inputs placed with p: the weight's
// own bytes, exactly the length its constructor's contract names, are what sit against
// the guards. built is false when the constructor declined (ok=false or an error); then
// msg says why.
type twKind struct {
	name   string // the constructor's exported name
	shapes []twShape
	int4   bool // runs the W4A8 entry points too
	build  func(p *kernreg.Placer, s twShape) (w WeightMat, built bool, msg string)
}

func twWeights(s twShape) []float32 { return seqF32(s.rows*s.cols, 7) }

func twQuantInt4(s twShape) (packed []byte, scales []float32) {
	return QuantizeGroupsInt4(twWeights(s), s.rows, s.cols, s.group)
}

// twKQuantRaw is rows·cols/256 GGUF super-blocks of blockBytes each, with the f16 scale
// fields at dOff (and dOff+2 for dmin when withMin) set to small finite values.
func twKQuantRaw(s twShape, blockBytes, dOff int, withMin bool) []byte {
	nb := s.rows * s.cols / 256
	raw := seqU8(nb*blockBytes, 5)
	d, dmin := F32ToF16(0.0078125), F32ToF16(0.00390625)
	for b := range nb {
		o := b*blockBytes + dOff
		raw[o], raw[o+1] = byte(d), byte(d>>8)
		if withMin {
			raw[o+2], raw[o+3] = byte(dmin), byte(dmin>>8)
		}
	}
	return raw
}

func twKinds() []twKind {
	plain, grouped, kq := twShapes(false), twShapes(true), twKQuantShapes()
	var int8Shapes []twShape
	for _, s := range plain {
		int8Shapes = append(int8Shapes, s)
		s.w8a8 = true
		int8Shapes = append(int8Shapes, s)
	}
	declined := func(ok bool, what string) string {
		if ok {
			return ""
		}
		return what + " declined (ok=false)"
	}
	errMsg := func(err error) string {
		if err != nil {
			return "error: " + err.Error()
		}
		return ""
	}
	return []twKind{
		{name: "WrapF32", shapes: plain, build: func(p *kernreg.Placer, s twShape) (WeightMat, bool, string) {
			return WrapF32(kernreg.Put(p, "w", twWeights(s)), s.rows, s.cols), true, ""
		}},
		{name: "WrapF16", shapes: plain, build: func(p *kernreg.Placer, s twShape) (WeightMat, bool, string) {
			// Element by element: F32ToF16Slice refuses an empty src with a runtime error
			// (a finding of its own, reported with the other parts' entry points).
			w := twWeights(s)
			u := make([]uint16, len(w))
			for i, v := range w {
				u[i] = F32ToF16(v)
			}
			return WrapF16(kernreg.Put(p, "u", u), s.rows, s.cols), true, ""
		}},
		{name: "WrapBF16", shapes: plain, build: func(p *kernreg.Placer, s twShape) (WeightMat, bool, string) {
			w := twWeights(s)
			u := make([]uint16, len(w))
			for i, v := range w {
				u[i] = F32ToBF16(v)
			}
			return WrapBF16(kernreg.Put(p, "u", u), s.rows, s.cols), true, ""
		}},
		{name: "WrapInt8", shapes: int8Shapes, build: func(p *kernreg.Placer, s twShape) (WeightMat, bool, string) {
			q, sc := QuantizeRowsInt8(twWeights(s), s.rows, s.cols)
			return WrapInt8(kernreg.Put(p, "q8", q), kernreg.Put(p, "scales", sc), s.rows, s.cols, s.w8a8), true, ""
		}},
		{name: "WrapInt4", shapes: grouped, int4: true, build: func(p *kernreg.Placer, s twShape) (WeightMat, bool, string) {
			q4, q4s := twQuantInt4(s)
			w, ok := WrapInt4(kernreg.Put(p, "q4", q4), kernreg.Put(p, "q4s16", F32ToF16Scales(q4s)), s.rows, s.cols, s.group, LayoutCanonical)
			return w, ok, declined(ok, "WrapInt4(LayoutCanonical)")
		}},
		// The repacked layouts through the same constructor; their names' lowercase last
		// part keeps them out of the completeness count, which WrapInt4 above answers.
		{name: "WrapInt4.row4", shapes: grouped, int4: true, build: func(p *kernreg.Placer, s twShape) (WeightMat, bool, string) {
			q4, q4s := twQuantInt4(s)
			r4, r4s16, _ := twRow4F16(q4, F32ToF16Scales(q4s), s)
			w, ok := WrapInt4(twPutOrNil(p, "q4Row4", r4), twPutOrNil(p, "q4Row4Scales", r4s16), s.rows, s.cols, s.group, LayoutRow4)
			return w, ok, declined(ok, "WrapInt4(LayoutRow4)")
		}},
		{name: "WrapInt4.splithalf", shapes: grouped, int4: true, build: func(p *kernreg.Placer, s twShape) (WeightMat, bool, string) {
			q4, q4s := twQuantInt4(s)
			sh := RepackW4A8SplitHalf(q4, s.rows, s.cols, s.group)
			w, ok := WrapInt4(kernreg.Put(p, "q4SplitHalf", sh), kernreg.Put(p, "q4s16", F32ToF16Scales(q4s)), s.rows, s.cols, s.group, LayoutSplitHalf)
			return w, ok, declined(ok, "WrapInt4(LayoutSplitHalf)")
		}},
		{name: "WrapQ4K", shapes: kq, build: func(p *kernreg.Placer, s twShape) (WeightMat, bool, string) {
			w, err := WrapQ4K(kernreg.Put(p, "raw", twKQuantRaw(s, 144, 0, true)), s.rows, s.cols)
			return w, err == nil, errMsg(err)
		}},
		{name: "WrapQ5K", shapes: kq, build: func(p *kernreg.Placer, s twShape) (WeightMat, bool, string) {
			w, err := WrapQ5K(kernreg.Put(p, "raw", twKQuantRaw(s, 176, 0, true)), s.rows, s.cols)
			return w, err == nil, errMsg(err)
		}},
		{name: "WrapQ6K", shapes: kq, build: func(p *kernreg.Placer, s twShape) (WeightMat, bool, string) {
			w, err := WrapQ6K(kernreg.Put(p, "raw", twKQuantRaw(s, 210, 208, false)), s.rows, s.cols)
			return w, err == nil, errMsg(err)
		}},
	}
}

// twPutOrNil places x, or passes nil through (the "no row4 layout" argument).
func twPutOrNil[T any](p *kernreg.Placer, name string, x []T) []T {
	if x == nil {
		return nil
	}
	return kernreg.Put(p, name, x)
}

// twOutcome is one entry point's result on one shape: its refusal ("" if it returned)
// and what it wrote.
type twOutcome struct {
	Refused string
	Out     []float32
}

// twMethod runs one WeightMat entry point on w, with a fresh Workspace and its input and
// output placed with p.
type twMethod struct {
	name string
	int4 bool // W4A8-only
	run  func(p *kernreg.Placer, w *WeightMat, s twShape) twOutcome
}

func twMethods() []twMethod {
	in := func(p *kernreg.Placer, s twShape) ([]float32, []float32) {
		return kernreg.Put(p, "a", seqF32(s.M*s.cols, 3)), kernreg.Put(p, "dst", make([]float32, s.M*s.rows))
	}
	return []twMethod{
		{name: "WeightMat.MatmulBT", run: func(p *kernreg.Placer, w *WeightMat, s twShape) twOutcome {
			a, dst := in(p, s)
			return twOutcome{kernreg.Refused(func() { w.MatmulBT(a, dst, s.M) }), dst}
		}},
		{name: "WeightMat.MatmulBTInto", run: func(p *kernreg.Placer, w *WeightMat, s twShape) twOutcome {
			a, dst := in(p, s)
			var ws Workspace
			return twOutcome{kernreg.Refused(func() { w.MatmulBTInto(&ws, a, dst, s.M) }), dst}
		}},
		{name: "WeightMat.Row", run: func(p *kernreg.Placer, w *WeightMat, s twShape) twOutcome {
			dst := kernreg.Put(p, "row", make([]float32, s.rows*s.cols))
			msg := kernreg.Refused(func() {
				for i := range s.rows {
					w.Row(i, dst[i*s.cols:(i+1)*s.cols])
				}
			})
			return twOutcome{msg, dst}
		}},
		{name: "WeightMat.MatmulBTW4A8Into", int4: true, run: func(p *kernreg.Placer, w *WeightMat, s twShape) twOutcome {
			a, dst := in(p, s)
			var ws Workspace
			return twOutcome{kernreg.Refused(func() { w.MatmulBTW4A8Into(&ws, a, dst, s.M) }), dst}
		}},
		{name: "WeightMat.QuantizeActW4A8", int4: true, run: func(p *kernreg.Placer, w *WeightMat, s twShape) twOutcome {
			a := kernreg.Put(p, "a", seqF32(s.M*s.cols, 3))
			var ws Workspace
			var q ActQ
			msg := kernreg.Refused(func() { w.QuantizeActW4A8(&ws, a, s.M, &q) })
			out := make([]float32, 0, len(q.Q)+len(q.S))
			for _, v := range q.Q[:min(len(q.Q), s.M*s.cols)] {
				out = append(out, float32(v))
			}
			return twOutcome{msg, append(out, q.S...)}
		}},
		{name: "WeightMat.MatmulBTW4A8PreInto", int4: true, run: func(p *kernreg.Placer, w *WeightMat, s twShape) twOutcome {
			a, dst := in(p, s)
			var ws Workspace
			var q ActQ
			msg := kernreg.Refused(func() {
				w.QuantizeActW4A8(&ws, a, s.M, &q)
				w.MatmulBTW4A8PreInto(&ws, &q, dst, s.M)
			})
			return twOutcome{msg, dst}
		}},
		{name: "WeightMat.Int4", int4: true, run: func(p *kernreg.Placer, w *WeightMat, s twShape) twOutcome {
			var q4s []float32
			var ok bool
			msg := kernreg.Refused(func() { _, q4s, _, ok = w.Int4() })
			return twOutcome{fmt.Sprintf("%s ok=%v", msg, ok), q4s}
		}},
		{name: "WeightMat.Int4Row4", int4: true, run: func(p *kernreg.Placer, w *WeightMat, s twShape) twOutcome {
			var s4 []float32
			var ok bool
			msg := kernreg.Refused(func() { _, s4, ok = w.Int4Row4() })
			return twOutcome{fmt.Sprintf("%s ok=%v", msg, ok), s4}
		}},
		{name: "WeightMat.RepackInt4SplitHalf", int4: true, run: func(p *kernreg.Placer, w *WeightMat, s twShape) twOutcome {
			a, dst := in(p, s)
			var ws Workspace
			var did bool
			msg := kernreg.Refused(func() {
				did = w.RepackInt4SplitHalf()
				w.MatmulBTW4A8Into(&ws, a, dst, s.M)
			})
			return twOutcome{fmt.Sprintf("%s repacked=%v", msg, did), dst}
		}},
		{name: "WeightMat.RepackInt4Row4", int4: true, run: func(p *kernreg.Placer, w *WeightMat, s twShape) twOutcome {
			a, dst := in(p, s)
			var ws Workspace
			var did bool
			msg := kernreg.Refused(func() {
				did = w.RepackInt4Row4()
				w.MatmulBTW4A8Into(&ws, a, dst, s.M)
			})
			return twOutcome{fmt.Sprintf("%s repacked=%v", msg, did), dst}
		}},
	}
}

// twCase is one (kind, shape) a method adapter runs.
type twCase struct {
	kind  twKind
	shape twShape
}

func (c twCase) String() string { return c.kind.name + " " + c.shape.String() }

// twBuildAndRun builds k's WeightMat for s and runs each of methods on it; the result is
// the constructor's refusal, then each method's outcome.
func twBuildAndRun(p *kernreg.Placer, k twKind, s twShape, methods []twMethod) any {
	var w WeightMat
	var built bool
	var msg string
	if r := kernreg.Refused(func() { w, built, msg = k.build(p, s) }); r != "" {
		return r
	}
	if !built {
		return msg
	}
	out := []any{msg}
	for _, m := range methods {
		if m.int4 && !k.int4 {
			continue
		}
		out = append(out, m.name, m.run(p, &w, s))
	}
	return out
}

func tinyWeightMatAdapters() []kernreg.Adapter {
	kinds, methods := twKinds(), twMethods()
	var ads []kernreg.Adapter
	// One adapter per constructor: every shape of its kind, through every method.
	for _, k := range kinds {
		ads = append(ads, kernreg.Adapter{
			Symbol:  k.name,
			Lengths: shapeIndices(len(k.shapes)),
			Label:   shapeLabels(k.shapes),
			Call:    func(p *kernreg.Placer, i int) any { return twBuildAndRun(p, k, k.shapes[i], methods) },
		})
	}
	// One adapter per method: every kind it applies to, over the shapes that vary M (the
	// method's own dimension) and the two corners. The constructor adapters above already
	// run every method over every rows, cols and group of their kind.
	for _, m := range methods {
		var cases []twCase
		for _, k := range kinds {
			if m.int4 && !k.int4 {
				continue
			}
			base := k.shapes[0]
			for _, s := range k.shapes {
				corner := s.rows <= 1 && s.cols <= 1
				if corner || (s.rows == base.rows && s.cols == base.cols && s.group == base.group) {
					cases = append(cases, twCase{k, s})
				}
			}
		}
		ads = append(ads, kernreg.Adapter{
			Symbol:  m.name,
			Lengths: shapeIndices(len(cases)),
			Label:   shapeLabels(cases),
			Call: func(p *kernreg.Placer, i int) any {
				return twBuildAndRun(p, cases[i].kind, cases[i].shape, []twMethod{m})
			},
		})
	}
	return append(ads, twWorkspaceAdapters()...)
}

// twMKN is one Workspace matmul shape, with the strided form's layout of b.
type twMKN struct{ M, K, N, off, rowStride, elemStride int }

func (s twMKN) String() string {
	return fmt.Sprintf("M=%d K=%d N=%d off=%d rowStride=%d elemStride=%d", s.M, s.K, s.N, s.off, s.rowStride, s.elemStride)
}

func twMKNShapes() []twMKN {
	base := twMKN{M: 2, K: 16, N: 3}
	var out []twMKN
	for _, d := range tinyDims {
		for dim := range 3 {
			s := base
			switch dim {
			case 0:
				s.M = d
			case 1:
				s.K = d
			default:
				s.N = d
			}
			out = append(out, s)
		}
	}
	return append(out, twMKN{}, twMKN{M: 1, K: 1, N: 1})
}

// twStridedShapes adds the strided form's layouts: packed, padded rows, every other
// element, and an offset, over a few of the plain shapes.
func twStridedShapes() []twMKN {
	var out []twMKN
	for _, s := range twMKNShapes() {
		for _, l := range [][3]int{{0, s.K, 1}, {5, s.K + 3, 1}, {1, 2*s.K + 1, 2}} {
			t := s
			t.off, t.rowStride, t.elemStride = l[0], l[1], l[2]
			out = append(out, t)
		}
	}
	return out
}

// twStridedLen is the elements of bMat the strided matmul reads: up to b[N-1][K-1].
func twStridedLen(s twMKN) int {
	if s.N == 0 || s.K == 0 {
		return s.off
	}
	return s.off + (s.N-1)*s.rowStride + (s.K-1)*s.elemStride + 1
}

func twWorkspaceAdapters() []kernreg.Adapter {
	mkn, strided := twMKNShapes(), twStridedShapes()
	ab := func(p *kernreg.Placer, s twMKN) (a, b, dst []float32) {
		return kernreg.Put(p, "a", seqF32(s.M*s.K, 1)), kernreg.Put(p, "b", seqF32(s.N*s.K, 2)), kernreg.Put(p, "dst", make([]float32, s.M*s.N))
	}
	return []kernreg.Adapter{
		{Symbol: "Workspace.MatmulBT", Lengths: shapeIndices(len(mkn)), Label: shapeLabels(mkn),
			Call: func(p *kernreg.Placer, i int) any {
				s := mkn[i]
				a, b, dst := ab(p, s)
				var ws Workspace
				return twOutcome{kernreg.Refused(func() { ws.MatmulBT(a, b, dst, s.M, s.K, s.N) }), dst}
			}},
		{Symbol: "Workspace.matmulBTAcc64", Lengths: shapeIndices(len(mkn)), Label: shapeLabels(mkn),
			Call: func(p *kernreg.Placer, i int) any {
				s := mkn[i]
				a, b, dst := ab(p, s)
				var ws Workspace
				return twOutcome{kernreg.Refused(func() { ws.matmulBTAcc64(a, b, dst, s.M, s.K, s.N) }), dst}
			}},
		{Symbol: "Workspace.matmulBTAcc64Strided", Lengths: shapeIndices(len(strided)), Label: shapeLabels(strided),
			Call: func(p *kernreg.Placer, i int) any {
				s := strided[i]
				a := kernreg.Put(p, "a", seqF32(s.M*s.K, 1))
				bMat := kernreg.Put(p, "bMat", seqF32(twStridedLen(s), 2))
				dst := kernreg.Put(p, "dst", make([]float32, s.M*s.N))
				var ws Workspace
				return twOutcome{kernreg.Refused(func() {
					ws.matmulBTAcc64Strided(a, bMat, dst, s.M, s.K, s.N, s.off, s.rowStride, s.elemStride)
				}), dst}
			}},
		{Symbol: "Workspace.QuantizeActivations", Lengths: shapeIndices(len(mkn)), Label: shapeLabels(mkn),
			Call: func(p *kernreg.Placer, i int) any {
				s := mkn[i]
				a := kernreg.Put(p, "a", seqF32(s.M*s.K, 1))
				var ws Workspace
				var aq []int8
				var sc []float32
				msg := kernreg.Refused(func() { aq, sc = ws.QuantizeActivations(a, s.M, s.K) })
				out := make([]float32, 0, len(aq)+len(sc))
				for _, v := range aq {
					out = append(out, float32(v))
				}
				return twOutcome{msg, append(out, sc...)}
			}},
	}
}

const twField = "returns a field or a value derived from the shape; reads no weight bytes and takes no shape"

var tinyWeightMatExempt = map[string]string{
	"ActQuantGroup":              "reports a package setting; takes no shape",
	"WeightMat.ActQuantGroup":    twField,
	"WeightMat.Cols":             twField,
	"WeightMat.Rows":             twField,
	"WeightMat.Kind":             twField,
	"WeightMat.IsInt4":           twField,
	"WeightMat.Int4Layout":       twField,
	"WeightMat.SplitHalfBytes":   twField,
	"WeightMat.F32":              twField,
	"WeightMat.Half16":           twField,
	"WeightMat.Int4F16":          twField,
	"WeightMat.Int4Row4F16":      twField,
	"WeightMat.Int4ScalesF16":    twField,
	"WeightMat.Int8":             twField,
	"WeightMat.Q4K":              twField,
	"WeightMat.Q5K":              twField,
	"WeightMat.Q6K":              twField,
	"WeightMat.SetActQuantGroup": "stores a setting; takes no shape",
	"WeightMat.MappedSpan":       "returns a page-aligned sub-slice of the weight's bytes by address arithmetic; reads none of them",
	"WeightMat.MappedSpanRow4":   "returns a page-aligned sub-slice of the row4 bytes by address arithmetic; reads none of them",
	"Workspace.SetActQuantGroup": "stores a setting; takes no shape",
	"Workspace.SetThreshold":     "stores a setting; takes no shape",
	"Workspace.SetWorkers":       "stores a setting; takes no shape",
}
