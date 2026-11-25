package linalg

import "fmt"

// BatchOp is one weight of a MatmulBTBatchInto: W the [N, K] weight, Dst its [M, N] output (N = W.Rows()).
type BatchOp struct {
	W   *WeightMat
	Dst []float32
}

// MatmulBTBatchInto computes op.Dst[M, N] = a[M, K] · op.Wᵀ for every op, the weights sharing one activation (a fused
// q‖k‖v or gate‖up). When every weight is W8A8 int8, or every weight is int4 with one group size, with one K and one
// activation group, the activation is quantized once and every op's columns run under one fork/join. Any other batch
// runs MatmulBTInto once per op. Either way each op's output is the bits MatmulBTInto gives it, a fused int4 op through
// the layouts it holds: row4 when present, split-half only for a weight that keeps no canonical bytes, else canonical
// (the layouts are bit-identical). A batch with a repacked-only int4 weight is fused only at M = 1 with every op's row
// count a multiple of 4, where no column range falls back to canonical bytes; otherwise it runs per op.
//
// Output contract: overwrites each op's Dst; do not pre-zero. Covers Dst[:M*op.W.Rows()] of every op.
func MatmulBTBatchInto(ws *Workspace, a []float32, M int, ops []BatchOp) {
	if len(ops) == 0 {
		return
	}
	for i, op := range ops {
		if op.W == nil {
			panic(fmt.Sprintf("linalg: MatmulBTBatchInto: ops[%d].W is nil", i))
		}
	}
	if ws == nil {
		ws = new(Workspace)
	}
	switch batchKind(ops, M) {
	case batchW8A8:
		w0 := ops[0].W
		w0.withWeightGroup(ws, func(ws *Workspace) {
			bops := growOps(&ws.batchW8, len(ops))
			for i, op := range ops {
				bops[i] = w8a8Op{BQ: op.W.q8, Scales: op.W.scales, Dst: op.Dst, N: op.W.rows}
			}
			defer clear(bops)
			matmulBTW8A8Batch(ws, a, M, w0.cols, bops)
		})
	case batchW4A8:
		w0 := ops[0].W
		w0.withWeightGroup(ws, func(ws *Workspace) {
			bops := growOps(&ws.batchW4, len(ops))
			for i, op := range ops {
				w := op.W
				bops[i] = w4a8Op{W4: w.q4, ScalesF16: w.q4s16, Row4: w.q4Row4, Row4ScalesF16: w.q4Row4Scales16, Dst: op.Dst, N: w.rows}
				if w.q4 == nil {
					bops[i].SplitHalf = w.q4SplitHalf
				}
			}
			defer clear(bops)
			matmulBTW4A8Batch(ws, a, M, w0.cols, w0.group, bops)
		})
	default:
		for _, op := range ops {
			op.W.MatmulBTInto(ws, a, op.Dst, M)
		}
	}
}

type batchKindT uint8

const (
	batchPerOp batchKindT = iota
	batchW8A8
	batchW4A8
)

// batchKind reports whether ops can share one quantized activation: all W8A8 int8, or all int4 with one group, with
// one K and one activation-group stamp. An int4 batch holding a repacked-only weight (no canonical bytes) also needs
// M = 1 and every op's row count a multiple of 4: the fused int4 kernel serves a repacked layout only at M = 1, and only
// over whole row4 quads, falling back to canonical bytes for anything else.
func batchKind(ops []BatchOp, M int) batchKindT {
	w0 := ops[0].W
	isW8 := func(w *WeightMat) bool { return w.kq == nil && w.q8 != nil && w.w8a8 }
	isW4 := func(w *WeightMat) bool {
		return w.kq == nil && (w.q4 != nil || w.q4Row4 != nil || w.q4SplitHalf != nil)
	}
	var kind batchKindT
	switch {
	case isW8(w0):
		kind = batchW8A8
	case isW4(w0):
		kind = batchW4A8
	default:
		return batchPerOp
	}
	repackedOnly, quadAligned := false, true
	for _, op := range ops {
		w := op.W
		if w.cols != w0.cols || w.actGroup != w0.actGroup {
			return batchPerOp
		}
		if (kind == batchW8A8 && !isW8(w)) || (kind == batchW4A8 && (!isW4(w) || w.group != w0.group)) {
			return batchPerOp
		}
		repackedOnly = repackedOnly || (kind == batchW4A8 && w.q4 == nil)
		quadAligned = quadAligned && w.rows%4 == 0
	}
	if repackedOnly && (M != 1 || !quadAligned) {
		return batchPerOp
	}
	return kind
}

func growOps[T any](buf *[]T, n int) []T {
	if cap(*buf) < n {
		*buf = make([]T, n)
	}
	return (*buf)[:n]
}
