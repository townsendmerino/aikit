package linalg

import "fmt"

// W4A8 over a pre-quantized activation (goinfer audit R-13). Every W4A8 entry quantizes its activation to int8
// first, and a caller that runs several projections over one input (q, k and v of one normed row; gate and up; every
// routed expert of a MoE layer) re-quantizes the same input once per call. Quantization is deterministic, so the
// repeats are dead work: QuantizeActQ (or WeightMat.QuantizeActW4A8) quantizes once, and the Pre entries run the same
// kernels on it. Each quantizing entry is itself QuantizeActQ into workspace scratch followed by the Pre entry's
// dispatch, so a Pre call is bit-identical to the matching non-Pre call by construction.

// ActQ is an activation block quantized once for the W4A8 Pre entries: M rows of K int8 codes, with one scale per
// row (Group 0) or one per group of Group consecutive elements (scales[m*ceil(K/Group)+g]), exactly as the
// quantizing entries produce them. Build it with QuantizeActQ or WeightMat.QuantizeActW4A8; the buffers are the
// caller's and are reused across calls when large enough.
type ActQ struct {
	Q     []int8
	S     []float32
	M, K  int
	Group int // 0: one scale per row; otherwise the activation quantization group
}

// QuantizeActQ quantizes M rows of K activations into q, one scale per row when group is 0 or one per group of
// `group` elements otherwise: the quantization MatmulBTW4A8F16Into performs under a workspace whose activation
// group is `group` (Workspace.SetActQuantGroup, or ActQuantGroup's global). q's buffers are grown when short.
//
// Output contract: overwrites q.Q[:M*K] and q.S[:M] (group 0) or q.S[:M*ceil(K/group)]; do not pre-zero.
func QuantizeActQ(a []float32, M, K, group int, q *ActQ) {
	if len(a) < M*K || M < 0 || K <= 0 || group < 0 {
		panic(fmt.Sprintf("linalg: QuantizeActQ: len(a)=%d M=%d K=%d group=%d", len(a), M, K, group))
	}
	nS := M
	if group > 0 {
		nS = M * ((K + group - 1) / group)
	}
	if cap(q.Q) < M*K {
		q.Q = make([]int8, M*K)
	}
	if cap(q.S) < nS {
		q.S = make([]float32, nS)
	}
	q.Q, q.S, q.M, q.K, q.Group = q.Q[:M*K], q.S[:nS], M, K, group
	if group > 0 {
		QuantizeActivationsGroupedInto(q.Q, q.S, a, M, K, group)
		return
	}
	quantizeRowsInto(q.Q, q.S, a, M, K, 0)
}

// quantizeActScratch is QuantizeActQ into the workspace's scratch, as every quantizing W4A8 entry did before it was
// split (the same buffers, the same quantizer).
func quantizeActScratch(ws *Workspace, group int, a []float32, M, K int) ActQ {
	if group > 0 {
		aq, aS, _ := quantizeActGrouped(ws, group, a, M, K)
		return ActQ{Q: aq, S: aS, M: M, K: K, Group: group}
	}
	aq := ws.int8Buf(M * K)
	aS := ws.f32Buf(M)
	quantizeRowsInto(aq, aS, a, M, K, ws.width)
	return ActQ{Q: aq, S: aS, M: M, K: K, Group: 0}
}

// checkActQ refuses an ActQ that was not quantized for the call it is handed to: another shape, or another
// activation group than the one the matching quantizing entry would use (the result would silently differ).
func checkActQ(fn string, q *ActQ, M, K, group int) {
	if q == nil || q.M != M || q.K != K || q.Group != group || len(q.Q) < M*K {
		panic(fmt.Sprintf("linalg: %s: activation quantized for M=%d K=%d group=%d, called with M=%d K=%d group=%d",
			fn, qField(q, 0), qField(q, 1), qField(q, 2), M, K, group))
	}
}

func qField(q *ActQ, i int) int {
	if q == nil {
		return -1
	}
	return [3]int{q.M, q.K, q.Group}[i]
}

// MatmulBTW4A8F16Pre is MatmulBTW4A8F16Into on an activation already quantized by QuantizeActQ with the
// workspace's activation group (0 when none is set): bit-identical to MatmulBTW4A8F16Into on the activation q
// was quantized from.
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:M*N].
func MatmulBTW4A8F16Pre(ws *Workspace, q *ActQ, w4 []byte, wScales16 []uint16, dst []float32, M, K, N, group int) {
	checkActQ("MatmulBTW4A8F16Pre", q, M, K, actGroupFor(ws))
	checkMatmulW4A8("MatmulBTW4A8F16Pre", M*K, len(w4), len(wScales16), len(dst), M, K, N, group)
	matmulBTW4A8F16Q(ws, q, w4, wScales16, dst, M, K, N, group)
}

// QuantizeActW4A8 quantizes M rows of w's input for w.MatmulBTW4A8PreInto under ws: one scale per row, or per
// group when ws (or w) carries an activation group, exactly as w.MatmulBTW4A8Into would. Weights that share their
// input and their activation group (q, k and v; gate and up; a MoE layer's experts) can share one block. Set the
// group on ws (Workspace.SetActQuantGroup) rather than only on w: arm64's row4 M=1 path reads the workspace's group
// alone, so a weight-only group can resolve differently across layouts, and the Pre entry then refuses the block.
//
// Output contract: overwrites q.Q[:M*K] and q.S (see QuantizeActQ); do not pre-zero.
func (w *WeightMat) QuantizeActW4A8(ws *Workspace, a []float32, M int, q *ActQ) {
	QuantizeActQ(a, M, w.cols, w.w4a8ActGroup(ws, M), q)
}

// MatmulBTW4A8PreInto is MatmulBTW4A8Into on an activation already quantized by QuantizeActW4A8 (or QuantizeActQ
// with w.ActQuantGroup's effective group): bit-identical to w.MatmulBTW4A8Into on the activation q was quantized
// from, on every layout and arch (it is the same dispatch).
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:M*N].
func (w *WeightMat) MatmulBTW4A8PreInto(ws *Workspace, q *ActQ, dst []float32, M int) {
	checkActQ("WeightMat.MatmulBTW4A8PreInto", q, M, w.cols, w.w4a8ActGroup(ws, M))
	w.matmulBTW4A8Q(ws, q, dst, M)
}
