//go:build !arm64 && !amd64

package linalg

import "fmt"

// MatmulBTW4A8Into takes the canonical path on every target with no repacked layout of its own:
// q4Row4 is never set here (RepackInt4Row4 is a no-op) and neither is q4SplitHalf
// (RepackInt4SplitHalf likewise), so there is nothing to dispatch on.
//
// audit M-22: a repacked-only WeightMat cannot be constructed on this target
// at all (Int4Row4Usable/Int4SplitHalfUsable are both always false here, via
// row4Usable/splitHalfUsable below), so w.q4 == nil should be unreachable —
// panic rather than pass nil into the canonical kernel's length checks if
// something (a hand-built WeightMat bypassing the constructors) reaches it
// anyway.
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:M*N].
func (w *WeightMat) MatmulBTW4A8Into(ws *Workspace, a, dst []float32, M int) {
	q := quantizeActScratch(ws, w.w4a8ActGroup(ws, M), a, M, w.cols)
	w.matmulBTW4A8Q(ws, &q, dst, M)
}

// w4a8ActGroup is the activation group MatmulBTW4A8Into quantizes with (R-13).
func (w *WeightMat) w4a8ActGroup(ws *Workspace, _ int) int { return w.groupFor(ws) }

// matmulBTW4A8Q is MatmulBTW4A8Into after its activation quantization: the dispatch it and MatmulBTW4A8PreInto
// share (R-13).
func (w *WeightMat) matmulBTW4A8Q(ws *Workspace, q *ActQ, dst []float32, M int) {
	if q.Group > 0 {
		matmulW4A8GroupedQ(ws, q, w.int4Layout(), dst, M, w.rows) // actgroup.go
		return
	}
	if w.q4 == nil {
		panic(fmt.Sprintf("linalg: WeightMat.MatmulBTW4A8Into: no canonical int4 bytes (rows=%d cols=%d) "+
			"and this target has no repacked layout to fall back to", w.rows, w.cols))
	}
	matmulBTW4A8F16Q(ws, q, w.q4, w.q4s16, dst, M, w.cols, w.rows, w.group)
}

// RepackInt4SplitHalf is a no-op off amd64: the split-half layout's only consumer is the AVX2
// kernel. Always returns false; w is never modified.
func (w *WeightMat) RepackInt4SplitHalf() bool { return false }

// splitHalfUsable is always false off amd64 — see the arm64 file's twin for
// the same reasoning. Audit M-22.
func splitHalfUsable() bool { return false }

// w4a8BatchSplitHalfSpan is unreachable off amd64, same reasoning as the
// arm64 twin. Audit M-22 follow-up.
func w4a8BatchSplitHalfSpan(aq []int8, aScale float32, splitHalf []byte, scales, dst []float32, K, N, nGroups, bpr, j0, j1 int) {
	panic("linalg: w4a8BatchSplitHalfSpan reached off amd64 — splitHalfUsable() should have gated this")
}

// w4a8BatchSplitHalfSpanF16 is unreachable off amd64, like its f32 twin above.
func w4a8BatchSplitHalfSpanF16(aq []int8, aScale float32, splitHalf []byte, s16 []uint16, dst []float32, K, N, nGroups, bpr, j0, j1 int) {
	panic("linalg: w4a8BatchSplitHalfSpanF16 reached off amd64 — splitHalfUsable() should have gated this")
}
