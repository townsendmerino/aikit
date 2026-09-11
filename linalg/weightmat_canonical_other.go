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
func (w *WeightMat) MatmulBTW4A8Into(ws *Workspace, a, dst []float32, M int) {
	if w.q4 == nil {
		panic(fmt.Sprintf("linalg: WeightMat.MatmulBTW4A8Into: no canonical int4 bytes (rows=%d cols=%d) "+
			"and this target has no repacked layout to fall back to", w.rows, w.cols))
	}
	MatmulBTW4A8Into(ws, a, w.q4, w.q4s, dst, M, w.cols, w.rows, w.group)
}

// RepackInt4SplitHalf is a no-op off amd64: the split-half layout's only consumer is the AVX2
// kernel. Always returns false; w is never modified.
func (w *WeightMat) RepackInt4SplitHalf() bool { return false }

// splitHalfUsable is always false off amd64 — see the arm64 file's twin for
// the same reasoning. Audit M-22.
func splitHalfUsable() bool { return false }
