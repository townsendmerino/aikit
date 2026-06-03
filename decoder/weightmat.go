package decoder

import "github.com/townsendmerino/aikit/internal/linalg"

// weightMat is one [rows, cols] = [out, in] weight matrix that is either f32
// (default) or per-row int8 (after quantizeInt8). It hides the storage choice
// from the forward pass: every matmul site calls matmul, and the tied
// embedding lookup calls embedRow, regardless of precision.
type weightMat struct {
	f32    []float32 // [rows*cols], set unless quantized
	q8     []int8    // [rows*cols], set when quantized
	scales []float32 // [rows], per-row scale when quantized
	rows   int       // out features (N)
	cols   int       // in features (K)
}

func newWeightMat(f32 []float32, rows, cols int) weightMat {
	return weightMat{f32: f32, rows: rows, cols: cols}
}

// quantizeInt8 converts f32 → per-row int8 in place and frees the f32 backing
// (the M8 memory win). No-op if already quantized.
func (w *weightMat) quantizeInt8() {
	if w.q8 != nil || w.f32 == nil {
		return
	}
	w.q8, w.scales = linalg.QuantizeRowsInt8(w.f32, w.rows, w.cols)
	w.f32 = nil
}

// matmul computes dst[M, rows] = a[M, cols] · this[rows, cols]ᵀ, dispatching to
// the int8 or f32 kernel. The f32 path uses the backend (so a GPU backend can
// substitute); the int8 path is CPU-only for now.
func (w *weightMat) matmul(be Backend, a, dst []float32, M int) {
	if w.q8 != nil {
		linalg.MatmulBTQ8(a, w.q8, w.scales, dst, M, w.cols, w.rows)
		return
	}
	be.MatmulBT(a, w.f32, dst, M, w.cols, w.rows)
}

// embedRow writes row id (one token's embedding) into dst[:cols], dequantizing
// on the fly when the table is int8.
func (w *weightMat) embedRow(id int, dst []float32) {
	lo := id * w.cols
	if w.q8 != nil {
		linalg.DequantizeRowInt8(w.q8[lo:lo+w.cols], w.scales[id], dst)
		return
	}
	copy(dst, w.f32[lo:lo+w.cols])
}
