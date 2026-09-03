//go:build !arm64

package linalg

// q8SpanScratchRows is how many widened weight rows q8Span's scratch holds on this
// arch: one — the span runs a column at a time through dotF32.
const q8SpanScratchRows = 1

// q8Span computes dst[i,j] for every row i and column j in [j0,j1) as
// dotF32(a_i, widen(bQ_j))·bScales[j], one column at a time (q8SpanColumn). arm64
// has the eight-column form in q8span_arm64.go; it is arm64-only because its
// bit-identity argument rests on dotNEON8x4 and dotNEON4 sharing one lane
// order and one fold, which the amd64 kernels do not.
func q8Span(a []float32, bQ []int8, bScales, dst []float32, M, K, N, j0, j1 int, deq []float32) {
	for j := j0; j < j1; j++ {
		q8SpanColumn(a, bQ, bScales, dst, M, K, N, j, deq)
	}
}
