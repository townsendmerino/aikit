//go:build !arm64

package linalg

// q8SpanMaxCols and q8SpanBlockFloats size q8Span's column block on this arch:
// sixteen widened weight rows, fewer only where that would pass q8SpanBlockFloats
// (1 MB per worker), which no transformer K up to 16384 reaches. The width is a
// measured choice, not a cache-fit one; the sweep is in aikit PR #10.
const (
	q8SpanMaxCols     = 16
	q8SpanBlockFloats = 256 << 10
)

// q8SpanResidentFloats is the activation-block size (M·K floats, 8 MB) at or below
// which q8Span keeps the column form: there the block stays in cache, and the
// column form's one widened row in L1 measured faster than a block of sixteen.
// Above it the column form re-reads the block from DRAM per column. Both choices
// are bit-identical; the measurements are in aikit PR #10.
const q8SpanResidentFloats = 2 << 20

// q8SpanCols is how many weight rows q8Span widens at once for inner dimension K.
func q8SpanCols(K int) int {
	return max(1, min(q8SpanMaxCols, q8SpanBlockFloats/max(K, 1)))
}

// q8SpanScratchLen is the length of the widened-row scratch q8Span needs for K.
func q8SpanScratchLen(K int) int { return q8SpanCols(K) * K }

// q8Span computes dst[i,j] for every row i and column j in [j0,j1) as
// dotF32(a_i, widen(bQ_j))·bScales[j]. Past q8SpanResidentFloats it widens a
// block of q8SpanCols(K) weight rows, then walks each activation row against all
// of them, so the activation block is read once per column block rather than once
// per column (q8SpanColumn).
// Every output is the same dotF32 call on the same values followed by the same
// scale multiply; only the order in which independent outputs are computed
// changes, so the result is bit-identical to q8SpanColumn
// (TestQ8Span8Cols_bitIdenticalToColumnForm). arm64 has its own eight-column
// form in q8span_arm64.go. deq holds at least q8SpanScratchLen(K) floats.
func q8Span(a []float32, bQ []int8, bScales, dst []float32, M, K, N, j0, j1 int, deq []float32) {
	jb := q8SpanCols(K)
	if M*K <= q8SpanResidentFloats {
		jb = 1
	}
	q8SpanBlocked(a, bQ, bScales, dst, M, K, N, j0, j1, jb, deq)
}

// q8SpanBlocked is q8Span with the column-block width jb given; deq holds at
// least jb*K floats.
func q8SpanBlocked(a []float32, bQ []int8, bScales, dst []float32, M, K, N, j0, j1, jb int, deq []float32) {
	if jb == 1 {
		for j := j0; j < j1; j++ {
			q8SpanColumn(a, bQ, bScales, dst, M, K, N, j, deq[:K])
		}
		return
	}
	for j := j0; j < j1; j += jb {
		nb := min(jb, j1-j)
		for r := range nb {
			dequantRowInt8(deq[r*K:r*K+K], bQ[(j+r)*K:(j+r+1)*K], 1.0)
		}
		scales := bScales[j : j+nb]
		for i := range M {
			arow := a[i*K : i*K+K]
			drow := dst[i*N+j : i*N+j+nb]
			for r, s := range scales {
				drow[r] = dotF32(arow, deq[r*K:r*K+K]) * s
			}
		}
	}
}
