//go:build amd64

package linalg

import "github.com/townsendmerino/aikit/internal/cpufeat"

// blockRows3x4 sweeps whole GROUPS OF THREE rows through the 12-accumulator kernel
// and returns the first row index it did not consume, so the caller finishes the
// 1–2 row remainder on its single-row path.
//
// Three rows because Zen 2 needs ≥10 independent accumulator chains to keep its two
// 5-cycle FMA pipes fed, and 3 rows × 4 columns is the first tile that reaches that
// (12) inside AVX2's 16 registers — see dot3x4_amd64.s. Columns advance in 4s, which
// divides the caller's 8-column tiling, so the column tail logic is unchanged.
func blockRows3x4(a, b, dst []float32, i, iEnd, K, N, k0, k4, kSpan, nStart, nEnd int) int {
	if !cpufeat.HasAVX2 || iEnd-i < 3 {
		return i
	}
	nEndAligned4 := nStart + ((nEnd-nStart)/4)*4
	var s [12]float32
	for ; i+2 < iEnd; i += 3 {
		a0, a1, a2 := &a[i*K+k0], &a[(i+1)*K+k0], &a[(i+2)*K+k0]
		n := nStart
		for ; n < nEndAligned4; n += 4 {
			dotFMA3x4(a0, a1, a2,
				&b[n*K+k0], &b[(n+1)*K+k0], &b[(n+2)*K+k0], &b[(n+3)*K+k0],
				k4*4, &s)
			for r := range 3 {
				ii := i + r
				for j := range 4 {
					sum := s[r*4+j]
					// K%4 scalar tail, exactly as the single-row path does it.
					for k := k4 * 4; k < kSpan; k++ {
						sum += a[ii*K+k0+k] * b[(n+j)*K+k0+k]
					}
					dst[ii*N+n+j] += sum
				}
			}
		}
		// Columns past the last multiple of 4 go through the single-row path, which
		// owns the <8-column and scalar tails.
		for r := range 3 {
			accumRowRange(a, b, dst, i+r, K, N, k0, k4, kSpan, n, nEnd)
		}
	}
	return i
}
