//go:build amd64

package linalg

import "github.com/townsendmerino/aikit/internal/cpufeat"

//go:noescape
func dotQ4KAVX2(row *byte, aq *int8, aS *float32, asumf *float32, nSB int) float32

// dotQ4K is the per-row Q4_K dot (q4k.go): the AVX2 kernel when available, the Go oracle otherwise.
// asumf[g] = aS[g]·sums[g], precomputed once per activation row by matmulQ4K.
func dotQ4K(row []byte, aq []int8, aS []float32, sums []int32, asumf []float32, K int) float32 {
	if cpufeat.HasQ4KAVX2 && K >= qkK {
		return dotQ4KAVX2(&row[0], &aq[0], &aS[0], &asumf[0], K/qkK)
	}
	return dotQ4KGo(row, aq, aS, sums, K)
}
