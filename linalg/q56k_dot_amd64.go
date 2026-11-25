//go:build amd64

package linalg

import "github.com/townsendmerino/aikit/internal/cpufeat"

//go:noescape
func dotQ5KAVX2(row *byte, aq *int8, aS *float32, asumf *float32, nSB int) float32

//go:noescape
func dotQ6KAVX2(row *byte, aq *int8, aS *float32, nSB int) float32

func kquantNative() bool { return cpufeat.HasQ4KAVX2 }

// dotQ5K is the per-row Q5_K dot (q56k.go): the AVX2 kernel when available (it needs Q4_K's AVX2 and
// F16C), the Go oracle otherwise. asumf[g] = aS[g]·sums[g], precomputed once per activation row.
func dotQ5K(row []byte, aq []int8, aS []float32, sums []int32, asumf []float32, K int) float32 {
	if cpufeat.HasQ4KAVX2 && K >= qkK {
		return dotQ5KAVX2(&row[0], &aq[0], &aS[0], &asumf[0], K/qkK)
	}
	return dotQ5KGo(row, aq, aS, sums, K)
}

// dotQ6K is the per-row Q6_K dot (q56k.go): the AVX2 kernel when available, the Go oracle otherwise.
func dotQ6K(row []byte, aq []int8, aS []float32, K int) float32 {
	if cpufeat.HasQ4KAVX2 && K >= qkK {
		return dotQ6KAVX2(&row[0], &aq[0], &aS[0], K/qkK)
	}
	return dotQ6KGo(row, aq, aS, K)
}
