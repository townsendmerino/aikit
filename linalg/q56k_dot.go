//go:build !amd64

package linalg

func kquantNative() bool { return false }

// dotQ5K is the per-row Q5_K dot (q56k.go): portable Go until an arch kernel replaces it here.
func dotQ5K(row []byte, aq []int8, aS []float32, sums []int32, asumf []float32, K int) float32 {
	return dotQ5KGo(row, aq, aS, sums, K)
}

// dotQ6K is the per-row Q6_K dot (q56k.go): portable Go until an arch kernel replaces it here.
func dotQ6K(row []byte, aq []int8, aS []float32, K int) float32 {
	return dotQ6KGo(row, aq, aS, K)
}
