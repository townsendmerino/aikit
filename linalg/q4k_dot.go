package linalg

// dotQ4K is the per-row Q4_K dot every Q4_K matmul calls (q4k.go). Portable Go until an arch kernel
// replaces it.
func dotQ4K(row []byte, aq []int8, aS []float32, sums []int32, K int) float32 {
	return dotQ4KGo(row, aq, aS, sums, K)
}
