//go:build amd64

package linalg

//go:noescape
func dotQ4KAVX2(row *byte, aq *int8, aS *float32, asumf *float32, nSB int) float32

// hasQ4KAVX2: the AVX2 Q4_K kernel also needs F16C (VCVTPH2PS) for the super-block's d and dmin.
var hasQ4KAVX2 = hasAVX2 && detectF16C()

// detectF16C reports CPUID.1:ECX bit 29 (every AVX2 CPU ships it; checked rather than assumed).
func detectF16C() bool {
	_, _, ecx1, _ := cpuid(1, 0)
	return ecx1&(1<<29) != 0
}

// dotQ4K is the per-row Q4_K dot (q4k.go): the AVX2 kernel when available, the Go oracle otherwise.
// asumf[g] = aS[g]·sums[g], precomputed once per activation row by matmulQ4K.
func dotQ4K(row []byte, aq []int8, aS []float32, sums []int32, asumf []float32, K int) float32 {
	if hasQ4KAVX2 && K >= qkK {
		return dotQ4KAVX2(&row[0], &aq[0], &aS[0], &asumf[0], K/qkK)
	}
	return dotQ4KGo(row, aq, aS, sums, K)
}
