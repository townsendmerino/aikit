package linalg

// dotF32Acc64 is the dot product of two equal-length float32 vectors with the
// reduction accumulated in float64, in sequential index order. It does NOT
// reassociate, so it is bit-identical to a scalar float64 reference loop.
//
// This is deliberately not SIMD-vectorized within the dot: a 4-wide f64 lane
// accumulate would reassociate the sum (differing from the sequential reference by
// a few f64 ULP), which is exactly the bit-identicality the downstream MoE router
// relies on (see MatmulBTAcc64). The speedup of MatmulBTAcc64 over a single-threaded
// f64 matmul comes from running the N independent output dots in parallel
// (parallelCols), not from vectorizing inside a dot.
func dotF32Acc64(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}
