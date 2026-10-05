package linalg

// Bulk int8→float32 weight widening.
//
// The q8 encoder path stores weights as int8 rows plus a per-row f32 scale and
// reconstructs w[n,k] = float32(q[n,k]) * scale[n] before the f32 GEMM. That
// widen is O(N·K) per matmul call and independent of the sequence length, so it
// does not amortize: measured on a 3700X it is ~190 ms per CodeRankEmbed forward
// (113 M elements over 12 layers × 5 matmuls) whether the input is 10 tokens or
// 350 — 35% of a short forward and 2% of a long one (perf-campaign item 22).
//
// Scalar Go compiles this to a load / sign-extend / int→float convert / multiply
// / store chain per element and does not vectorize it. The AVX2 path does 32
// elements per iteration.
//
// BIT-IDENTICAL to the scalar loop by construction: the value is one float32
// multiply of an exactly-representable integer by the scale, in both paths.
// TestDequantizeRowsInt8_bitIdentical asserts it.

// DequantizeRowsInt8Into widens `rows` int8 rows of `cols` elements into dst:
//
//	dst[n*cols+k] = float32(q[n*cols+k]) * scales[n]
//
// dst must have len ≥ rows*cols, q len ≥ rows*cols, scales len ≥ rows.
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:rows*cols].
func DequantizeRowsInt8Into(dst []float32, q []int8, scales []float32, rows, cols int) {
	if rows < 0 || cols < 0 {
		panic("linalg: DequantizeRowsInt8Into negative dims")
	}
	n := rows * cols
	if len(dst) < n || len(q) < n || len(scales) < rows {
		panic("linalg: DequantizeRowsInt8Into short buffer")
	}
	for r := range rows {
		dequantRowInt8(dst[r*cols:r*cols+cols], q[r*cols:r*cols+cols], scales[r])
	}
}

// dequantRowInt8Scalar is the portable reference: one row, one scale.
func dequantRowInt8Scalar(dst []float32, q []int8, scale float32) {
	for k, c := range q {
		dst[k] = float32(c) * scale
	}
}
