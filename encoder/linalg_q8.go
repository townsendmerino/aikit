package encoder

// matmulBTQ8 is the M8 int8-weight variant of matmulBT. Same shape:
// dst = a · bᵀ where a is [M, K] f32 (activations) and b is logically
// [N, K] f32 stored as int8 quantized rows + per-row f32 scales:
//
//	a       — [M, K] f32 row-major (activations, NOT quantized)
//	bQ      — [N*K] int8 row-major (the quantized [N, K] matrix)
//	bScales — [N] f32 (per-row scale: row n's f32 value ≈ float32(bQ[n,k]) * bScales[n])
//	dst     — [M, N] f32 row-major, freshly allocated
//
// The kernel is the M3 blocked-matmul with one twist: weight reads come
// from a tightly-packed int8 array (4× less memory bandwidth than f32),
// the multiply-accumulate happens in f32 (each int8 weight gets
// converted to f32 inside the inner loop), and the final accumulator
// is scaled by the row's bScale once per (i, n) tile cell at write-back.
//
// Why this saves time: at M3's blocked-kernel measurement, the GEMM
// was bandwidth-bound on weight reads at ~6.5 GFLOP/s. Reducing weight
// bytes 4× pushes the bound out by ~3× (memory subsystem can deliver
// 4× more weights per cycle, but per-multiply work is unchanged; the
// f32×f32 multiply itself doesn't get faster). Empirically the win
// lands closer to 2× than 4× because Go's compiler doesn't auto-SIMD
// the int8-to-f32 conversion as tightly as the pure-f32 inner loop.
//
// Dispatch: matmulBTQ8 is always blocked (the M8 path is only
// triggered for the big linear layers Wqkv/OutProj/fc11/fc12/fc2,
// all of which have M*K*N ≫ the matmulBT small-shape threshold).
func matmulBTQ8(a []float32, bQ []int8, bScales []float32, M, K, N int) []float32 {
	dst := make([]float32, M*N)
	matmulBTQ8Into(dst, a, bQ, bScales, M, K, N, make([]float32, N*K))
	return dst
}

// matmulBTQ8Into is matmulBTQ8 writing into a caller-supplied dst[:M*N] and using a
// caller-supplied deqW[:N*K] weight-dequant buffer — both pooled in the q8 forward's
// scratch. It widens each int8 weight to f32 ONCE (N*K) into deqW, then runs the
// vectorized f32 matmulBTInto.
//
// This replaced a scalar blocked kernel that did the int8→f32 widen INLINE in the
// GEMM (so the conversion ran M times per weight), which measured ~26× slower than
// the f32 SIMD matmul and ~36× slower than the SDOT W8A8 kernel on the Wqkv shape —
// the actual reason LoadQ8 was ~5× slower than Load, NOT the allocation churn the
// pooling fix already removed. Dequant-then-SIMD keeps the weight-only numerics
// exactly (cosine vs f32 unchanged at 0.997), unlike full W8A8 which quantizes
// activations and fell below the 0.97 reranker bar; the weights stay int8 in storage
// (¼ the //go:embed footprint) — deqW is transient runtime scratch only.
func matmulBTQ8Into(dst, a []float32, bQ []int8, bScales []float32, M, K, N int, deqW []float32) {
	if len(a) != M*K || len(bQ) != N*K || len(bScales) != N || len(dst) < M*N || len(deqW) < N*K {
		panic("encoder: matmulBTQ8Into shape mismatch")
	}
	w := deqW[:N*K]
	for n := 0; n < N; n++ {
		sc := bScales[n]
		row := w[n*K : n*K+K]
		bq := bQ[n*K : n*K+K]
		for k := range row {
			row[k] = float32(bq[k]) * sc
		}
	}
	matmulBTInto(a, w, dst, M, K, N)
}
