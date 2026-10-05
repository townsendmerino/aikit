package linalg

// Strided-second-operand matmul for attention, so a consumer can feed a KV-cache head-block
// (or its transpose) to the f64-accumulated attention matmul WITHOUT first materialising a
// packed/transposed copy.
//
// WHY. A decoder's KV cache stores all heads interleaved as [nKeys, kvDim=nKV·hd]. To attend
// one head with the packed MatmulBTAcc64 (which needs contiguous [nKeys,hd] rows), goinfer
// re-copies every key row and re-transposes every value row into scratch on every token — at
// long context that gather is a measured ~10% of per-token decode time (P1). Both gathers are
// just an ADDRESSING pattern over the cache:
//
//	QKᵀ  (K re-copy):  b[j][k] = keys[j·kvDim + kvh·hd + k]   → bRowStride=kvDim, bElemStride=1
//	scores·V (V trans): b[j][k] = vals[k·kvDim + kvh·hd + j]  → bRowStride=1,     bElemStride=kvDim
//
// BIT-IDENTICAL BY CONSTRUCTION. MatmulBTAcc64 reduces each output dot in float64 in sequential
// k order (dotF32Acc64 — deliberately not vectorised, so it does not reassociate). This variant
// runs the SAME sequential f64 reduction in the SAME k order; only the address of b[j][k]
// changes. So the result is byte-for-byte identical to MatmulBTAcc64 run on a packed/transposed
// copy of the same logical b — no parity argument, a substitution. (This is the f64 path every
// live attendBatchedHeads caller uses. The f32 MatmulBT path, used only by gemma4, is a separate
// entry point: its NEON dot reassociates within the dot, so a bit-identical strided V read there
// needs a gather kernel — deferred; its K re-copy, being contiguous rows, is trivially served by
// a leading-dimension variant.)

// MatmulBTAcc64Strided is MatmulBTAcc64 — dst[M,N] = a[M,K] · b[N,K]ᵀ, each dot accumulated in
// float64 — with the second operand read as b[j][k] = bMat[bOff + j·bRowStride + k·bElemStride]
// instead of the packed bMat[j·K + k]. Packed MatmulBTAcc64 is bOff=0, bRowStride=K,
// bElemStride=1, and this returns byte-identical results to it on the same logical b.
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:M*N].
func MatmulBTAcc64Strided(a, bMat, dst []float32, M, K, N, bOff, bRowStride, bElemStride int) {
	checkMatmulBTStrided("MatmulBTAcc64Strided", len(a), len(bMat), len(dst), M, K, N, bOff, bRowStride, bElemStride)
	parallelCols(M*N*K, N, func(j0, j1 int) {
		matmulBTAcc64StridedSpan(a, bMat, dst, M, K, N, bOff, bRowStride, bElemStride, j0, j1)
	})
}

// MatmulBTAcc64Strided run through a Workspace uses its scoped threshold and worker pool, like
// the other Workspace matmuls — the shape a steady-state decode stream uses.
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:M*N].
func (w *Workspace) MatmulBTAcc64Strided(a, bMat, dst []float32, M, K, N, bOff, bRowStride, bElemStride int) {
	checkMatmulBTStrided("MatmulBTAcc64Strided", len(a), len(bMat), len(dst), M, K, N, bOff, bRowStride, bElemStride)
	w.parallelCols(M*N*K, N, func(j0, j1 int) {
		matmulBTAcc64StridedSpan(a, bMat, dst, M, K, N, bOff, bRowStride, bElemStride, j0, j1)
	})
}

// matmulBTAcc64StridedSpan is matmulBTAcc64Span with b addressed by (bOff, bRowStride,
// bElemStride). The k loop order and the float64 accumulation are identical to dotF32Acc64, so
// the packed strides reduce this to exactly matmulBTAcc64Span.
func matmulBTAcc64StridedSpan(a, bMat, dst []float32, M, K, N, bOff, bRowStride, bElemStride, j0, j1 int) {
	for i := range M {
		arow := a[i*K : i*K+K]
		drow := dst[i*N : i*N+N]
		for j := j0; j < j1; j++ {
			bBase := bOff + j*bRowStride
			var s float64
			for k := range K {
				s += float64(arow[k]) * float64(bMat[bBase+k*bElemStride])
			}
			drow[j] = float32(s)
		}
	}
}

// checkMatmulBTStrided validates the packed operands like checkMatmulBT, plus that the strided
// b view stays in bounds: the largest index read is bOff + (N-1)·bRowStride + (K-1)·bElemStride.
func checkMatmulBTStrided(kernel string, aLen, bLen, dstLen, M, K, N, bOff, bRowStride, bElemStride int) {
	if M < 0 || K < 0 || N < 0 {
		panic("linalg: " + kernel + " negative dim")
	}
	if bOff < 0 || bRowStride < 0 || bElemStride < 0 {
		panic("linalg: " + kernel + " negative b stride/offset")
	}
	requireLen(kernel, "a", aLen, mul(M, K))
	requireLen(kernel, "dst", dstLen, mul(M, N))
	if M == 0 || N == 0 || K == 0 {
		return
	}
	maxIdx := bOff + (N-1)*bRowStride + (K-1)*bElemStride
	if maxIdx >= bLen {
		panic("linalg: " + kernel + " strided b out of range")
	}
}
