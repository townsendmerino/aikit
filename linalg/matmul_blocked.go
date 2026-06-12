package linalg

import "sync"

// Cache-blocked + register-blocked f32 GEMM (dst[M,N] = a[M,K]·b[N,K]ᵀ, the BT /
// PyTorch [out,in] layout). This is the shared home for the blocked kernel — both the
// public MatmulBT / MatmulBTInto and the encoder's transformer matmuls route through it,
// so there is one tiling implementation, not two. It was hoisted out of the encoder once
// the public MatmulBT (a naive dot-per-output span that re-streamed b from DRAM per
// a-row) measured ~7% of peak at prefill shapes — every kit consumer of MatmulBT, not
// just the encoder, was paying for the missing cache blocking.

// Tile defaults. Tuned via the M10 sweep for the 8x4 NEON kernel (BenchmarkMatmulTile_*).
//
// Why 32×32×768: kBlock=768 matches the transformer hidden dim, so K=768 projections run
// a single k-tile (zero k-loop overhead) and K=3072 (fc2) takes 4 tiles of 768 — far less
// overhead than 24 tiles of 128. A 32×768 a-tile (96 KB) + 32×768 b-tile (96 KB) + 32×32
// output (4 KB) fit an M-series P-core L1 for the whole nBlock iteration. kBlock MUST be a
// multiple of 4 (NEON 4-lane load); nBlock benefits from a multiple of 8 (the 8x4 unroll).
const (
	mBlockDefault = 32
	nBlockDefault = 32
	kBlockDefault = 768
)

// blockedMACThreshold is the MAC count (M*K*N) below which MatmulBT stays on the naive
// dot-per-output span: the blocked kernel's tiling/prologue overhead isn't worth it for
// tiny matmuls (e.g. an attention QKᵀ at short sequence length). Matches the encoder's
// long-standing naive/blocked cutoff.
const blockedMACThreshold = 4_000_000

func zeroSpanF32(s []float32) {
	for i := range s {
		s[i] = 0
	}
}

// MatmulBTInto computes dst[M,N] = a[M,K]·b[N,K]ᵀ via the cache+register-blocked kernel,
// SERIALLY (no goroutines), overwriting dst (which must have len ≥ M*N). It is the entry
// for callers that own their own parallelism — the encoder row-splits a batch across
// cores and wants each matmul serial; goinfer's batch/vision paths likewise. For
// process-level column parallelism use MatmulBT. Experimental surface.
func MatmulBTInto(dst, a, b []float32, M, K, N int) {
	if len(dst) < M*N {
		panic("linalg: MatmulBTInto dst too small")
	}
	zeroSpanF32(dst[:M*N])
	blockedFill(a, b, dst, M, K, N, 0, N, mBlockDefault, nBlockDefault, kBlockDefault)
}

// blockedFill accumulates a[M,K]·b[N,K]ᵀ into dst[i, n] for rows i∈[0,M) and columns
// n∈[nStart,nEnd). dst MUST be zero in that region at entry (the k-tile loop accumulates).
// Splitting on the column range is what lets MatmulBT fan the same kernel across workers
// (each owns a disjoint [j0,j1)); MatmulBTInto calls it once over [0,N). On arm64 it folds
// row pairs through the 2×8 register kernel (Dot2x8); the odd final row, the <8-col tail,
// and all non-arm64 use the Dot8x4/Dot4x4/scalar path via accumRowRange.
func blockedFill(a, b, dst []float32, M, K, N, nStart, nEnd, mBlock, nBlock, kBlock int) {
	// Large K: pack b's 8-row groups into a contiguous low-stride buffer first. At large
	// K the 8 b-rows a Dot2x8 reads simultaneously sit K·4 bytes apart and collide in L1
	// cache sets (associativity conflicts) — packing them ~kBlock·4 apart kills that,
	// lifting K=4096 prefill 46%→68% and the encoder's own K=3072 fc2 +15%. Packing copies
	// the same b values in the same order, so it stays BIT-IDENTICAL to the unpacked path
	// (verified against the encoder golden parity). Below packKThreshold the copy cost
	// isn't worth it (K=768 b-rows are already close), so those keep the unpacked path.
	// has2x8Kernel gates packing to arm64: packedFill runs Dot2x8/Dot8x4 over the packed
	// buffer, and Dot2x8 is the NEON asm only on arm64 (the scalar fallback elsewhere would
	// be slower than amd64's AVX2 unpacked path). The AVX2 packed path waits on §2.4.
	if has2x8Kernel && K >= packKThreshold && nEnd-nStart >= 8 {
		packedFill(a, b, dst, M, K, N, nStart, nEnd, packKBlockFor(K))
		return
	}
	for i0 := 0; i0 < M; i0 += mBlock {
		iEnd := min(i0+mBlock, M)
		for n0 := nStart; n0 < nEnd; n0 += nBlock {
			nTileEnd := min(n0+nBlock, nEnd)
			for k0 := 0; k0 < K; k0 += kBlock {
				kEnd := min(k0+kBlock, K)
				kSpan := kEnd - k0
				k4 := kSpan / 4
				i := i0
				if has2x8Kernel {
					// arm64: fold row PAIRS through the 2×8 register kernel — each
					// b-load feeds 2 FMLAs (vs 1 in the 1×8 Dot8x4) across 16 live
					// accumulators. 8-col groups go through Dot2x8; the <8-col n-tail
					// (and any odd final row, below) reuse accumRowRange.
					nEndAligned8 := n0 + ((nTileEnd-n0)/8)*8
					var s [64]float32
					for ; i+1 < iEnd; i += 2 {
						a0 := &a[i*K+k0]
						a1 := &a[(i+1)*K+k0]
						n := n0
						for ; n < nEndAligned8; n += 8 {
							Dot2x8(a0, a1,
								&b[n*K+k0], &b[(n+1)*K+k0], &b[(n+2)*K+k0], &b[(n+3)*K+k0],
								&b[(n+4)*K+k0], &b[(n+5)*K+k0], &b[(n+6)*K+k0], &b[(n+7)*K+k0],
								k4, &s)
							for r := 0; r < 2; r++ {
								ii := i + r
								base := r * 32
								for j := 0; j < 8; j++ {
									sum := s[base+j*4] + s[base+j*4+1] + s[base+j*4+2] + s[base+j*4+3]
									for k := k4 * 4; k < kSpan; k++ {
										sum += a[ii*K+k0+k] * b[(n+j)*K+k0+k]
									}
									dst[ii*N+n+j] += sum
								}
							}
						}
						accumRowRange(a, b, dst, i, K, N, k0, k4, kSpan, n, nTileEnd)
						accumRowRange(a, b, dst, i+1, K, N, k0, k4, kSpan, n, nTileEnd)
					}
				}
				for ; i < iEnd; i++ {
					accumRowRange(a, b, dst, i, K, N, k0, k4, kSpan, n0, nTileEnd)
				}
			}
		}
	}
}

// accumRowRange computes dst[i, n] += Σ_{k∈[k0,k0+kSpan)} a[i,k]·b[n,k] for n∈[nStart,
// nEnd), via the 8-col Dot8x4 kernel, a 4-col Dot4x4 tail, and a scalar column tail. k4 =
// kSpan/4. The single-row body, shared by blockedFill's odd-row and <8-col-tail cases.
func accumRowRange(a, b, dst []float32, i, K, N, k0, k4, kSpan, nStart, nEnd int) {
	aRowPtr := &a[i*K+k0]
	n := nStart
	nEndAligned8 := nStart + ((nEnd-nStart)/8)*8
	var sums8 [32]float32
	for ; n < nEndAligned8; n += 8 {
		Dot8x4(aRowPtr,
			&b[n*K+k0], &b[(n+1)*K+k0], &b[(n+2)*K+k0], &b[(n+3)*K+k0],
			&b[(n+4)*K+k0], &b[(n+5)*K+k0], &b[(n+6)*K+k0], &b[(n+7)*K+k0],
			k4, &sums8)
		s0 := sums8[0] + sums8[1] + sums8[2] + sums8[3]
		s1 := sums8[4] + sums8[5] + sums8[6] + sums8[7]
		s2 := sums8[8] + sums8[9] + sums8[10] + sums8[11]
		s3 := sums8[12] + sums8[13] + sums8[14] + sums8[15]
		s4 := sums8[16] + sums8[17] + sums8[18] + sums8[19]
		s5 := sums8[20] + sums8[21] + sums8[22] + sums8[23]
		s6 := sums8[24] + sums8[25] + sums8[26] + sums8[27]
		s7 := sums8[28] + sums8[29] + sums8[30] + sums8[31]
		for k := k4 * 4; k < kSpan; k++ {
			av := a[i*K+k0+k]
			s0 += av * b[n*K+k0+k]
			s1 += av * b[(n+1)*K+k0+k]
			s2 += av * b[(n+2)*K+k0+k]
			s3 += av * b[(n+3)*K+k0+k]
			s4 += av * b[(n+4)*K+k0+k]
			s5 += av * b[(n+5)*K+k0+k]
			s6 += av * b[(n+6)*K+k0+k]
			s7 += av * b[(n+7)*K+k0+k]
		}
		dst[i*N+n+0] += s0
		dst[i*N+n+1] += s1
		dst[i*N+n+2] += s2
		dst[i*N+n+3] += s3
		dst[i*N+n+4] += s4
		dst[i*N+n+5] += s5
		dst[i*N+n+6] += s6
		dst[i*N+n+7] += s7
	}
	nEndAligned4 := nStart + ((nEnd-nStart)/4)*4
	var sums [16]float32
	for ; n < nEndAligned4; n += 4 {
		Dot4x4(aRowPtr,
			&b[n*K+k0], &b[(n+1)*K+k0], &b[(n+2)*K+k0], &b[(n+3)*K+k0],
			k4, &sums)
		s0 := sums[0] + sums[1] + sums[2] + sums[3]
		s1 := sums[4] + sums[5] + sums[6] + sums[7]
		s2 := sums[8] + sums[9] + sums[10] + sums[11]
		s3 := sums[12] + sums[13] + sums[14] + sums[15]
		for k := k4 * 4; k < kSpan; k++ {
			av := a[i*K+k0+k]
			s0 += av * b[n*K+k0+k]
			s1 += av * b[(n+1)*K+k0+k]
			s2 += av * b[(n+2)*K+k0+k]
			s3 += av * b[(n+3)*K+k0+k]
		}
		dst[i*N+n+0] += s0
		dst[i*N+n+1] += s1
		dst[i*N+n+2] += s2
		dst[i*N+n+3] += s3
	}
	for ; n < nEnd; n++ {
		var s float32
		for k := 0; k < kSpan; k++ {
			s += a[i*K+k0+k] * b[n*K+k0+k]
		}
		dst[i*N+n] += s
	}
}

// B-panel packing for large K (see the dispatch in blockedFill). packKThreshold is the K
// above which packing's copy cost is repaid by avoiding L1 associativity conflicts.
const packKThreshold = 2048

// packKBlockFor picks the packed k-tile. When K is a multiple of the unpacked default
// (768 — the transformer dims), packing reuses 768 so the k-tiling, and thus the f32
// accumulation order, is IDENTICAL to the unpacked path — the encoder's K=3072 fc2 stays
// bit-for-bit equal. For other K (e.g. 4096) there is no bit-identity constraint (not an
// encoder shape), so it uses 1024, the measured large-K optimum (4096 prefill 61%→68%).
func packKBlockFor(K int) int {
	if K%kBlockDefault == 0 {
		return kBlockDefault
	}
	return 1024
}

// packBufPool recycles the 8×kBlock pack buffer (≤ 8·1024 f32 = 32 KB). MatmulBT fans
// packedFill across goroutines; each gets its own buffer from the pool.
var packBufPool = sync.Pool{New: func() any { s := make([]float32, 8*1024); return &s }}

// packedFill is blockedFill's large-K path: for each contiguous group of 8 output columns
// it copies those 8 b-rows' k-strip into a low-stride buffer (rows kBlock apart, not K
// apart) and runs Dot2x8/Dot8x4 over the packed buffer — same kernels, same accumulation
// order, just conflict-free b loads. The loop is n→k→pack→i so the pack is amortised over
// all M rows. The <8-column remainder (only when N%8≠0; never for the transformer dims)
// falls back to the unpacked path. dst must be zero in [nStart,nEnd) at entry.
func packedFill(a, b, dst []float32, M, K, N, nStart, nEnd, kBlock int) {
	bufp := packBufPool.Get().(*[]float32)
	defer packBufPool.Put(bufp)
	bPack := (*bufp)[:8*kBlock]

	n0 := nStart
	for ; n0+8 <= nEnd; n0 += 8 {
		for k0 := 0; k0 < K; k0 += kBlock {
			kEnd := min(k0+kBlock, K)
			kSpan := kEnd - k0
			k4 := kSpan / 4
			for bi := 0; bi < 8; bi++ {
				copy(bPack[bi*kSpan:bi*kSpan+kSpan], b[(n0+bi)*K+k0:(n0+bi)*K+kEnd])
			}
			p0, p1, p2, p3 := &bPack[0], &bPack[kSpan], &bPack[2*kSpan], &bPack[3*kSpan]
			p4, p5, p6, p7 := &bPack[4*kSpan], &bPack[5*kSpan], &bPack[6*kSpan], &bPack[7*kSpan]
			i := 0
			var s [64]float32
			for ; i+1 < M; i += 2 {
				Dot2x8(&a[i*K+k0], &a[(i+1)*K+k0], p0, p1, p2, p3, p4, p5, p6, p7, k4, &s)
				for r := 0; r < 2; r++ {
					ii := i + r
					base := r * 32
					for j := 0; j < 8; j++ {
						sum := s[base+j*4] + s[base+j*4+1] + s[base+j*4+2] + s[base+j*4+3]
						for k := k4 * 4; k < kSpan; k++ {
							sum += a[ii*K+k0+k] * b[(n0+j)*K+k0+k]
						}
						dst[ii*N+n0+j] += sum
					}
				}
			}
			for ; i < M; i++ {
				var s8 [32]float32
				Dot8x4(&a[i*K+k0], p0, p1, p2, p3, p4, p5, p6, p7, k4, &s8)
				for j := 0; j < 8; j++ {
					sum := s8[j*4] + s8[j*4+1] + s8[j*4+2] + s8[j*4+3]
					for k := k4 * 4; k < kSpan; k++ {
						sum += a[i*K+k0+k] * b[(n0+j)*K+k0+k]
					}
					dst[i*N+n0+j] += sum
				}
			}
		}
	}
	if n0 < nEnd { // <8-column remainder via the unpacked path
		blockedFill(a, b, dst, M, K, N, n0, nEnd, mBlockDefault, nBlockDefault, kBlockDefault)
	}
}
