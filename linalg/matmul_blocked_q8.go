package linalg

import "sync"

// Fused int8-weight blocked GEMM: dst[M,N] = a[M,K] · dequant(bQ,bScales)[N,K]ᵀ,
// widening each int8 weight to f32 INSIDE the b-panel pack instead of materializing
// the whole [N,K] f32 matrix first (perf-campaign item 22 fix (b)).
//
// The non-fused path (encoder's matmulBTQ8Into) widens the entire weight matrix into
// a pooled deqW buffer — up to 9.4 MB per matmul, far past L2 — then runs the f32
// GEMM over it: ~0.45 GB of stores + 0.45 GB of reads per CodeRankEmbed forward, all
// of it round-tripping through DRAM. packedFill already copies b's 8-row groups into
// a small (≤32 KB, L1-resident) conflict-free buffer before the kernel; this widens
// int8→f32 during that copy, so the f32 weights exist only as the L1 pack tile and the
// DRAM round-trip is gone. Only the int8 weights (¼ the bytes) are streamed.
//
// BIT-IDENTICAL to DequantizeRowsInt8Into + MatmulBTInto: the widen value is the same
// float32(int8)*scale (same dequantRowInt8), the k-tiling matches the reference
// (kBlockFusedQ8), and the Dot2x8/Dot8x4 kernels see the same b values in the same
// order — so every dst[i,n] reduces identically. TestMatmulBTQ8Fused_bitIdentical
// asserts exact equality over the real encoder shapes, mutation-checked.
//
// arm64 only in practice: it reuses packedFill's Dot2x8 (the NEON asm), and the
// encoder gates the fused path on has2x8Kernel && N%8==0. On amd64 / odd N the caller
// keeps the dequant-then-f32-GEMM path.

// FusedQ8Applies reports whether the fused widen-in-pack kernel should run for this
// shape instead of dequant-then-f32-GEMM. It needs the NEON packed kernel (arm64) and
// N%8==0 (every encoder Q8 shape: N∈{768,2304,3072}) so there is no <8-column tail.
//
// packedFill trades blockedFill's m-blocking for a conflict-free b-layout, re-streaming
// the a-panel once per 8-column group — which collapses throughput when a single call
// runs a large M (perf item 23's deferred m-blocking gap). The encoder's dispatch never
// lets that happen: small M goes column-parallel (a-panel small), large M goes
// ROW-parallel so each worker runs the fused kernel over a small row block. With that,
// fusing every K shape beat the dequant baseline at every sequence length measured
// (L=8→512, −25% at L=8 down to −2% at L=512 on an M1 Pro) — so there is no K or M
// threshold here; the parallel axis, chosen upstream, is what keeps it a win.
func FusedQ8Applies(K, N int) bool {
	if has2x8Kernel {
		return HasFusedQ8Kernel && N%8 == 0
	}
	// Off arm64 the fusion is the N-STRIP form (fusedQ8Strip), which reuses
	// blockedFill unchanged and so places no constraint on N (audit M-23).
	return HasFusedQ8Kernel
}

// kBlockFusedQ8 picks the k-tile so the fused reduction order matches the reference
// (DequantizeRowsInt8Into + MatmulBTInto). Below packKThreshold the reference runs the
// unpacked blockedFill at kBlockDefault; at/above it runs packedFill at packKBlockFor.
// Mirroring both keeps the f32 accumulation order — and thus the bits — identical.
func kBlockFusedQ8(K int) int {
	// Mirror the REFERENCE's condition exactly, has2x8Kernel included:
	// blockedFill only takes the packed path when has2x8Kernel is true (see its
	// gate), so off arm64 the reference is always the unpacked kBlockDefault
	// path and the fused form must match that to stay bit-identical.
	if has2x8Kernel && K >= packKThreshold {
		return packKBlockFor(K)
	}
	return kBlockDefault
}

func checkMatmulBTQ8(name string, la, lbq, lbs, ldst, M, K, N int) {
	if M < 0 || K < 0 || N < 0 {
		panic("linalg: " + name + " negative dims")
	}
	if la < M*K || lbq < N*K || lbs < N || ldst < M*N {
		panic("linalg: " + name + " short buffer")
	}
}

// MatmulBTQ8FusedInto computes dst[M,N] = a[M,K]·dequant(bQ,bScales)[N,K]ᵀ SERIALLY,
// overwriting dst (len ≥ M*N), widening int8→f32 inside the pack. For callers that own
// their parallelism (the encoder row-splits a batch and wants each matmul serial).
// Requires N a multiple of 8 (every encoder Q8 shape is); the caller routes odd N
// through the dequant-then-f32-GEMM fallback.
func MatmulBTQ8FusedInto(dst, a []float32, bQ []int8, bScales []float32, M, K, N int) {
	checkMatmulBTQ8("MatmulBTQ8FusedInto", len(a), len(bQ), len(bScales), len(dst), M, K, N)
	zeroSpanF32(dst[:M*N])
	fillQ8(a, bQ, bScales, dst, M, K, N, 0, N, kBlockFusedQ8(K))
}

// MatmulBTQ8Fused is the column-parallel sibling (mirrors MatmulBT), for a lone forward
// that wants intra-op parallelism. Each worker widen-packs its own disjoint column
// range, so no shared f32 weight buffer exists at all.
func MatmulBTQ8Fused(dst, a []float32, bQ []int8, bScales []float32, M, K, N int) {
	checkMatmulBTQ8("MatmulBTQ8Fused", len(a), len(bQ), len(bScales), len(dst), M, K, N)
	zeroSpanF32(dst[:M*N])
	kb := kBlockFusedQ8(K)
	parallelCols(M*N*K, N, func(j0, j1 int) {
		fillQ8(a, bQ, bScales, dst, M, K, N, j0, j1, kb)
	})
}

// packedFillQ8 is packedFill's int8-weight twin: for each contiguous group of 8 output
// columns it WIDENS those 8 b-rows' k-strip (int8→f32, ×row scale) into the low-stride
// pack buffer — the widen replacing packedFill's plain copy — then runs the same
// Dot2x8/Dot8x4 over it. [nStart,nEnd) must be 8-aligned (the encoder's N%8==0 shapes,
// and parallelSpawnCols shards on 8-column boundaries), so there is no <8 remainder;
// packedFillQ8Tail is the correctness path for an odd tail, which no encoder shape hits.
func packedFillQ8(a []float32, bQ []int8, bScales []float32, dst []float32, M, K, N, nStart, nEnd, kBlock int) {
	bufp := packBufPool.Get().(*[]float32)
	defer packBufPool.Put(bufp)
	bPack := *bufp

	n0 := nStart
	for ; n0+8 <= nEnd; n0 += 8 {
		for k0 := 0; k0 < K; k0 += kBlock {
			kEnd := min(k0+kBlock, K)
			kSpan := kEnd - k0
			k4 := kSpan / 4
			// Space the rows by a cache line when kSpan is a power of two (item 24),
			// exactly as packedFill does — bit-identity + the same conflict avoidance.
			stride := kSpan
			if kSpan&(kSpan-1) == 0 {
				stride += packStridePad
			}
			// Widen int8→f32 straight into the pack tile (the fusion). dequantRowInt8
			// is the same kernel DequantizeRowsInt8Into uses, so the f32 bits match.
			for bi := range 8 {
				dequantRowInt8(bPack[bi*stride:bi*stride+kSpan], bQ[(n0+bi)*K+k0:(n0+bi)*K+kEnd], bScales[n0+bi])
			}
			p0, p1, p2, p3 := &bPack[0], &bPack[stride], &bPack[2*stride], &bPack[3*stride]
			p4, p5, p6, p7 := &bPack[4*stride], &bPack[5*stride], &bPack[6*stride], &bPack[7*stride]
			i := 0
			var s [64]float32
			for ; i+1 < M; i += 2 {
				Dot2x8(&a[i*K+k0], &a[(i+1)*K+k0], p0, p1, p2, p3, p4, p5, p6, p7, k4, &s)
				for r := range 2 {
					ii := i + r
					base := r * 32
					for j := range 8 {
						sum := s[base+j*4] + s[base+j*4+1] + s[base+j*4+2] + s[base+j*4+3]
						// k-remainder from the pack tile (widened), not the int8 source.
						for k := k4 * 4; k < kSpan; k++ {
							sum += a[ii*K+k0+k] * bPack[j*stride+k]
						}
						dst[ii*N+n0+j] += sum
					}
				}
			}
			for ; i < M; i++ {
				var s8 [32]float32
				Dot8x4(&a[i*K+k0], p0, p1, p2, p3, p4, p5, p6, p7, k4, &s8)
				for j := range 8 {
					sum := s8[j*4] + s8[j*4+1] + s8[j*4+2] + s8[j*4+3]
					for k := k4 * 4; k < kSpan; k++ {
						sum += a[i*K+k0+k] * bPack[j*stride+k]
					}
					dst[i*N+n0+j] += sum
				}
			}
		}
	}
	if n0 < nEnd { // <8-column remainder — unreachable for N%8==0 encoder shapes
		packedFillQ8Tail(a, bQ, bScales, dst, M, K, N, n0, nEnd)
	}
}

// packedFillQ8Tail handles a <8-column remainder by widening those columns and running
// the scalar dot. It is the correctness path for N%8≠0, which no encoder Q8 shape hits
// (D=768 ⇒ N∈{768,2304,3072}, all multiples of 8), so it is never on the measured path;
// unlike the 8-group body it does not reproduce the kernel's tiled reduction bit-for-bit.
func packedFillQ8Tail(a []float32, bQ []int8, bScales []float32, dst []float32, M, K, N, nStart, nEnd int) {
	wcol := make([]float32, K)
	for n := nStart; n < nEnd; n++ {
		dequantRowInt8(wcol, bQ[n*K:(n+1)*K], bScales[n])
		for i := range M {
			var s float32
			aRow := a[i*K : i*K+K]
			for k := range K {
				s += aRow[k] * wcol[k]
			}
			dst[i*N+n] += s
		}
	}
}

// fillQ8 dispatches the fused int8-weight fill to the form that matches this
// architecture's f32 reference, so both stay bit-identical to
// DequantizeRowsInt8Into + MatmulBTInto on their own arch.
//
// arm64 packs 8 b-rows into a low-stride tile and runs Dot2x8/Dot8x4 over it,
// mirroring blockedFill's packed path. Off arm64 blockedFill never packs — its
// pack gate is has2x8Kernel — so the packed form would neither match the
// reference's reduction order nor use a real kernel (Dot2x8 is the scalar
// fallback there). fusedQ8Strip is the form that does match.
func fillQ8(a []float32, bQ []int8, bScales []float32, dst []float32, M, K, N, nStart, nEnd, kBlock int) {
	if has2x8Kernel {
		packedFillQ8(a, bQ, bScales, dst, M, K, N, nStart, nEnd, kBlock)
		return
	}
	fusedQ8Strip(a, bQ, bScales, dst, M, K, N, nStart, nEnd, kBlock)
}

// fusedQ8StripCols is how many weight ROWS are widened at a time by
// fusedQ8Strip. It MUST be a multiple of nBlockDefault: blockedFill tiles its
// column range in nBlock steps from nStart, so a strip that is a whole number
// of n-tiles produces exactly the tiles the full-width call would, and hence
// exactly its accumulation order.
//
// 256 rows x K=3072 x 4 B = 3 MB — past L1 but comfortably L2-resident, against
// the 9.4 MB a full [N,K] widen of that shape materializes. That is the whole
// point: the widened weights must live in cache, not in DRAM.
const fusedQ8StripCols = 8 * nBlockDefault

// fusedQ8Strip is the off-arm64 fusion (audit M-23): widen the weights one
// N-strip at a time and run the ORDINARY blockedFill over each strip, instead
// of widening the entire [N,K] matrix into a pooled buffer first.
//
// WHY THIS SHAPE. The encoder's amd64 Q8 path widened every int8 weight matrix
// to f32 on every matmul call — O(N*K) independent of sequence length, ~113M
// weights per forward per worker, up to 9.4 MB per matmul written and then read
// straight back. The f32 tile only ever needs to be as large as the block the
// GEMM is about to consume, so the fix is the buffer's LIFETIME, not the
// arithmetic.
//
// BIT-IDENTICAL, by reusing blockedFill rather than reimplementing it:
//
//   - the widened values are the same, from the same dequantRowInt8 the
//     reference's DequantizeRowsInt8Into uses;
//   - dst is sliced at the strip's first column and N is kept as the row
//     STRIDE, so blockedFill's dst[i*N+n] lands on exactly the element the
//     full-width call would write;
//   - the strip is a whole number of nBlock tiles, so the n-tiling, and with it
//     the m- and k-tiling inside each tile, is identical to the full-width
//     call's.
//
// It also needs no new kernel: the same blockRows3x4 / Dot8x4 / Dot4x4 path the
// f32 GEMM already runs on amd64 does the arithmetic.
func fusedQ8Strip(a []float32, bQ []int8, bScales []float32, dst []float32, M, K, N, nStart, nEnd, kBlock int) {
	if nEnd <= nStart {
		return
	}
	bufp := fusedQ8StripPool.Get().(*[]float32)
	defer fusedQ8StripPool.Put(bufp)
	need := fusedQ8StripCols * K
	if cap(*bufp) < need {
		*bufp = make([]float32, need)
	}
	w := (*bufp)[:need]

	for n0 := nStart; n0 < nEnd; n0 += fusedQ8StripCols {
		n1 := min(n0+fusedQ8StripCols, nEnd)
		rows := n1 - n0
		for r := range rows {
			dequantRowInt8(w[r*K:(r+1)*K], bQ[(n0+r)*K:(n0+r+1)*K], bScales[n0+r])
		}
		// dst[n0:] with N still the row stride puts column (n-n0) of this call
		// at absolute column n — see the bit-identity note above.
		blockedFill(a, w[:rows*K], dst[n0:], M, K, N, 0, rows, mBlockDefault, nBlockDefault, kBlock)
	}
}

var fusedQ8StripPool = sync.Pool{New: func() any { b := make([]float32, 0); return &b }}
