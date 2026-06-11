package linalg

import "math"

// Per-row symmetric int8 weight quantization.
// Each output row (channel) of a [rows, cols] weight matrix gets its own f32
// scale; the symmetric [-127,127] range keeps zero at zero with no zero-point
// bookkeeping. This is the standard per-channel scheme bitsandbytes/GPTQ use.
// Halves memory vs the bf16 checkpoint (and quarters it vs the f32 we widen to
// on load), at a per-row reconstruction error the model tolerates.
//
// The decoder uses weight-only int8: weights are int8, activations stay f32,
// and the int8→f32 widen happens inside the matmul (MatmulBTQ8) — the win is
// the 2–4× smaller weight footprint and the memory bandwidth it saves.

// QuantizeRowsInt8 quantizes a [rows, cols] f32 matrix (row-major) to int8
// weights + per-row f32 scales. Reconstruct: W[i,j] ≈ float32(q[i*cols+j]) *
// scales[i]. An all-zero row gets scale 1 (its codes are all zero anyway).
func QuantizeRowsInt8(w []float32, rows, cols int) (q []int8, scales []float32) {
	if rows*cols != len(w) {
		panic("linalg: QuantizeRowsInt8 shape mismatch")
	}
	q = make([]int8, rows*cols)
	scales = make([]float32, rows)
	for i := range rows {
		scales[i] = QuantizeRowInt8(w[i*cols:(i+1)*cols], q[i*cols:(i+1)*cols])
	}
	return q, scales
}

// QuantizeRowInt8 quantizes one f32 row into q (len cols) and returns its scale —
// the single-row core of QuantizeRowsInt8 (bit-identical), exposed so a loader can
// quantize each row as it is dequantized, without buffering the whole f32 matrix.
// An all-zero row gets scale 1.
func QuantizeRowInt8(row []float32, q []int8) (scale float32) {
	var maxAbs float32
	for _, v := range row {
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs == 0 {
		for j := range q {
			q[j] = 0
		}
		return 1
	}
	s := maxAbs / 127.0
	inv := 1.0 / s
	for j, v := range row {
		x := math.Round(float64(v * inv))
		if x > 127 {
			x = 127
		} else if x < -127 {
			x = -127
		}
		q[j] = int8(x)
	}
	return s
}

// DequantizeRowInt8 reconstructs one row into dst: dst[j] = float32(q[j])*scale.
// Used for the tied embedding lookup when the table is stored int8.
func DequantizeRowInt8(q []int8, scale float32, dst []float32) {
	checkDequantInt8(q, dst)
	for j, c := range q {
		dst[j] = float32(c) * scale
	}
}

// MatmulBTQ8 computes dst[M,N] = a[M,K] · bᵀ where b is the [N,K] matrix stored
// as int8 rows bQ + per-row f32 scales bScales (b[j,k] ≈ float32(bQ[j,k]) *
// bScales[j]). Each output row is widened int8→f32 into a reused scratch buffer,
// then the SIMD dotF32 kernel (AVX2/NEON — the primitive MatmulBT uses) runs over
// the whole row and the per-row scale is applied at write-back. Only the cheap
// int8→f32 widen stays scalar; the multiply-accumulate is vectorized. The scratch
// is one row wide and allocated once per worker. Parallelized over the N columns.
func MatmulBTQ8(a []float32, bQ []int8, bScales []float32, dst []float32, M, K, N int) {
	parallelCols(M*N*K, N, func(j0, j1 int) {
		deq := make([]float32, K) // per-worker scratch: one widened b-row
		for i := range M {
			arow := a[i*K : i*K+K]
			drow := dst[i*N : i*N+N]
			for j := j0; j < j1; j++ {
				bq := bQ[j*K : j*K+K]
				for k := range K {
					deq[k] = float32(bq[k])
				}
				drow[j] = dotF32(arow, deq) * bScales[j]
			}
		}
	})
}

// dotI8Scalar returns Σ a[i]*b[i] as an int32 over two int8 vectors (the products
// fit: 127*127*K stays well within int32 for transformer K). It is the portable
// reference and the tail/fallback for the SIMD dotI8 dispatcher (see the per-arch
// quant_i8_*.go).
func dotI8Scalar(a, b []int8) int32 {
	var s int32
	for k := range a {
		s += int32(a[k]) * int32(b[k])
	}
	return s
}

// quantizeRowInt8 dynamically quantizes one f32 activation row to int8 with a
// single symmetric scale (maxabs/127). Returns the codes (into dst) and the
// scale; an all-zero row gets scale 0 and zero codes.
func quantizeRowInt8(a []float32, dst []int8) (scale float32) {
	var maxAbs float32
	for _, v := range a {
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs == 0 {
		for k := range dst {
			dst[k] = 0
		}
		return 0
	}
	scale = maxAbs / 127
	inv := 1.0 / scale
	for k, v := range a {
		x := math.Round(float64(v * inv))
		if x > 127 {
			x = 127
		} else if x < -127 {
			x = -127
		}
		dst[k] = int8(x)
	}
	return scale
}

// MatmulBTW8A8 computes dst[M,N] = a[M,K] · bᵀ as full int8×int8→int32 (W8A8):
// the f32 activation row is quantized to int8 on the fly (dynamic per-row scale),
// the integer dot accumulates in int32, and the result is rescaled by the
// activation scale × the per-row weight scale. Unlike MatmulBTQ8 (weight-only
// int8, f32 activations) this also quantizes the activations, so it is lossier —
// the tradeoff for an integer kernel. Parallelized over the N columns.
func MatmulBTW8A8(a []float32, bQ []int8, bScales []float32, dst []float32, M, K, N int) {
	var ws Workspace
	MatmulBTW8A8Into(&ws, a, bQ, bScales, dst, M, K, N)
}

// MatmulBTW8A8Into is MatmulBTW8A8 with caller-supplied scratch, so a steady-
// state decode loop allocates nothing. It also quantizes each activation row
// ONCE (into ws) rather than once per worker — the old code re-quantized the
// same row in every parallel chunk and allocated a scratch buffer per worker,
// which was the bulk of decode alloc_space. Output is bit-identical to
// MatmulBTW8A8 (same quantizeRowInt8 / dotI8 / rescale, just hoisted).
func MatmulBTW8A8Into(ws *Workspace, a []float32, bQ []int8, bScales []float32, dst []float32, M, K, N int) {
	aq := ws.int8Buf(M * K)
	aScales := ws.f32Buf(M)
	for i := range M {
		aScales[i] = quantizeRowInt8(a[i*K:i*K+K], aq[i*K:i*K+K])
	}
	// Serial fast-path calls the named span directly (no closure → no heap
	// escape → zero alloc, the steady-state decode case). Only the parallel
	// branch pays a closure allocation, where it's noise next to the goroutines.
	if M*N*K < ws.thr() || N < 2 {
		w8a8Span(aq, aScales, bQ, bScales, dst, M, K, N, 0, N)
		return
	}
	ws.parallel(N, func(j0, j1 int) {
		w8a8Span(aq, aScales, bQ, bScales, dst, M, K, N, j0, j1)
	})
}

// w8a8Span computes output columns [j0,j1) for every row: dst[i,j] =
// dotI8(aq[i], bQ[j]) · aScales[i] · bScales[j]. A named function (not a
// closure) so the serial caller invokes it without a heap allocation.
//
// Column-outer: each weight row bj is loaded once and reused across all M
// activation rows (served from L1/L2 for rows 2..M instead of re-streamed from
// RAM). Decode is bandwidth-bound and weights dominate, so at M>1 (speculative
// verify's M=K, prefill, the encoder) this streams the weight matrix once
// rather than M times. M=1 is bit-identical to the old row-outer order (one i
// iteration), so the single-token decode hot path is unchanged. The output of
// each dst[i,j] is the same float32 expression regardless of loop order —
// bit-identical for any M.
func w8a8Span(aq []int8, aScales []float32, bQ []int8, bScales, dst []float32, M, K, N, j0, j1 int) {
	for j := j0; j < j1; j++ {
		bj := bQ[j*K : j*K+K]
		bs := bScales[j]
		for i := 0; i < M; i++ {
			if aScales[i] == 0 {
				dst[i*N+j] = 0
				continue
			}
			dst[i*N+j] = float32(dotI8(aq[i*K:i*K+K], bj)) * aScales[i] * bs
		}
	}
}

// W8A8Op is one weight matrix in a batched W8A8 matmul: BQ is the [N,K] int8
// weights (row-major, used in place — NOT copied), Scales the [N] per-row
// weight scales, Dst the [M,N] output. N is the column count.
type W8A8Op struct {
	BQ     []int8
	Scales []float32
	Dst    []float32
	N      int
}

// MatmulBTW8A8Batch runs several W8A8 matmuls that share the SAME activation
// a[M,K] — fused q/k/v or gate/up — in ONE parallel region: the activation is
// quantized once and the goroutine fork/join is amortized across every op's
// columns (the concatenated [0, ΣN) column space is split across workers),
// instead of one quantize + one fork/join per matmul. The weights stay in
// place, so a consumer that aliases int8 weights zero-copy (goinfer's prequant
// path) gets the dispatch reduction with NO concat copy.
//
// Numerically identical to calling MatmulBTW8A8Into once per op.
func MatmulBTW8A8Batch(ws *Workspace, a []float32, M, K int, ops []W8A8Op) {
	if len(ops) == 0 {
		return
	}
	aq := ws.int8Buf(M * K)
	aScales := ws.f32Buf(M)
	for i := range M {
		aScales[i] = quantizeRowInt8(a[i*K:i*K+K], aq[i*K:i*K+K])
	}
	totalN := 0
	for _, op := range ops {
		totalN += op.N
	}
	if M*totalN*K < ws.thr() || totalN < 2 {
		w8a8BatchSpan(aq, aScales, ops, M, K, 0, totalN)
		return
	}
	ws.parallel(totalN, func(g0, g1 int) {
		w8a8BatchSpan(aq, aScales, ops, M, K, g0, g1)
	})
}

// w8a8BatchSpan computes the [g0,g1) slice of the ops' concatenated column
// space, mapping each global column back to its op and local column. Named
// (not a closure) so the serial caller pays no allocation.
func w8a8BatchSpan(aq []int8, aScales []float32, ops []W8A8Op, M, K, g0, g1 int) {
	base := 0
	for _, op := range ops {
		lo, hi := max(g0, base), min(g1, base+op.N) // this op's slice of [g0,g1)
		if lo < hi {
			// Column-outer (see w8a8Span): weight row reused across M rows.
			for j := lo; j < hi; j++ {
				jj := j - base
				bj := op.BQ[jj*K : jj*K+K]
				bs := op.Scales[jj]
				for i := 0; i < M; i++ {
					if aScales[i] == 0 {
						op.Dst[i*op.N+jj] = 0
						continue
					}
					op.Dst[i*op.N+jj] = float32(dotI8(aq[i*K:i*K+K], bj)) * aScales[i] * bs
				}
			}
		}
		base += op.N
	}
}

// Group-wise symmetric int4 weight quantization. Per-ROW int8 is too coarse at
// 4 bits, so each row is split into groups
// of `group` consecutive input features (along K), and each group gets its own
// f32 scale: W[i, g*group+e] ≈ (nibble-8) * scale[i,g], with the nibble a 4-bit
// code in [1,15] (8 = zero, symmetric range [-7,7]). Two nibbles pack per byte
// (even k = low nibble, odd k = high). At group 32 this is ~0.625 byte/element
// (4-bit code + the per-group scale amortized), ≈ 6.4× smaller than f32 and
// ~1.6× smaller than per-row int8 — the footprint that fits a 7B-class model in
// laptop RAM. The matmul (MatmulBTQ4) dequantizes per group inside the inner
// loop; activations stay f32.

// groupsFor returns the number of groups a K-wide row splits into (the final
// group is ragged when group does not divide K) and the packed bytes per row.
func groupsFor(cols, group int) (nGroups, bytesPerRow int) {
	return (cols + group - 1) / group, (cols + 1) / 2
}

// QuantizeGroupsInt4 quantizes a [rows, cols] f32 matrix (row-major) to packed
// 4-bit codes + per-group f32 scales. Reconstruct: W[i, k] ≈ (nibble(i,k)-8) *
// scales[i*nGroups + k/group]. An all-zero group gets scale 1 (codes all 8).
func QuantizeGroupsInt4(w []float32, rows, cols, group int) (packed []byte, scales []float32) {
	if rows*cols != len(w) {
		panic("linalg: QuantizeGroupsInt4 shape mismatch")
	}
	nGroups, bpr := groupsFor(cols, group)
	packed = make([]byte, rows*bpr)
	scales = make([]float32, rows*nGroups)
	for i := range rows {
		QuantizeGroupInt4Row(w[i*cols:(i+1)*cols], cols, group, packed[i*bpr:(i+1)*bpr], scales[i*nGroups:(i+1)*nGroups])
	}
	return packed, scales
}

// QuantizeGroupInt4Row quantizes one f32 row into packed (len (cols+1)/2) +
// per-group scales (len ⌈cols/group⌉) — the single-row core of QuantizeGroupsInt4
// (bit-identical), exposed so a loader can quantize each row as it is dequantized,
// without buffering the whole f32 matrix. packed is assumed zeroed on entry (a
// fresh per-row slice).
func QuantizeGroupInt4Row(row []float32, cols, group int, packed []byte, scales []float32) {
	nGroups := (cols + group - 1) / group
	for g := range nGroups {
		ks := g * group
		ke := min(ks+group, cols)
		var maxAbs float32
		for k := ks; k < ke; k++ {
			if v := row[k]; v > maxAbs {
				maxAbs = v
			} else if -v > maxAbs {
				maxAbs = -v
			}
		}
		s := float32(1)
		if maxAbs > 0 {
			s = maxAbs / 7
		}
		scales[g] = s
		inv := 1.0 / s
		for k := ks; k < ke; k++ {
			q := int(math.Round(float64(row[k] * inv)))
			if q > 7 {
				q = 7
			} else if q < -7 {
				q = -7
			}
			nib := byte(q + 8) // [1,15]; 8 = zero
			bi := k / 2
			if k&1 == 0 {
				packed[bi] = (packed[bi] &^ 0x0F) | (nib & 0x0F)
			} else {
				packed[bi] = (packed[bi] &^ 0xF0) | (nib << 4)
			}
		}
	}
}

// DequantizeRowInt4 reconstructs one row into dst[:cols] from its packed nibbles
// and per-group scales (both already sliced to the row). Used for the tied
// embedding lookup when the table is stored int4.
func DequantizeRowInt4(packed []byte, scales []float32, group, cols int, dst []float32) {
	checkDequantInt4(packed, scales, group, cols, dst)
	for k := range cols {
		b := packed[k/2]
		var nib byte
		if k&1 == 0 {
			nib = b & 0x0F
		} else {
			nib = b >> 4
		}
		dst[k] = float32(int(nib)-8) * scales[k/group]
	}
}

// MatmulBTW4A8 computes dst[M,N] = a[M,K] · bᵀ as int4 WEIGHTS × int8
// ACTIVATIONS — the int4 analogue of MatmulBTW8A8, and the fast M=1 (decode)
// path that MatmulBTQ4 can't be. b is group-wise int4 (w4 nibbles + wScales per
// group, QuantizeGroupsInt4 layout); the f32 activations are dynamically
// quantized to int8 per row (per-row scale, like MatmulBTW8A8).
//
// Each output is one fused dotW4A8 call that streams the whole weight row in the
// integer domain — unpack each int4 group to int8 (nibble−8), int8×int8 SDOT
// into int32, fold in the group's weight scale — with NO per-weight f32 dequant
// and NO per-group Go↔asm transition (the arm64 kernel loops groups internally).
// That is what keeps it fast at M=1: MatmulBTQ4 spends ~72% of decode in the f32
// dequant, which the column-outer M-reuse can only amortize at M>1; W4A8 removes
// the dequant outright. Lossier than MatmulBTQ4 (activations are int8, not f32)
// — the W8A8 tradeoff — so it's the explicit-opt-in kernel for RAM-constrained
// int4 CPU decode, not a drop-in for the f32-activation path.
func MatmulBTW4A8(a []float32, w4 []byte, wScales []float32, dst []float32, M, K, N, group int) {
	checkGroupMatmul("MatmulBTW4A8", len(a), w4, wScales, len(dst), M, K, N, group)
	nGroups, bpr := groupsFor(K, group)
	aq := make([]int8, M*K)
	aScales := make([]float32, M)
	for i := range M {
		aScales[i] = quantizeRowInt8(a[i*K:i*K+K], aq[i*K:i*K+K])
	}
	parallelCols(M*N*K, N, func(j0, j1 int) {
		sums := make([]int32, nGroups) // per-worker scratch: asm group dots
		for j := j0; j < j1; j++ {
			prow := w4[j*bpr : j*bpr+bpr]
			srow := wScales[j*nGroups : j*nGroups+nGroups]
			for i := range M {
				if aScales[i] == 0 {
					dst[i*N+j] = 0
					continue
				}
				dst[i*N+j] = dotW4A8(aq[i*K:i*K+K], prow, srow, group, K, sums) * aScales[i]
			}
		}
	})
}

// unpackInt4Row unpacks n group-int4 nibbles into dst[:n] as centered int8
// (nibble−8), two nibbles per byte (even k = low, odd k = high — the
// QuantizeGroupInt4Row layout), branchless in the hot pair loop.
func unpackInt4Row(packed []byte, n int, dst []int8) {
	full := n &^ 1
	di := 0
	for bi := 0; di < full; bi++ {
		b := packed[bi]
		dst[di] = int8(int(b&0x0F) - 8)
		dst[di+1] = int8(int(b>>4) - 8)
		di += 2
	}
	if di < n { // trailing odd nibble (low)
		dst[di] = int8(int(packed[di>>1]&0x0F) - 8)
	}
}

// MatmulBTQ4 computes dst[M,N] = a[M,K] · bᵀ where b is the [N,K] matrix stored
// as group-wise int4 (bPacked nibbles + bScales per group; see
// QuantizeGroupsInt4). Each weight row is dequantized ONCE into a full K-wide
// f32 scratch (4-bit code − 8, times its group scale — the same per-element
// reconstruction as DequantizeRowInt4), then the SIMD dotF32 kernel (AVX2/NEON,
// the primitive MatmulBT uses) runs over the WHOLE row. So the dequant is the
// only scalar work (O(K), like the int8-widen in MatmulBTQ8) and the
// multiply-accumulate is one vectorized pass — not K/group tiny 32-wide dots,
// which were so per-call-overhead-bound they ran slower than scalar.
//
// Column-outer: each weight row's dequant is reused across the M activation
// rows (prefill / speculative verify), streaming the weight once. Activations
// stay f32. The scratch is K wide, allocated once per worker.
//
// Output matches DequantizeRowInt4-then-MatmulBT bit-for-bit (the same order the
// Q4 parity test references); the prior per-group-dot kernel only matched within
// tolerance, so this is also slightly MORE faithful, not less.
func MatmulBTQ4(a []float32, bPacked []byte, bScales []float32, dst []float32, M, K, N, group int) {
	checkGroupMatmul("MatmulBTQ4", len(a), bPacked, bScales, len(dst), M, K, N, group)
	nGroups, bpr := groupsFor(K, group)
	parallelCols(M*N*K, N, func(j0, j1 int) {
		deq := make([]float32, K) // per-worker scratch: one full dequantized weight row
		for j := j0; j < j1; j++ {
			DequantizeRowInt4(bPacked[j*bpr:j*bpr+bpr], bScales[j*nGroups:j*nGroups+nGroups], group, K, deq)
			for i := 0; i < M; i++ {
				dst[i*N+j] = dotF32(a[i*K:i*K+K], deq)
			}
		}
	})
}
