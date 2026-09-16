package encoder

import (
	"math"
	"sync"
)

// selfAttention runs one block's bidirectional multi-head self-attention and
// adds the output to the residual `h`, returning h resliced to mOut*D rows.
// Caller applies the post-attention LayerNorm on the returned slice in place
// (post-norm structure, plan §6.2).
//
// Shapes throughout this function:
//
//	h:        [L, D]                  hidden states (D = HiddenDim = heads * headDim)
//	Wqkv:     [3D, D]                 fused Q/K/V input projection (PyTorch layout)
//	OutProj:  [D, D]                  attention output projection
//	heads, headDim: D / heads, D / heads
//	rope:     precomputed cos/sin for positions 0..L-1
//
// mOut trims the OUTPUT rows to the caller's actual need (see attentionCore);
// forward.go/forward_tokens.go always pass L (every row is read downstream —
// CLS pooling happens after the full stack, not via a trimmed last layer), so
// this is a pure signature generalization for them, not a behavior change.
// gte.go's forward passes L−1-trimming on its final CLS-only layer.
//
// M8c: takes a *scratch arena so qkv / Q / K / V / ctx / qH/kH/vH /
// scores buffers are reused across the 12 layers per forward (and
// across forwards on the same worker via sync.Pool). Caller must have
// run s.ensureLayer(L, D, intermediate, heads, headDim) at the top of
// the forward.
//
// WqkvB / OutProjB are the optional projection biases: Nomic's original
// checkpoints (CodeRankEmbed, nomic-embed-text-v1.5) carry none, so they pass nil
// and the arithmetic is unchanged; nomic-embed-text-v2-moe sets qkv_proj_bias and
// an out_proj bias, and those rows are broadcast over the sequence.
func selfAttention(h []float32, Wqkv, WqkvB, OutProj, OutProjB []float32, heads, headDim, D, L, mOut int, rope *ropeTable, s *scratch) []float32 {
	if heads*headDim != D {
		panic("encoder: heads*headDim != D")
	}
	// 1) Project: QKV = h · Wqkvᵀ (+ bias) -> [L, 3D] into scratch, at full L —
	// see gte.go's forward for why the projection itself is never trimmed to
	// mOut even when the output rows are.
	qkv := s.qkv[:L*3*D]
	s.mm(h, Wqkv, qkv, L, D, 3*D)
	addRowBias(qkv, WqkvB, L, 3*D)

	// 2) Split QKV into Q, K, V — each [L, D]. Reuse scratch buffers.
	Q := s.Q[:L*D]
	K := s.K[:L*D]
	V := s.V[:L*D]
	if L > 0 && D > 0 {
		_ = qkv[L*3*D-1]
		_ = Q[L*D-1]
		_ = K[L*D-1]
		_ = V[L*D-1]
	}
	for i := range L {
		copy(Q[i*D:(i+1)*D], qkv[i*3*D:i*3*D+D])
		copy(K[i*D:(i+1)*D], qkv[i*3*D+D:i*3*D+2*D])
		copy(V[i*D:(i+1)*D], qkv[i*3*D+2*D:i*3*D+3*D])
	}

	// 3) RoPE on Q and K per head, full headDim, rotate_half.
	rope.apply(Q, heads)
	rope.apply(K, heads)

	// 4) Scaled dot-product attention, output projection, and residual —
	// shared with the BERT/GTE hand-rolled paths from this point on.
	return attentionCore(h, Q, K, V, OutProj, OutProjB, heads, headDim, D, L, mOut, s)
}

// attentionCore runs multi-head scaled-dot-product attention from already
// projected Q, K, V ([L, D] row-major each — only rows [0, mOut) of Q are
// ever read) through the output projection, then adds the result into
// h[:mOut*D] as the residual and returns that resliced h. Caller applies the
// post-attention LayerNorm on the returned slice.
//
// Shared by selfAttention (fused-qkv + RoPE path: CodeRankEmbed/nomic) and the
// BERT/GTE forwards, whose QKV *projections* differ (separate Wq/Wk/Wv vs a
// packed Wqkv, no RoPE vs RoPE) but whose attention math from here on is
// identical — this is that identical core, extracted once instead of hand-
// rolled per model.
//
// mOut trims the OUTPUT rows only: K and V stay at full L regardless, because
// attention must read every position no matter how many output rows are
// needed. mOut < L exists for exactly one case, a CLS-only forward's final
// layer, where only row 0 is ever read downstream — see BERT.clsHiddenState
// and GTE.clsHiddenState for the bit-identical contract this must preserve.
func attentionCore(h []float32, Q, K, V, OutProj, OutProjB []float32, heads, headDim, D, L, mOut int, s *scratch) []float32 {
	// Scaled dot-product attention per head. Scratch holds qH/kH (per-head Q/K
	// extracts), vHT (V transposed to [headDim, L]) and scores ([mOut, L]). Both
	// matmuls go through the SIMD A·Bᵀ kernel: QKᵀ = qH·kHᵀ, then the context
	// scores·V = scores·(vHT)ᵀ — the latter was a scalar triple-loop and the
	// single hottest line in Encode before this (≈⅓ of total).
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	ctx := s.ctx[:mOut*D] // every (i, head) column is written exactly once below
	qH := s.qH[:mOut*headDim]
	kH := s.kH[:L*headDim]
	vHT := s.vH[:headDim*L]
	ctxHead := s.ctxHead[:mOut*headDim]
	scores := s.scores[:mOut*L]

	if w := attnHeadWorkers(s, heads, mOut, headDim, L); w > 1 {
		// Head-parallel (audit M-05). Each head is an independent computation
		// over its own slice of Q/K/V writing its own columns of ctx, so this is
		// bit-identical to the serial loop — no reduction crosses a head.
		s.ensureWorkerScratch(w, mOut, headDim, L)
		var wg sync.WaitGroup
		for wi := range w {
			wg.Add(1)
			go func(wi int) {
				defer wg.Done()
				hs := &s.workerScratch[wi]
				for headIdx := wi; headIdx < heads; headIdx += w {
					attendOneHead(ctx, Q, K, V, headIdx, headDim, D, L, mOut, scale,
						hs.qH, hs.kH, hs.vH, hs.ctxHead, hs.scores, s.mm,
						softmaxRowsScaledSerial)
				}
			}(wi)
		}
		wg.Wait()
	} else {
		for headIdx := range heads {
			attendOneHead(ctx, Q, K, V, headIdx, headDim, D, L, mOut, scale,
				qH, kH, vHT, ctxHead, scores, s.mm, softmaxRowsScaled)
		}
	}

	// Output projection into scratch (+ bias).
	out := s.out[:mOut*D]
	s.mm(ctx, OutProj, out, mOut, D, D)
	addRowBias(out, OutProjB, mOut, D)

	// Residual: h[:mOut*D] += out (in place), resliced.
	h = h[:mOut*D]
	nh := len(h)
	if nh > 0 {
		_ = h[nh-1]
		_ = out[nh-1]
		i := 0
		for ; i+3 < nh; i += 4 {
			h[i+0] += out[i+0]
			h[i+1] += out[i+1]
			h[i+2] += out[i+2]
			h[i+3] += out[i+3]
		}
		for ; i < nh; i++ {
			h[i] += out[i]
		}
	}
	return h
}

// attendOneHead runs one attention head into its columns of ctx, using
// caller-supplied per-head buffers so the head loop can run serially on one
// scratch or in parallel on several. Extracted rather than duplicated: the two
// paths must not be able to drift apart.
//
// softmax is a parameter for one reason — nesting. The serial path passes
// softmaxRowsScaled, which fans its rows across cores. Inside a head worker
// that would be a second fan-out under the first, so the parallel path passes
// the serial form (softmaxRowsScaledSerial) and the parallelism stays on the
// head axis where the work is coarser.
func attendOneHead(
	ctx, Q, K, V []float32,
	headIdx, headDim, D, L, mOut int, scale float32,
	qH, kH, vHT, ctxHead, scores []float32,
	mm func(a, b, dst []float32, M, K, N int),
	softmax func(scores []float32, scale float32, rows, cols int),
) {
	headOff := headIdx * headDim
	for i := range L {
		src := i*D + headOff
		if i < mOut {
			copy(qH[i*headDim:(i+1)*headDim], Q[src:src+headDim])
		}
		copy(kH[i*headDim:(i+1)*headDim], K[src:src+headDim])
	}
	// V transposed: vHT[d, i] = V[i, head, d], folded into the extract so
	// scores·V can use the A·Bᵀ matmul (which needs Vᵀ as its b operand).
	if L%4 == 0 && headDim%4 == 0 && L > 0 && headDim > 0 {
		_ = V[(L-1)*D+headOff+headDim-1]
		_ = vHT[(headDim-1)*L+L-1]
		for i0 := 0; i0 < L; i0 += 4 {
			for d0 := 0; d0 < headDim; d0 += 4 {
				s0 := (i0+0)*D + headOff + d0
				s1 := (i0+1)*D + headOff + d0
				s2 := (i0+2)*D + headOff + d0
				s3 := (i0+3)*D + headOff + d0

				v00, v01, v02, v03 := V[s0], V[s0+1], V[s0+2], V[s0+3]
				v10, v11, v12, v13 := V[s1], V[s1+1], V[s1+2], V[s1+3]
				v20, v21, v22, v23 := V[s2], V[s2+1], V[s2+2], V[s2+3]
				v30, v31, v32, v33 := V[s3], V[s3+1], V[s3+2], V[s3+3]

				t0 := (d0+0)*L + i0
				t1 := (d0+1)*L + i0
				t2 := (d0+2)*L + i0
				t3 := (d0+3)*L + i0

				vHT[t0], vHT[t0+1], vHT[t0+2], vHT[t0+3] = v00, v10, v20, v30
				vHT[t1], vHT[t1+1], vHT[t1+2], vHT[t1+3] = v01, v11, v21, v31
				vHT[t2], vHT[t2+1], vHT[t2+2], vHT[t2+3] = v02, v12, v22, v32
				vHT[t3], vHT[t3+1], vHT[t3+2], vHT[t3+3] = v03, v13, v23, v33
			}
		}
	} else {
		for i := range L {
			src := i*D + headOff
			if headDim > 0 {
				_ = V[src+headDim-1]
				_ = vHT[(headDim-1)*L+i]
				d := 0
				for ; d+3 < headDim; d += 4 {
					vHT[(d+0)*L+i] = V[src+d+0]
					vHT[(d+1)*L+i] = V[src+d+1]
					vHT[(d+2)*L+i] = V[src+d+2]
					vHT[(d+3)*L+i] = V[src+d+3]
				}
				for ; d < headDim; d++ {
					vHT[d*L+i] = V[src+d]
				}
			}
		}
	}
	mm(qH, kH, scores, mOut, headDim, L)
	softmax(scores, scale, mOut, L)
	// ctxHead[mOut, headDim] = scores[mOut, L] · V[L, headDim], as scores · (vHT)ᵀ.
	mm(scores, vHT, ctxHead, mOut, L, headDim)
	// Scatter this head's context into the interleaved ctx[mOut, D].
	for i := range mOut {
		dst := i*D + headOff
		copy(ctx[dst:dst+headDim], ctxHead[i*headDim:(i+1)*headDim])
	}
}

// softmaxRowsScaledSerial is softmaxRowsScaled without the row fan-out, for a
// caller that is already running on a worker goroutine.
func softmaxRowsScaledSerial(scores []float32, scale float32, rows, cols int) {
	for i := range rows {
		softmaxRowScaled(scores[i*cols:(i+1)*cols], scale)
	}
}

// attnHeadWorkers decides the head-loop fan-out (audit M-05).
//
// The per-head matmuls sit BELOW parallelThreshold for every sequence this
// library actually encodes — with headDim=64 the QK^T shape clears 32M only at
// L >= 708, and BERT/CodeRankEmbed/rerank cap at maxSeq=512 — so before this the
// head loop and both of its matmuls ran on one core while the linears around
// them used every core. Roughly a third of a lone forward.
//
// The conditions are deliberately narrow:
//
//   - a backend (GPU) is not fanned out: s.mm would be called concurrently and
//     a device queue is not promised to be safe for that;
//   - if the per-head matmul would ITSELF parallelize, the cores are already
//     busy and a head fan-out on top would oversubscribe — so head-parallelism
//     is taken exactly when the inner matmuls are serial, which is the case the
//     finding is about;
//   - inflightForwards > 1 means sibling forwards already occupy the machine,
//     the same guard parallelRows and wantParallelMatmul use.
func attnHeadWorkers(s *scratch, heads, mOut, headDim, L int) int {
	if heads < 2 || s.be != nil || inflightForwards.Load() > 1 {
		return 1
	}
	if wantParallelMatmul(mOut, headDim, L) {
		return 1
	}
	// Below this the head loop is a few hundred microseconds and the spawn is
	// pure overhead. mOut*L*headDim is one head's QK^T MAC count.
	if int64(mOut)*int64(L)*int64(headDim) < attnHeadParallelThreshold {
		return 1
	}
	return min(numCPU, heads)
}

// attnHeadParallelThreshold is the per-head QK^T MAC count at/above which
// fanning the head loop pays for the goroutine spawn.
const attnHeadParallelThreshold = 1 << 20
