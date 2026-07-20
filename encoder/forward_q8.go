package encoder

import (
	"math"

	"github.com/townsendmerino/aikit/linalg"
)

// (*WeightsQ8).forward runs the int8 forward pass on a single sequence. It mirrors
// Weights.forward exactly — same pooled scratch arena, same attention math — except
// the five big linear layers (Wqkv, OutProj, fc11, fc12, fc2) route through
// matmulBTQ8Into (int8 weights) instead of matmulBTInto. LN, softmax, RoPE, residual
// adds, and the CLS pool stay f32.
//
// Two things made LoadQ8 ~5× slower than Load on arm64; both are fixed here.
// (1) Allocation: the pre-pool q8 path allocated every per-layer matmul output and
// per-head attention temporary fresh (~4.4 GiB for a 50-doc rerank); pooling the
// scratch arena (mirrored from f32) brings it in line. (2) The kernel: matmulBTQ8Into
// dequantizes the int8 weights to f32 ONCE per matmul and runs the SIMD f32 matmul,
// replacing a scalar inline-widen kernel that ran ~26× slower than f32 SIMD — the
// dominant cost the allocation alone didn\'t explain. Net: q8 reaches f32 latency
// parity at ¼ the weight storage and far less runtime memory, with weight-only
// numerics unchanged (cosine 0.997). Full W8A8 (SDOT, even faster) was rejected — it
// quantizes activations and fell below the 0.97 reranker bar. BenchmarkRerankN50_f32_vs_q8
// guards both fixes.
func (w *WeightsQ8) forward(ids []int32) []float32 {
	enterForward()
	defer leaveForward()
	L := len(ids)
	D := w.Cfg.HiddenDim
	if L == 0 {
		return make([]float32, D)
	}
	heads := w.Cfg.NumHeads
	headDim := w.Cfg.HeadDim()
	intermediate := w.Cfg.IntermediateDim
	eps := w.Cfg.LayerNormEpsilon

	s := getScratch()
	defer putScratch(s)
	s.ensureLayer(L, D, intermediate, heads, headDim, L)
	s.ensureDeqW(D, intermediate)

	h := make([]float32, L*D)
	tte0 := w.TokenTypeEmb[:D]
	for i, id := range ids {
		src := w.WordEmb[clampTokenID(id, w.Cfg.VocabSize)*D:][:D]
		dst := h[i*D : (i+1)*D]
		for j := range D {
			dst[j] = src[j] + tte0[j]
		}
	}
	layerNorm(h, w.EmbLN_W, w.EmbLN_B, L, D, eps)
	rope := newRopeTable(L, headDim, w.Cfg.RoPEBase)
	for i := 0; i < w.Cfg.NumLayers; i++ {
		l := &w.Layers[i]
		selfAttentionQ8(h, &l.Wqkv, &l.OutProj,
			heads, headDim, D, L, rope, s)
		layerNorm(h, l.Norm1W, l.Norm1B, L, D, eps)
		swigluMLPQ8(h, &l.Fc11, &l.Fc12, &l.Fc2, D, intermediate, L, s)
		layerNorm(h, l.Norm2W, l.Norm2B, L, D, eps)
	}
	// Route through poolOne(Cfg.pooling) like the f32 forward — the Q8 path used
	// to hardcode CLS, so a mean-pooling checkpoint would silently return CLS.
	return poolOne(h[:L*D], L, D, w.Cfg.pooling)
}

// (*WeightsQ8).forwardBatch is the int8 + batched (M7) combination. Mirrors
// Weights.forwardBatch with the linear projections in int8.
func (w *WeightsQ8) forwardBatch(idsList [][]int32) [][]float32 {
	enterForward() // see Weights.forwardBatch for the in-flight-gate rationale
	defer leaveForward()
	B := len(idsList)
	D := w.Cfg.HiddenDim
	if B == 0 {
		return nil
	}
	if B == 1 {
		return [][]float32{w.forward(idsList[0])}
	}
	Lmax := 0
	realLen := make([]int, B)
	for b, ids := range idsList {
		realLen[b] = len(ids)
		if len(ids) > Lmax {
			Lmax = len(ids)
		}
	}
	if Lmax == 0 {
		out := make([][]float32, B)
		for i := range out {
			out[i] = make([]float32, D)
		}
		return out
	}
	heads := w.Cfg.NumHeads
	headDim := w.Cfg.HeadDim()
	intermediate := w.Cfg.IntermediateDim
	eps := w.Cfg.LayerNormEpsilon
	BL := B * Lmax

	s := getScratch()
	defer putScratch(s)
	s.ensureLayer(BL, D, intermediate, heads, headDim, Lmax)
	s.ensureDeqW(D, intermediate)

	h := make([]float32, BL*D)
	tte0 := w.TokenTypeEmb[:D]
	for b, ids := range idsList {
		base := b * Lmax * D
		for i, id := range ids {
			src := w.WordEmb[clampTokenID(id, w.Cfg.VocabSize)*D:][:D]
			dst := h[base+i*D : base+(i+1)*D]
			for j := range D {
				dst[j] = src[j] + tte0[j]
			}
		}
	}
	layerNorm(h, w.EmbLN_W, w.EmbLN_B, BL, D, eps)
	rope := newRopeTable(Lmax, headDim, w.Cfg.RoPEBase)
	for li := 0; li < w.Cfg.NumLayers; li++ {
		l := &w.Layers[li]
		selfAttentionQ8Batched(h, &l.Wqkv, &l.OutProj,
			heads, headDim, D, B, Lmax, realLen, rope, s)
		layerNorm(h, l.Norm1W, l.Norm1B, BL, D, eps)
		swigluMLPQ8(h, &l.Fc11, &l.Fc12, &l.Fc2, D, intermediate, BL, s)
		layerNorm(h, l.Norm2W, l.Norm2B, BL, D, eps)
	}
	out := make([][]float32, B)
	for b := range B {
		seq := h[b*Lmax*D : b*Lmax*D+realLen[b]*D]
		out[b] = poolOne(seq, realLen[b], D, w.Cfg.pooling)
	}
	return out
}

// selfAttentionQ8 mirrors selfAttention (pooled scratch, vectorized QKᵀ + scores·V)
// with Wqkv/OutProj in int8 via matmulBTQ8Into. Attention itself stays f32.
func selfAttentionQ8(h []float32, wqkv, outProj *linalg.WeightMat,
	heads, headDim, D, L int, rope *ropeTable, s *scratch) {
	qkv := s.qkv[:L*3*D]
	WqkvQ, WqkvScales, _, _ := wqkv.Int8()
	matmulBTQ8Into(qkv, h, WqkvQ, WqkvScales, L, D, 3*D, s.deqW)
	Q := s.Q[:L*D]
	K := s.K[:L*D]
	V := s.V[:L*D]
	for i := range L {
		copy(Q[i*D:(i+1)*D], qkv[i*3*D:i*3*D+D])
		copy(K[i*D:(i+1)*D], qkv[i*3*D+D:i*3*D+2*D])
		copy(V[i*D:(i+1)*D], qkv[i*3*D+2*D:i*3*D+3*D])
	}
	rope.apply(Q, heads)
	rope.apply(K, heads)

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	ctx := s.ctx[:L*D]
	qH := s.qH[:L*headDim]
	kH := s.kH[:L*headDim]
	vHT := s.vH[:headDim*L]
	ctxHead := s.ctxHead[:L*headDim]
	scores := s.scores[:L*L]
	for headIdx := range heads {
		for i := range L {
			src := i*D + headIdx*headDim
			copy(qH[i*headDim:(i+1)*headDim], Q[src:src+headDim])
			copy(kH[i*headDim:(i+1)*headDim], K[src:src+headDim])
			for d := range headDim {
				vHT[d*L+i] = V[src+d]
			}
		}
		matmulBTInto(qH, kH, scores, L, headDim, L)
		for i := range scores {
			scores[i] *= scale
		}
		for i := range L {
			softmaxRow(scores[i*L : (i+1)*L])
		}
		matmulBTInto(scores, vHT, ctxHead, L, L, headDim)
		for i := range L {
			dst := i*D + headIdx*headDim
			copy(ctx[dst:dst+headDim], ctxHead[i*headDim:(i+1)*headDim])
		}
	}
	out := s.out[:L*D]
	OutProjQ, OutProjScales, _, _ := outProj.Int8()
	matmulBTQ8Into(out, ctx, OutProjQ, OutProjScales, L, D, D, s.deqW)
	for i := range h {
		h[i] += out[i]
	}
}

// selfAttentionQ8Batched mirrors selfAttentionBatched (pooled scratch, hoisted
// per-(b,head) buffers, vectorized scores·V) with the linears in int8.
func selfAttentionQ8Batched(h []float32, wqkv, outProj *linalg.WeightMat,
	heads, headDim, D, B, Lmax int, realLen []int, rope *ropeTable, s *scratch) {
	BL := B * Lmax
	qkv := s.qkv[:BL*3*D]
	WqkvQ, WqkvScales, _, _ := wqkv.Int8()
	matmulBTQ8Into(qkv, h, WqkvQ, WqkvScales, BL, D, 3*D, s.deqW)
	Q := s.Q[:BL*D]
	K := s.K[:BL*D]
	V := s.V[:BL*D]
	for i := range BL {
		copy(Q[i*D:(i+1)*D], qkv[i*3*D:i*3*D+D])
		copy(K[i*D:(i+1)*D], qkv[i*3*D+D:i*3*D+2*D])
		copy(V[i*D:(i+1)*D], qkv[i*3*D+2*D:i*3*D+3*D])
	}
	for b := range B {
		off := b * Lmax * D
		rope.apply(Q[off:off+Lmax*D], heads)
		rope.apply(K[off:off+Lmax*D], heads)
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	ctx := s.ctx[:BL*D]
	zeroF32Slice(ctx) // padded positions stay 0
	qH := s.qH[:Lmax*headDim]
	kH := s.kH[:Lmax*headDim]
	vHT := s.vH[:Lmax*headDim]
	ctxHead := s.ctxHead[:Lmax*headDim]
	scores := s.scores[:Lmax*Lmax]
	for b := range B {
		L := realLen[b]
		if L == 0 {
			continue
		}
		seqOff := b * Lmax * D
		for headIdx := range heads {
			qHl := qH[:L*headDim]
			kHl := kH[:L*headDim]
			vHTl := vHT[:headDim*L]
			for i := range L {
				src := seqOff + i*D + headIdx*headDim
				copy(qHl[i*headDim:(i+1)*headDim], Q[src:src+headDim])
				copy(kHl[i*headDim:(i+1)*headDim], K[src:src+headDim])
				for d := range headDim {
					vHTl[d*L+i] = V[src+d]
				}
			}
			scoresL := scores[:L*L]
			matmulBTInto(qHl, kHl, scoresL, L, headDim, L)
			for i := range scoresL {
				scoresL[i] *= scale
			}
			for i := range L {
				softmaxRow(scoresL[i*L : (i+1)*L])
			}
			ctxHeadL := ctxHead[:L*headDim]
			matmulBTInto(scoresL, vHTl, ctxHeadL, L, L, headDim)
			for i := range L {
				dst := seqOff + i*D + headIdx*headDim
				copy(ctx[dst:dst+headDim], ctxHeadL[i*headDim:(i+1)*headDim])
			}
		}
	}
	out := s.out[:BL*D]
	OutProjQ, OutProjScales, _, _ := outProj.Int8()
	matmulBTQ8Into(out, ctx, OutProjQ, OutProjScales, BL, D, D, s.deqW)
	for i := range h {
		h[i] += out[i]
	}
}

// swigluMLPQ8 mirrors swigluMLP with fc11/fc12/fc2 in int8 via matmulBTQ8Into and the
// SiLU gate in f32, writing into pooled scratch (val/gate/mid).
func swigluMLPQ8(h []float32, fc11, fc12, fc2 *linalg.WeightMat,
	D, intermediate, L int, s *scratch) {
	val := s.val[:L*intermediate]
	gate := s.gate[:L*intermediate]
	Fc11Q, Fc11Scales, _, _ := fc11.Int8()
	Fc12Q, Fc12Scales, _, _ := fc12.Int8()
	matmulBTQ8Into(val, h, Fc11Q, Fc11Scales, L, D, intermediate, s.deqW)
	matmulBTQ8Into(gate, h, Fc12Q, Fc12Scales, L, D, intermediate, s.deqW)
	for i, v := range val {
		val[i] = v * silu(gate[i])
	}
	mid := s.mid[:L*D]
	Fc2Q, Fc2Scales, _, _ := fc2.Int8()
	matmulBTQ8Into(mid, val, Fc2Q, Fc2Scales, L, intermediate, D, s.deqW)
	for i := range h {
		h[i] += mid[i]
	}
}
