package decoder

import "math"

// addBias adds a per-output bias vector to a projection result in place
// (Qwen2's q/k/v projections). len(b) must equal len(x).
func addBias(x, b []float32) {
	for i := range x {
		x[i] += b[i]
	}
}

// causalAttention runs one decoder block's grouped-query causal attention
// for a single decode step (the query is the one new position; keys/values
// come from the KV cache plus this step's own K/V).
//
// SCAFFOLD: the signature, shapes and the sequence of operations are spelled
// out so M3/M4/M5 are a fill-in. The body returns errNotImplemented.
//
// Shapes / steps the implementation will follow:
//
//	h        [HiddenDim]                  the current position's hidden state (post pre-norm)
//	QProj    [NumHeads*HeadDim, HiddenDim]
//	KProj/VProj [NumKVHeads*HeadDim, HiddenDim]
//	OProj    [HiddenDim, NumHeads*HeadDim]
//
//	1. q = QProj·h   (NumHeads heads);  k,v = KProj·h, VProj·h  (NumKVHeads heads)
//	2. QK-norm: rmsNorm each q head and each k head over HeadDim (Gemma 3)
//	3. RoPE q,k at absolute position cache.Pos() using the LOCAL or GLOBAL
//	   table per global; rotate_half (reuse encoder/rope.go, shared).
//	4. cache.Append(layer, k, v)
//	5. for each query head, attend over cache keys in
//	   [cache.WindowStart(global), cache.Pos()) — the GQA group maps query
//	   head h → kv head h/(NumHeads/NumKVHeads). scale by 1/sqrt(QueryPreAttnScalar).
//	6. softmax (encoder.softmaxRow, shared) → weighted sum of values → ctx
//	7. out = OProj·ctx ; caller applies post-attn norm + residual add.
func causalAttention(
	layer int,
	h []float32,
	lw *LayerWeights,
	arch *Architecture,
	cache *KVCache,
	be Backend,
) ([]float32, error) {
	hidden, nH, nKV, hd := arch.HiddenDim, arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	qDim, kvDim := nH*hd, nKV*hd
	global := arch.isGlobalLayer(layer)
	pos := cache.Pos() // this token's absolute position (stable across layers in one forward)

	// 1. Project to q/k/v for the new position.
	q := make([]float32, qDim)
	k := make([]float32, kvDim)
	v := make([]float32, kvDim)
	lw.QProj.matmul(be, h, q, 1)
	lw.KProj.matmul(be, h, k, 1)
	lw.VProj.matmul(be, h, v, 1)
	if arch.QKVBias {
		addBias(q, lw.QBias)
		addBias(k, lw.KBias)
		addBias(v, lw.VBias)
	}

	// 2. QK-norm (Gemma 3, Qwen3): RMSNorm over head_dim, per head, before RoPE.
	if arch.QKNorm {
		rmsNorm(q, lw.QNorm, nH, hd, arch.NormEps, arch.RMSAddOne)
		rmsNorm(k, lw.KNorm, nKV, hd, arch.NormEps, arch.RMSAddOne)
	}

	// 3. RoPE at pos with the per-layer inv-freq table (Gemma: local 10k vs
	// global 1e6 base; Llama-3: llama3 scaling baked in; Phi: partial rotary).
	// GPT-2 uses learned absolute positions instead, so it skips RoPE.
	if !arch.LearnedPosEmbed {
		invFreq := arch.ropeInvFreq(layer)
		applyRoPE(q, nH, hd, pos, invFreq)
		applyRoPE(k, nKV, hd, pos, invFreq)
	}

	// 4. Append this position's K/V, then attend over the stored history.
	cache.Append(layer, k, v)
	keys, vals := cache.Keys(layer), cache.Vals(layer)
	nKeys := len(keys) / kvDim // == pos+1
	start := cache.WindowStart(pos, global)
	scale := arch.AttnScale // resolved: query_pre_attn_scalar^-0.5 (Gemma) or 1/sqrt(headDim)
	group := nH / nKV       // GQA: query heads per KV head

	ctx := make([]float32, qDim)
	scores := make([]float32, nKeys)
	for qh := range nH {
		kvh := qh / group
		qHead := q[qh*hd : qh*hd+hd]

		// 5. Scaled dot-product scores over the causal/window range.
		maxS := math.Inf(-1)
		for s := start; s < nKeys; s++ {
			kHead := keys[s*kvDim+kvh*hd : s*kvDim+kvh*hd+hd]
			var dot float64
			for d := range hd {
				dot += float64(qHead[d]) * float64(kHead[d])
			}
			sc := dot * scale
			scores[s] = float32(sc)
			if sc > maxS {
				maxS = sc
			}
		}

		// 6. Softmax → weighted sum of values into this head's context slice.
		var sum float64
		for s := start; s < nKeys; s++ {
			e := math.Exp(float64(scores[s]) - maxS)
			scores[s] = float32(e)
			sum += e
		}
		inv := 1.0 / sum
		oHead := ctx[qh*hd : qh*hd+hd]
		for s := start; s < nKeys; s++ {
			w := float32(float64(scores[s]) * inv)
			vHead := vals[s*kvDim+kvh*hd : s*kvDim+kvh*hd+hd]
			for d := range hd {
				oHead[d] += w * vHead[d]
			}
		}
	}

	// 7. Output projection (+ bias for GPT-2); caller applies post-attn norm + residual.
	out := make([]float32, hidden)
	lw.OProj.matmul(be, ctx, out, 1)
	if arch.OutBias {
		addBias(out, lw.OBias)
	}
	return out, nil
}
