package decoder

import "math"

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
	cfg *Config,
	cache *KVCache,
	be Backend,
) ([]float32, error) {
	hidden, nH, nKV, hd := cfg.HiddenDim, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	qDim, kvDim := nH*hd, nKV*hd
	global := cfg.IsGlobalLayer(layer)
	pos := cache.Pos() // this token's absolute position (stable across layers in one forward)

	// 1. Project to q/k/v for the new position.
	q := make([]float32, qDim)
	k := make([]float32, kvDim)
	v := make([]float32, kvDim)
	be.MatmulBT(h, lw.QProj, q, 1, hidden, qDim)
	be.MatmulBT(h, lw.KProj, k, 1, hidden, kvDim)
	be.MatmulBT(h, lw.VProj, v, 1, hidden, kvDim)

	// 2. QK-norm: Gemma 3 RMSNorm((1+w)) over head_dim, per head, before RoPE.
	rmsNorm(q, lw.QNorm, nH, hd, cfg.RMSNormEps)
	rmsNorm(k, lw.KNorm, nKV, hd, cfg.RMSNormEps)

	// 3. RoPE at pos with the per-layer base (local 10k vs global 1e6).
	base := cfg.RoPELocalBase
	if global {
		base = cfg.RoPEGlobalBase
	}
	applyRoPE(q, nH, hd, pos, base)
	applyRoPE(k, nKV, hd, pos, base)

	// 4. Append this position's K/V, then attend over the stored history.
	cache.Append(layer, k, v)
	keys, vals := cache.Keys(layer), cache.Vals(layer)
	nKeys := len(keys) / kvDim // == pos+1
	start := cache.WindowStart(global)
	scale := math.Pow(cfg.QueryPreAttnScalar, -0.5)
	group := nH / nKV // GQA: query heads per KV head

	ctx := make([]float32, qDim)
	scores := make([]float32, nKeys)
	for qh := 0; qh < nH; qh++ {
		kvh := qh / group
		qHead := q[qh*hd : qh*hd+hd]

		// 5. Scaled dot-product scores over the causal/window range.
		maxS := math.Inf(-1)
		for s := start; s < nKeys; s++ {
			kHead := keys[s*kvDim+kvh*hd : s*kvDim+kvh*hd+hd]
			var dot float64
			for d := 0; d < hd; d++ {
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
			for d := 0; d < hd; d++ {
				oHead[d] += w * vHead[d]
			}
		}
	}

	// 7. Output projection; caller applies post-attn norm + residual.
	out := make([]float32, hidden)
	be.MatmulBT(ctx, lw.OProj, out, 1, qDim, hidden)
	return out, nil
}

// groupForHead maps a query head to its KV head under GQA.
func groupForHead(qHead, numHeads, numKVHeads int) int {
	return qHead / (numHeads / numKVHeads)
}
