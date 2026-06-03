package decoder

import "fmt"

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
	_ = layer
	_ = h
	_ = lw
	_ = cfg
	_ = cache
	_ = be
	global := cfg.IsGlobalLayer(layer)
	return nil, fmt.Errorf("decoder.causalAttention(layer=%d global=%v): %w [M3 attention, M4 KV cache, M5 sliding window, QK-norm]",
		layer, global, errNotImplemented)
}

// groupForHead maps a query head to its KV head under GQA.
func groupForHead(qHead, numHeads, numKVHeads int) int {
	return qHead / (numHeads / numKVHeads)
}
