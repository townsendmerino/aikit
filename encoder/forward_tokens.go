package encoder

// forwardTokens runs the same CodeRankEmbed transformer as `forward`
// but returns the per-token hidden-state matrix (L*D row-major)
// instead of CLS-pooling. Intended for late-interaction / MaxSim
// experiments where the caller wants every token's contextualized
// vector rather than the single CLS summary.
//
// Length L is len(ids) (no truncation here; the tokenizer applies
// the maxSeqLength cap upstream via EncodeQuery / EncodeDoc). The
// returned slice has length L*D; row i is `out[i*D : (i+1)*D]`.
//
// The body is intentionally a near-mirror of `forward` — same
// scratch pool, same layer math — so any future change to the
// transformer block stays a one-place edit per path. The only
// divergence is step 5: instead of returning h[:D] we return all
// of h.
func (w *Weights) forwardTokens(ids []int32) []float32 {
	enterForward()
	defer leaveForward()
	L := len(ids)
	D := w.Cfg.HiddenDim
	if L == 0 {
		return make([]float32, 0)
	}
	heads := w.Cfg.NumHeads
	headDim := w.Cfg.HeadDim()
	intermediate := w.Cfg.IntermediateDim
	eps := w.Cfg.LayerNormEpsilon

	s := getScratch()
	defer putScratch(s)
	s.ensureLayer(L, D, intermediate, heads, headDim, L)

	h := make([]float32, L*D)
	tte0 := w.TokenTypeEmb[:D]
	for i, id := range ids {
		if int(id) < 0 || int(id) >= w.Cfg.VocabSize {
			id = 100
		}
		src := w.WordEmb[int(id)*D : int(id)*D+D]
		dst := h[i*D : (i+1)*D]
		for j := 0; j < D; j++ {
			dst[j] = src[j] + tte0[j]
		}
	}
	layerNorm(h, w.EmbLN_W, w.EmbLN_B, L, D, eps)

	rope := newRopeTable(L, headDim, w.Cfg.RoPEBase)
	for i := 0; i < w.Cfg.NumLayers; i++ {
		l := &w.Layers[i]
		selfAttention(h, l.Wqkv, l.OutProj, heads, headDim, D, L, rope, s)
		layerNorm(h, l.Norm1W, l.Norm1B, L, D, eps)
		swigluMLP(h, l.Fc11, l.Fc12, l.Fc2, D, intermediate, L, s)
		layerNorm(h, l.Norm2W, l.Norm2B, L, D, eps)
	}

	// Return the full L*D matrix. h is the per-call scratch from
	// `make` above, not borrowed from the pool, so the caller owns
	// it and can mutate / L2-normalize per row safely.
	return h
}

// EncodeTokens tokenizes `text` (query or doc per the prefix rule)
// and returns the per-token hidden-state matrix from the forward
// pass. Returns (vectors []float32 of length L*D, L int, error).
// Row i is vectors[i*D : (i+1)*D]; D is m.HiddenDim().
//
// Token-id boundaries: with isQuery=true the token sequence is
// "[CLS] <prefix tokens…> <query tokens…> [SEP]". Callers that want
// to exclude the instruction prefix can find its boundary via
// EncodeQueryPrefixIDs (TODO if needed) or by re-encoding just the
// prefix and counting; for v0 the MaxSim probe in ken handles the
// exclusion by looking up well-known [CLS]/[SEP] ids and the prefix
// token sequence directly.
//
// Used by ken's MaxSim rerank probe; mirror of Encode but returning
// every position instead of CLS-pooling.
func (m *Model) EncodeTokens(text string, isQuery bool) ([]float32, int, error) {
	var (
		ids []int32
		err error
	)
	if isQuery {
		ids, err = EncodeQuery(m.tok, text, m.maxSeqLength)
	} else {
		ids, err = EncodeDoc(m.tok, text, m.maxSeqLength)
	}
	if err != nil {
		return nil, 0, err
	}
	return m.weights.forwardTokens(ids), len(ids), nil
}

// EncodeTokensWithIDs is EncodeTokens plus the underlying token-id
// sequence — useful for the MaxSim probe to mask out the
// query-prefix range and special tokens without re-tokenizing.
func (m *Model) EncodeTokensWithIDs(text string, isQuery bool) ([]float32, []int32, error) {
	var (
		ids []int32
		err error
	)
	if isQuery {
		ids, err = EncodeQuery(m.tok, text, m.maxSeqLength)
	} else {
		ids, err = EncodeDoc(m.tok, text, m.maxSeqLength)
	}
	if err != nil {
		return nil, nil, err
	}
	return m.weights.forwardTokens(ids), ids, nil
}

// HiddenDim is already exposed by model.go; no redeclaration here.
