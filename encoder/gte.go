package encoder

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// gte.go — the GTE encoder (Alibaba gte-multilingual-base; the base of
// Snowflake/snowflake-arctic-embed-*-v2.0). It is a post-norm BERT-family encoder
// like bert.go, but on three axes it matches the nomic-bert path instead:
// ROTARY positions (not learned-absolute), a PACKED qkv projection, and a GATED
// GELU MLP (GeGLU) — distinct from both the plain-GELU BERT FFN and the SiLU
// SwiGLU of the nomic v1.5 path. The RoPE convention is bit-identical to
// encoder/rope.go (rotate_half, full head-dim), so that is reused verbatim at the
// model's base (rope_theta = 160000).
//
//	emb   = LayerNorm( word[ids] + token_type[0] )        // no position embedding
//	attn  = softmax( (RoPE(Q)·RoPE(K)ᵀ)/√d ) · V          // packed qkv, biases on both
//	h     = LayerNorm( h + attn·Woᵀ )                     // post-norm
//	mlp   = ( gelu(gate) ⊙ up ) · Downᵀ                   // up_gate fused [2I,H]
//	h     = LayerNorm( h + mlp )                          // post-norm
//
// Parity is pinned against arctic-embed-m-v2.0 (TestGTE_parity, scripts/oracle/pin_gte.py).

type gteConfig struct {
	VocabSize    int     `json:"vocab_size"`
	Hidden       int     `json:"hidden_size"`
	Layers       int     `json:"num_hidden_layers"`
	Heads        int     `json:"num_attention_heads"`
	Intermediate int     `json:"intermediate_size"`
	MaxPos       int     `json:"max_position_embeddings"`
	TypeVocab    int     `json:"type_vocab_size"`
	LNEps        float64 `json:"layer_norm_eps"`
	Act          string  `json:"hidden_act"`
	PosType      string  `json:"position_embedding_type"`
	RopeTheta    float64 `json:"rope_theta"`
	LNType       string  `json:"layer_norm_type"`
	ModelType    string  `json:"model_type"`
}

type gteLayer struct {
	Wqkv, WqkvB      []float32 // packed [3*hidden, hidden] + [3*hidden]
	Wo, WoB          []float32 // attention out_proj [hidden, hidden] + [hidden]
	AttnLNW, AttnLNB []float32 // post-attention LayerNorm
	UpGate           []float32 // fused up/gate [2*intermediate, hidden] (no bias)
	Down, DownB      []float32 // down_proj [hidden, intermediate] + [hidden]
	MLPLNW, MLPLNB   []float32 // post-MLP LayerNorm
}

// GTE is a loaded gte-multilingual-class encoder. Immutable after load; the
// forward is read-only-safe for concurrent use.
type GTE struct {
	// be is the compute backend for f32 matmuls; nil = the pure-Go path (default).
	be Backend

	cfg     gteConfig
	wordEmb []float32 // [vocab, hidden]
	typeEmb []float32 // [typeVocab, hidden] — row 0 added to every token
	embLNW  []float32
	embLNB  []float32
	layers  []gteLayer
	tok     *embed.Tokenizer
	maxSeq  int
	pool    pooling
	st      *embed.SafetensorsFile
	// ropeCache holds the rotary table across Encodes instead of rebuilding it
	// per call. Grows to the longest sequence seen; zero value ready to use.
	ropeCache ropeCache
}

// LoadGTE loads a GTE encoder (config.json + model.safetensors with GTE tensor
// names) from dir. It validates the architecture assumptions this forward
// implements: rotary positions, GELU activation, plain LayerNorm.
func LoadGTE(dir string) (*GTE, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("encoder: read GTE config: %w", err)
	}
	var c gteConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("encoder: parse GTE config: %w", err)
	}
	switch {
	case c.Act != "gelu":
		return nil, fmt.Errorf("encoder: GTE hidden_act=%q unsupported (gelu only)", c.Act)
	case c.PosType != "rope":
		return nil, fmt.Errorf("encoder: GTE position_embedding_type=%q unsupported (rope only)", c.PosType)
	case c.LNType != "" && c.LNType != "layer_norm":
		return nil, fmt.Errorf("encoder: GTE layer_norm_type=%q unsupported (layer_norm only)", c.LNType)
	}
	if err := validateDims("GTE", c.Hidden, c.Heads, c.Layers, c.Intermediate, true); err != nil {
		return nil, err
	}
	if c.LNEps == 0 {
		c.LNEps = 1e-12
	}
	if c.RopeTheta == 0 {
		c.RopeTheta = 10000
	}

	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("encoder: open GTE safetensors: %w", err)
	}
	D, I := c.Hidden, c.Intermediate
	g := &GTE{cfg: c, st: st, layers: make([]gteLayer, c.Layers)}

	get := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		var v []float32
		v, err = loadF32(st, name, want)
		return v
	}
	g.wordEmb = get("embeddings.word_embeddings.weight", c.VocabSize, D)
	if c.TypeVocab > 0 {
		g.typeEmb = get("embeddings.token_type_embeddings.weight", c.TypeVocab, D)
	}
	g.embLNW = get("embeddings.LayerNorm.weight", D)
	g.embLNB = get("embeddings.LayerNorm.bias", D)
	for i := range g.layers {
		p := fmt.Sprintf("encoder.layer.%d.", i)
		l := &g.layers[i]
		l.Wqkv, l.WqkvB = get(p+"attention.qkv_proj.weight", 3*D, D), get(p+"attention.qkv_proj.bias", 3*D)
		l.Wo, l.WoB = get(p+"attention.o_proj.weight", D, D), get(p+"attention.o_proj.bias", D)
		l.AttnLNW, l.AttnLNB = get(p+"attn_ln.weight", D), get(p+"attn_ln.bias", D)
		l.UpGate = get(p+"mlp.up_gate_proj.weight", 2*I, D)
		l.Down, l.DownB = get(p+"mlp.down_proj.weight", D, I), get(p+"mlp.down_proj.bias", D)
		l.MLPLNW, l.MLPLNB = get(p+"mlp_ln.weight", D), get(p+"mlp_ln.bias", D)
	}
	if err != nil {
		_ = st.Close()
		return nil, err
	}

	g.maxSeq = c.MaxPos
	if sb, e := os.ReadFile(filepath.Join(dir, "sentence_bert_config.json")); e == nil {
		var v struct {
			MaxSeqLength int `json:"max_seq_length"`
		}
		if json.Unmarshal(sb, &v) == nil && v.MaxSeqLength > 0 {
			g.maxSeq = min(v.MaxSeqLength, c.MaxPos)
		}
	}
	if tok, terr := embed.LoadTokenizer(filepath.Join(dir, "tokenizer.json")); terr == nil {
		g.tok = tok
	}
	// GTE (arctic-embed-v2.0) pools CLS; read it from 1_Pooling rather than assume.
	if g.pool, err = poolingFromConfig(dir, poolCLS); err != nil {
		_ = st.Close()
		return nil, err
	}
	return g, nil
}

// Close releases the mmap-backed weights. Idempotent.
func (g *GTE) Close() error {
	if g.st == nil {
		return nil
	}
	return g.st.Close()
}

// Encode tokenizes text (SentencePiece/Unigram, wrapped <s>…</s>, right-truncated
// to the model's max sequence length) and returns the pooled, L2-normalized
// sentence embedding.
func (g *GTE) Encode(text string) ([]float32, error) {
	if g.tok == nil {
		return nil, fmt.Errorf("encoder: GTE has no usable tokenizer; use Embed with pre-tokenized ids")
	}
	ids, err := g.tok.EncodeWithSpecials(text, g.maxSeq)
	if err != nil {
		return nil, err
	}
	return g.Embed(ids), nil
}

// Embed returns the pooled (per 1_Pooling; CLS by default), L2-normalized sentence
// embedding for token ids (already wrapped with <s>…</s>).
func (g *GTE) Embed(ids []int32) []float32 {
	D := g.cfg.Hidden
	if g.pool == poolCLS {
		// Only row 0 is read, so the last layer need not produce the rest.
		return l2norm(poolOne(g.clsHiddenState(ids), 1, D, poolCLS))
	}
	h := g.hiddenStates(ids)
	return l2norm(poolOne(h, len(h)/D, D, g.pool))
}

// hiddenStates runs the GTE transformer forward on token ids and returns the last
// hidden state [L, hidden], row-major.
func (g *GTE) hiddenStates(ids []int32) []float32 { return g.forward(ids, false) }

// clsHiddenState is hiddenStates for a caller that reads only row 0, returning
// just that row. See BERT.clsHiddenState — same argument, same bit-identity
// contract, a second forward implementation (RoPE + a gated MLP) rather than a
// shared one.
func (g *GTE) clsHiddenState(ids []int32) []float32 { return g.forward(ids, true) }

func (g *GTE) forward(ids []int32, clsOnly bool) []float32 {
	// In-flight accounting for the intra-op matmul gate — see BERT.hiddenStates
	// (perf-campaign item 6). GTE.Encode had the same permanently-0 counter.
	enterForward()
	defer leaveForward()

	c := g.cfg
	if len(ids) > g.maxSeq {
		ids = ids[:g.maxSeq]
	}
	L, D := len(ids), c.Hidden
	headDim := D / c.Heads
	I := c.Intermediate
	eps := c.LNEps

	s := getScratch()
	s.be = g.be
	defer putScratch(s)
	s.ensureLayer(L, D, I, c.Heads, headDim, L)
	s.ensureFusedMLP(L, I)
	rope := g.ropeCache.get(L, headDim, c.RopeTheta)

	// Embeddings: word + token_type[0] (positions come from RoPE), then LayerNorm.
	h := make([]float32, L*D)
	vocab := len(g.wordEmb) / D
	for i, id := range ids {
		w := g.wordEmb[clampTokenID(id, vocab)*D:][:D]
		row := h[i*D : i*D+D]
		copy(row, w)
		if g.typeEmb != nil { // type_vocab_size == 1 → row 0 added to every token
			for j := range D {
				row[j] += g.typeEmb[j]
			}
		}
	}
	layerNorm(h, g.embLNW, g.embLNB, L, D, eps)

	upGate := s.upGate[:L*2*I]
	for li := range g.layers {
		l := &g.layers[li]

		// mOut is how many rows of this layer's OUTPUT are needed — L everywhere
		// except the final layer of a CLS-only forward. K and V stay at full L:
		// attention reads every position.
		mOut := L
		if clsOnly && li == len(g.layers)-1 {
			mOut = 1
		}

		// selfAttention's packed QKV projection stays at FULL L even in the
		// trimmed layer, and that is a deliberate retreat from the obvious
		// version. Wqkv is [3D, D] row-major, so Wq/Wk/Wv are contiguous and
		// splitting it to compute Q for one row is easy — and measurably worse
		// at short input: three narrower matmuls in place of one wide one cost
		// more than the L−1 rows of Q they save, and GTEEncode/L16 regressed
		// 10% while L128 and L512 gained 7%. Keeping it packed leaves ~17% of
		// the last layer's available saving on the table (the projection is
		// 906 of 5235 MFLOP at L=512) and removes the regression entirely; the
		// MLP, which is 69% of the layer, is trimmed either way.
		//
		// The attention math itself — RoPE, scaled dot-product, output
		// projection, residual — is shared with the CodeRankEmbed/nomic path
		// (also packed Wqkv + RoPE) via encoder/attention.go; BERT shares only
		// the post-projection half (attentionCore), since it has neither RoPE
		// nor a packed qkv.
		h = selfAttention(h, l.Wqkv, l.WqkvB, l.Wo, l.WoB, c.Heads, headDim, D, L, mOut, rope, s)
		layerNorm(h, l.AttnLNW, l.AttnLNB, mOut, D, eps)

		// GeGLU MLP: up_gate = h·UpGateᵀ [L,2I]; split up=[:I], gate=[I:2I];
		// mid = gelu(gate) ⊙ up; ffn = mid·Downᵀ + b; residual; post-norm.
		s.mm(h, l.UpGate, upGate, mOut, D, 2*I)
		mid := s.val[:mOut*I]
		// Split by token row: each writes only its own mid[i], and the
		// activation is elementwise, so this is numerically inert. It is worth
		// splitting because geluScalar is math.Erf per element — after the
		// attention softmax was parallelized this was the largest remaining
		// serial stage of GTE.Encode, ~27% of the call at L=690.
		// Each row's gate is contiguous (I-length), so linalg.GELUInto (vectorized
		// under GOEXPERIMENT=simd — T1, autoresearch round 2) applies per row, in
		// place, before the up-multiply — this row-parallel loop was the encoder's
		// own biggest GELU hotspot (~27% of GTE.Encode at L=690) that the batched
		// kernel's vectorization never actually reached until now.
		parallelRows(mOut, mOut*I, func(start, end int) {
			for i := start; i < end; i++ {
				up := upGate[i*2*I : i*2*I+I]
				gate := upGate[i*2*I+I : i*2*I+2*I]
				m := mid[i*I : (i+1)*I]
				linalg.GELUContractInto(gate, gate)
				for j := range I {
					m[j] = gate[j] * up[j]
				}
			}
		})
		ffn := s.mid[:mOut*D]
		s.mm(mid, l.Down, ffn, mOut, I, D)
		addBias(ffn, l.DownB, mOut, D)
		for i := range h {
			h[i] += ffn[i] // residual
		}
		layerNorm(h, l.MLPLNW, l.MLPLNB, mOut, D, eps)
	}
	return h
}
