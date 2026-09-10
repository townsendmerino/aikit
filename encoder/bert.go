package encoder

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// bert.go — a MiniLM-class BERT encoder (roadmap §2.2), separate from the
// NomicBert/CodeRankEmbed path so that forward stays untouched. It implements the
// three axes a sentence-transformers BERT model differs on: learned ABSOLUTE
// position embeddings (not RoPE), a GELU FFN (intermediate→output dense, not
// SwiGLU), and mean pooling (not CLS). Post-norm, standard scaled-dot attention.
//
// Weights are PyTorch [out, in], so each linear is h·Wᵀ via matmulBT (A·Bᵀ); the
// shared layerNorm / softmaxRow / poolOne primitives are reused. Parity is pinned
// against all-MiniLM-L6-v2 (TestBERT_parity, golden from scripts/oracle/pin_minilm.py).

type bertConfig struct {
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
	ModelType    string  `json:"model_type"`   // "bert" | "roberta" | "xlm-roberta"
	PadTokenID   int     `json:"pad_token_id"` // RoBERTa/XLM-R offset positions by this + 1
}

type bertLayer struct {
	Wq, Wk, Wv       []float32 // [hidden, hidden]
	Bq, Bk, Bv       []float32 // [hidden]
	Wo, Bo           []float32 // attention output dense [hidden, hidden] + [hidden]
	AttnLNW, AttnLNB []float32 // post-attention LayerNorm [hidden]
	Wi, Bi           []float32 // intermediate.dense [intermediate, hidden] + [intermediate]
	Wd, Bd           []float32 // output.dense [hidden, intermediate] + [hidden]
	OutLNW, OutLNB   []float32 // post-FFN LayerNorm [hidden]
}

// BERT is a loaded MiniLM-class encoder. Immutable after load; the forward is
// read-only-safe for concurrent use (no shared mutable state).
type BERT struct {
	// be is the compute backend for f32 matmuls; nil = the pure-Go path (default).
	be Backend

	cfg     bertConfig
	wordEmb []float32 // [vocab, hidden]
	posEmb  []float32 // [maxPos, hidden] — learned absolute positions
	typeEmb []float32 // [typeVocab, hidden]
	embLNW  []float32
	embLNB  []float32
	layers  []bertLayer
	tok     *embed.Tokenizer       // WordPiece tokenizer (tokenizer.json)
	maxSeq  int                    // sentence-transformers max_seq_length (right-truncation)
	pool    pooling                // reduction read from 1_Pooling/config.json (mean default)
	posOff  int                    // position-id offset (0 for BERT; pad_token_id+1 for RoBERTa/XLM-R)
	st      *embed.SafetensorsFile // retained so the aliased weights stay valid
}

// positionOffset returns the learned-position-id offset for a BERT-family model.
// RoBERTa/XLM-R number positions from padding_idx+1
// (create_position_ids_from_input_ids), so posEmb[0..padding_idx] are reserved
// and real tokens start at padding_idx+1; BERT starts at 0. Applying the offset
// to BERT (a common config that also carries pad_token_id) is a classic
// silent-wrong, so it is gated on model_type, not just pad_token_id.
func positionOffset(modelType string, padTokenID int) int {
	switch modelType {
	case "roberta", "xlm-roberta":
		if padTokenID < 0 {
			return 0
		}
		return padTokenID + 1
	default:
		return 0
	}
}

// LoadBERT loads a sentence-transformers BERT model (config.json +
// model.safetensors with BERT tensor names) from dir. It validates the two
// architecture assumptions this forward implements: GELU activation and learned
// absolute positions.
func LoadBERT(dir string) (*BERT, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("encoder: read BERT config: %w", err)
	}
	var c bertConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("encoder: parse BERT config: %w", err)
	}
	switch {
	case c.Act != "gelu":
		return nil, fmt.Errorf("encoder: BERT hidden_act=%q unsupported (gelu only)", c.Act)
	case c.PosType != "" && c.PosType != "absolute":
		return nil, fmt.Errorf("encoder: BERT position_embedding_type=%q unsupported (absolute only)", c.PosType)
	}
	// BERT uses learned absolute positions, not RoPE, so an odd head dim is fine.
	if err := validateDims("BERT", c.Hidden, c.Heads, c.Layers, c.Intermediate, false); err != nil {
		return nil, err
	}
	if c.LNEps == 0 {
		c.LNEps = 1e-12
	}

	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("encoder: open BERT safetensors: %w", err)
	}
	D, I := c.Hidden, c.Intermediate
	b := &BERT{cfg: c, st: st, layers: make([]bertLayer, c.Layers)}

	// Encoder tensors are bare in sentence-transformers exports (embeddings.*,
	// encoder.layer.N.*) but carry a model-name prefix in a raw *Model /
	// *ForMaskedLM: "bert." (BERT, e.g. SPLADE) or "roberta." (XLM-R/RoBERTa —
	// same tensor layout, different top-level name). Detect which.
	prefix := ""
	if _, e := st.Tensor("embeddings.word_embeddings.weight"); e != nil {
		for _, p := range []string{"bert.", "roberta."} {
			if _, e2 := st.Tensor(p + "embeddings.word_embeddings.weight"); e2 == nil {
				prefix = p
				break
			}
		}
	}

	get := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		var v []float32
		v, err = loadF32(st, name, want)
		return v
	}
	b.posOff = positionOffset(c.ModelType, c.PadTokenID)
	b.wordEmb = get(prefix+"embeddings.word_embeddings.weight", c.VocabSize, D)
	b.posEmb = get(prefix+"embeddings.position_embeddings.weight", c.MaxPos, D)
	// token_type_embeddings is optional — some BERT-family exports (type_vocab_size
	// 0) omit it. When absent, the embedding sum simply skips the segment term.
	if c.TypeVocab > 0 {
		b.typeEmb = get(prefix+"embeddings.token_type_embeddings.weight", c.TypeVocab, D)
	}
	b.embLNW = get(prefix+"embeddings.LayerNorm.weight", D)
	b.embLNB = get(prefix+"embeddings.LayerNorm.bias", D)
	for i := range b.layers {
		p := fmt.Sprintf("%sencoder.layer.%d.", prefix, i)
		l := &b.layers[i]
		l.Wq, l.Bq = get(p+"attention.self.query.weight", D, D), get(p+"attention.self.query.bias", D)
		l.Wk, l.Bk = get(p+"attention.self.key.weight", D, D), get(p+"attention.self.key.bias", D)
		l.Wv, l.Bv = get(p+"attention.self.value.weight", D, D), get(p+"attention.self.value.bias", D)
		l.Wo, l.Bo = get(p+"attention.output.dense.weight", D, D), get(p+"attention.output.dense.bias", D)
		l.AttnLNW, l.AttnLNB = get(p+"attention.output.LayerNorm.weight", D), get(p+"attention.output.LayerNorm.bias", D)
		l.Wi, l.Bi = get(p+"intermediate.dense.weight", I, D), get(p+"intermediate.dense.bias", I)
		l.Wd, l.Bd = get(p+"output.dense.weight", D, I), get(p+"output.dense.bias", D)
		l.OutLNW, l.OutLNB = get(p+"output.LayerNorm.weight", D), get(p+"output.LayerNorm.bias", D)
	}
	if err != nil {
		_ = st.Close()
		return nil, err
	}

	// max sequence length: sentence-transformers right-truncates here. The hard
	// ceiling is the usable position count — MaxPos minus the offset, since a
	// token at index i reads posEmb[i+posOff].
	usablePos := c.MaxPos - b.posOff
	b.maxSeq = usablePos
	if sb, e := os.ReadFile(filepath.Join(dir, "sentence_bert_config.json")); e == nil {
		var v struct {
			MaxSeqLength int `json:"max_seq_length"`
		}
		if json.Unmarshal(sb, &v) == nil && v.MaxSeqLength > 0 {
			// Clamp to the usable position capacity: a checkpoint whose
			// sentence_bert_config claims a longer max_seq_length than the
			// position table would otherwise index posEmb out of range on the
			// first long input instead of truncating.
			b.maxSeq = min(v.MaxSeqLength, usablePos)
		}
	}
	// Tokenizer is best-effort: aikit parses WordPiece and Unigram/SentencePiece
	// (XLM-R family), but a model with some other tokenizer.json can still run the
	// forward on PRE-TOKENIZED ids via Embed. Encode(text), which needs the
	// tokenizer, errors when it's absent.
	if tok, terr := embed.LoadTokenizer(filepath.Join(dir, "tokenizer.json")); terr == nil {
		b.tok = tok
	}
	// Pooling is a declared per-model property (sentence-transformers
	// 1_Pooling/config.json), not assumed: MiniLM pools mean, BGE pools CLS.
	// Absent file → mean, the sentence-transformers BERT default.
	if b.pool, err = poolingFromConfig(dir, poolMean); err != nil {
		_ = st.Close()
		return nil, err
	}
	return b, nil
}

// Close releases the mmap-backed weights. Call once after the last Encode;
// idempotent. See Model.Close.
func (b *BERT) Close() error {
	if b.st == nil {
		return nil
	}
	return b.st.Close()
}

// Encode tokenizes text (WordPiece, wrapped [CLS]…[SEP], right-truncated to the
// model's max sequence length) and returns the mean-pooled, L2-normalized sentence
// embedding — the end-to-end MiniLM equivalent of sentence-transformers' .encode().
func (b *BERT) Encode(text string) ([]float32, error) {
	if b.tok == nil {
		return nil, fmt.Errorf("encoder: BERT has no usable tokenizer (unsupported tokenizer.json); use Embed with pre-tokenized ids")
	}
	ids, err := b.tok.EncodeWithSpecials(text, b.maxSeq)
	if err != nil {
		return nil, err
	}
	return b.Embed(ids), nil
}

// hiddenStates runs the transformer forward on token ids (already wrapped with
// [CLS]…[SEP]) and returns the last hidden state [L, hidden], row-major. segs is the
// token-type (segment) id per position; nil means a single segment (all 0), as the
// embedder uses — a cross-encoder passes two segments for the query/document pair.
func (b *BERT) hiddenStates(ids, segs []int32) []float32 {
	return b.forward(ids, segs, false)
}

// clsHiddenState is hiddenStates for a caller that reads only row 0, returning
// just that row — the [CLS] vector, [hidden] long rather than [L, hidden].
//
// The last transformer layer's other L−1 output rows exist to feed layer N+1's
// K and V, and there is no layer N+1. At L=512, D=768, I=3072 that layer writes
// 31.5 MB to have 1.5 KB read (lens doc §4.1). This runs its Q projection,
// attention output, both LayerNorms and the whole MLP at M=1, keeping only K and
// V at full L because attention genuinely reads every position.
//
// BIT-IDENTICAL to hiddenStates()[0:D], and the reason is the M-invariance
// contract the blocked matmul already owes item 14: dst[i,j] reduces over k-tiles
// fixed by kBlock and K, and M only chooses the m-block boundary. Everything else
// the layer does — addBias, gelu, the residual, layerNorm, softmaxRows — is
// per-row by construction. TestCLSHiddenState_matchesFullRow0 asserts it exactly.
//
// (The lens doc warns that encoder/linalg.go's 4-MFLOP naive/blocked threshold
// would break this at M=1, since 1·768·3072 = 2.36 M falls under it and the naive
// path accumulates in a different order. That threshold no longer exists —
// perf-campaign item 27 deleted it — so the caveat is moot. Do not reintroduce a
// shape-dependent kernel switch without revisiting this.)
func (b *BERT) clsHiddenState(ids, segs []int32) []float32 {
	return b.forward(ids, segs, true)
}

// forward runs the transformer. clsOnly trims the final layer to row 0 — see
// clsHiddenState.
func (b *BERT) forward(ids, segs []int32, clsOnly bool) []float32 {
	// The whole BERT family — BERT.Embed, CrossEncoder.Score, SPLADE.Expand —
	// enters the transformer here, so this is where the in-flight accounting
	// parallel.go:37-40 requires has to happen. Without it that counter stayed
	// permanently 0 for every one of them, and the intra-op gate read "nothing
	// else is running" unconditionally: the natural reranker loop (N goroutines
	// each calling Score) became NumCPU×NumCPU runnable goroutines contending at
	// the join barrier — exactly the oversubscription the gate exists to prevent
	// (perf-campaign item 6).
	enterForward()
	defer leaveForward()

	c := b.cfg
	// Bound to the learned-position table: posEmb has only maxSeq (≤ MaxPos) rows,
	// so gathering posEmb[i*D] for i beyond it panics. Encode already truncates
	// upstream, but the exported Embed takes raw ids with no documented limit —
	// truncate here so every entry point is safe (segs, if present, in lockstep).
	if len(ids) > b.maxSeq {
		ids = ids[:b.maxSeq]
		if segs != nil {
			segs = segs[:b.maxSeq]
		}
	}
	L, D := len(ids), c.Hidden
	headDim := D / c.Heads
	eps := c.LNEps

	// Reuse a pooled scratch arena for the per-layer/per-head buffers instead of
	// allocating qH/kH/vHT + an L² scores per (layer, head) and Q/K/V/ctx/out/
	// inter/ffn per layer — the ~432 small mallocs per forward the f32 path
	// already eliminated. Also routes the projections through matmulBTInto, which
	// (unlike the allocating matmulBT) parallelizes a lone forward — so a bare
	// SPLADE.Expand / CrossEncoder.Score no longer runs single-threaded.
	s := getScratch()
	s.be = b.be
	defer putScratch(s)
	s.ensureLayer(L, D, c.Intermediate, c.Heads, headDim, L)

	// Embeddings: word + learned position (offset for RoBERTa/XLM-R) + optional
	// token-type[seg], then LayerNorm.
	h := make([]float32, L*D)
	vocab, typeVocab := len(b.wordEmb)/D, len(b.typeEmb)/D
	for i, id := range ids {
		w := b.wordEmb[clampTokenID(id, vocab)*D:][:D]
		pos := b.posEmb[(i+b.posOff)*D : (i+b.posOff)*D+D]
		row := h[i*D : i*D+D]
		for j := range D {
			row[j] = w[j] + pos[j]
		}
		// token-type is optional (absent when type_vocab_size == 0).
		if typeVocab > 0 {
			seg := 0
			if segs != nil {
				seg = int(segs[i])
			}
			if seg < 0 || seg >= typeVocab {
				seg = 0
			}
			typ := b.typeEmb[seg*D : seg*D+D]
			for j := range D {
				row[j] += typ[j]
			}
		}
	}
	layerNorm(h, b.embLNW, b.embLNB, L, D, eps)

	for li := range b.layers {
		l := &b.layers[li]

		// mOut is how many rows of this layer's OUTPUT are needed. It is L
		// everywhere except the final layer of a CLS-only forward, where only row
		// 0 is ever read. K and V stay at full L regardless: attention reads every
		// position, and trimming them would change the result rather than skip
		// unused work.
		mOut := L
		if clsOnly && li == len(b.layers)-1 {
			mOut = 1
		}

		// Self-attention (no RoPE): Q,K,V = hWᵀ + b, into scratch. Q is
		// projected only for the mOut rows actually needed; K,V stay full L
		// since attention reads every position regardless of mOut.
		Q, K, V := s.Q[:mOut*D], s.K[:L*D], s.V[:L*D]
		s.mm(h, l.Wq, Q, mOut, D, D)
		s.mm(h, l.Wk, K, L, D, D)
		s.mm(h, l.Wv, V, L, D, D)
		addBias(Q, l.Bq, mOut, D)
		addBias(K, l.Bk, L, D)
		addBias(V, l.Bv, L, D)

		// Scaled dot-product attention, output projection and residual — shared
		// with the CodeRankEmbed/nomic and GTE forwards via encoder/attention.go.
		// BERT has neither RoPE nor a packed qkv projection, so it builds Q/K/V
		// itself above and calls attentionCore directly rather than selfAttention.
		h = attentionCore(h, Q, K, V, l.Wo, l.Bo, c.Heads, headDim, D, L, mOut, s)
		layerNorm(h, l.AttnLNW, l.AttnLNB, mOut, D, eps)

		// GELU FFN: intermediate → gelu → output, residual, LayerNorm.
		inter := s.val[:mOut*c.Intermediate]
		s.mm(h, l.Wi, inter, mOut, D, c.Intermediate)
		addBias(inter, l.Bi, mOut, c.Intermediate)
		gelu(inter)
		ffn := s.mid[:mOut*D]
		s.mm(inter, l.Wd, ffn, mOut, c.Intermediate, D)
		addBias(ffn, l.Bd, mOut, D)
		for i := range h {
			h[i] += ffn[i] // residual
		}
		layerNorm(h, l.OutLNW, l.OutLNB, mOut, D, eps)
	}
	return h
}

// Embed returns the pooled (per the model's 1_Pooling config; mean by default),
// L2-normalized sentence embedding for token ids.
func (b *BERT) Embed(ids []int32) []float32 {
	D := b.cfg.Hidden
	if b.pool == poolCLS {
		// Only row 0 is read, so the last layer need not produce the rest
		// (lens doc §4.1). poolOne(h, 1, D, poolCLS) is a copy of h[0:D].
		return l2norm(poolOne(b.clsHiddenState(ids, nil), 1, D, poolCLS))
	}
	h := b.hiddenStates(ids, nil)
	// L = len(h)/D is the token count hiddenStates actually produced — it
	// truncates ids to maxSeq, so len(ids) can be larger and would over-read.
	v := poolOne(h, len(h)/D, D, b.pool)
	return l2norm(v)
}

// l2norm is the encoder-package name for the kit's single canonical
// L2-normalize; it forwards to embed.L2Normalize so the pooling path, the
// Matryoshka Truncate, and every encoder embedder share one implementation and
// one degenerate-case behavior (audit #21).
func l2norm(v []float32) []float32 { return embed.L2Normalize(v) }

// gelu applies the exact (erf-based) GELU in place: x·Φ(x) = 0.5x(1+erf(x/√2)).
// transformers' "gelu" activation is the erf form, not the tanh approximation.
func gelu(x []float32) {
	// Chunked so a long activation block (BERT's [L,intermediate]) splits across
	// cores; each chunk is expensive enough (math.Erf-class transcendental) that
	// this is one of the largest serial stages once the matmuls are parallel.
	// Elementwise and in place, so the split is numerically inert. Each chunk
	// routes through linalg.GELUInto (vectorized under GOEXPERIMENT=simd — T1,
	// autoresearch round 2) instead of a per-element scalar loop, so this now
	// layers SIMD-within-a-chunk under the existing core-level parallelism —
	// this was the encoder's own gap the autoresearch loop found: GELUInto had
	// real callers (vision/qwen_encoder.go) but not this one, its biggest.
	const chunk = 4096
	chunks := (len(x) + chunk - 1) / chunk
	parallelRows(chunks, len(x), func(start, end int) {
		lo, hi := start*chunk, min(end*chunk, len(x))
		linalg.GELUContractInto(x[lo:hi], x[lo:hi])
	})
}

// geluTanh applies the tanh-approximation GELU in place — HF's "gelu_new" /
// "gelu_fast" / "gelu_pytorch_tanh", a different function from the exact erf
// form gelu applies (see Config.geluTanh and linalg.GELUTanhF32). Same chunking
// as gelu, for the same reason: elementwise and in place, so splitting is
// numerically inert. Same SIMD-under-parallelism fix as gelu (T1, autoresearch
// round 3): each chunk routes through linalg.GELUTanhInto instead of a
// per-element scalar loop.
func geluTanh(x []float32) {
	const chunk = 4096
	chunks := (len(x) + chunk - 1) / chunk
	parallelRows(chunks, len(x), func(start, end int) {
		lo, hi := start*chunk, min(end*chunk, len(x))
		linalg.GELUTanhContractInto(x[lo:hi], x[lo:hi])
	})
}
