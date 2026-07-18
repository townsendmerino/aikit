package encoder

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
)

// bert.go — a MiniLM-class BERT encoder (roadmap §2.2), separate from the
// NomicBert/CodeRankEmbed path so that forward stays untouched. It implements the
// three axes a sentence-transformers BERT model differs on: learned ABSOLUTE
// position embeddings (not RoPE), a GELU FFN (intermediate→output dense, not
// SwiGLU), and mean pooling (not CLS). Post-norm, standard scaled-dot attention.
//
// Weights are PyTorch [out, in], so each linear is h·Wᵀ via matmulBT (A·Bᵀ); the
// shared layerNorm / softmaxRow / poolOne primitives are reused. Parity is pinned
// against all-MiniLM-L6-v2 (TestBERT_parity, golden from scripts/pin_minilm.py).

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
	cfg     bertConfig
	wordEmb []float32 // [vocab, hidden]
	posEmb  []float32 // [maxPos, hidden] — learned absolute positions
	typeEmb []float32 // [typeVocab, hidden]
	embLNW  []float32
	embLNB  []float32
	layers  []bertLayer
	tok     *embed.Tokenizer       // WordPiece tokenizer (tokenizer.json)
	maxSeq  int                    // sentence-transformers max_seq_length (right-truncation)
	st      *embed.SafetensorsFile // retained so the aliased weights stay valid
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
	case c.Hidden == 0 || c.Heads == 0 || c.Layers == 0 || c.Intermediate == 0:
		return nil, fmt.Errorf("encoder: BERT config missing a required dim")
	case c.Hidden%c.Heads != 0:
		return nil, fmt.Errorf("encoder: BERT hidden %d not divisible by heads %d", c.Hidden, c.Heads)
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
	// encoder.layer.N.*) but carry a "bert." prefix in a raw BertModel /
	// BertForMaskedLM (e.g. SPLADE). Detect which.
	prefix := ""
	if _, e := st.Tensor("embeddings.word_embeddings.weight"); e != nil {
		if _, e2 := st.Tensor("bert.embeddings.word_embeddings.weight"); e2 == nil {
			prefix = "bert."
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
	b.wordEmb = get(prefix+"embeddings.word_embeddings.weight", c.VocabSize, D)
	b.posEmb = get(prefix+"embeddings.position_embeddings.weight", c.MaxPos, D)
	b.typeEmb = get(prefix+"embeddings.token_type_embeddings.weight", c.TypeVocab, D)
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

	// max sequence length: sentence-transformers right-truncates here (the position
	// table is the hard ceiling). Falls back to the position capacity.
	b.maxSeq = c.MaxPos
	if sb, e := os.ReadFile(filepath.Join(dir, "sentence_bert_config.json")); e == nil {
		var v struct {
			MaxSeqLength int `json:"max_seq_length"`
		}
		if json.Unmarshal(sb, &v) == nil && v.MaxSeqLength > 0 {
			// Clamp to the position-embedding capacity: a checkpoint whose
			// sentence_bert_config claims a longer max_seq_length than
			// max_position_embeddings would otherwise index posEmb out of range
			// on the first long input instead of truncating.
			b.maxSeq = min(v.MaxSeqLength, c.MaxPos)
		}
	}
	tok, terr := embed.LoadTokenizer(filepath.Join(dir, "tokenizer.json"))
	if terr != nil {
		_ = st.Close()
		return nil, fmt.Errorf("encoder: BERT tokenizer: %w", terr)
	}
	b.tok = tok
	return b, nil
}

// Encode tokenizes text (WordPiece, wrapped [CLS]…[SEP], right-truncated to the
// model's max sequence length) and returns the mean-pooled, L2-normalized sentence
// embedding — the end-to-end MiniLM equivalent of sentence-transformers' .encode().
func (b *BERT) Encode(text string) ([]float32, error) {
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
	c := b.cfg
	L, D := len(ids), c.Hidden
	headDim := D / c.Heads
	eps := c.LNEps

	// Embeddings: word + learned position + token-type[seg], then LayerNorm.
	h := make([]float32, L*D)
	vocab, typeVocab := len(b.wordEmb)/D, len(b.typeEmb)/D
	for i, id := range ids {
		seg := 0
		if segs != nil {
			seg = int(segs[i])
		}
		// Defensive: a corrupt tokenizer/checkpoint could emit an OOB id or
		// segment; substitute row 0 (always in range) rather than panic deep in
		// the embedding gather. The tokenizer should never produce these.
		if int(id) < 0 || int(id) >= vocab {
			id = 0
		}
		if seg < 0 || seg >= typeVocab {
			seg = 0
		}
		w := b.wordEmb[int(id)*D : int(id)*D+D]
		pos := b.posEmb[i*D : i*D+D]
		typ := b.typeEmb[seg*D : seg*D+D]
		row := h[i*D : i*D+D]
		for j := range D {
			row[j] = w[j] + pos[j] + typ[j]
		}
	}
	layerNorm(h, b.embLNW, b.embLNB, L, D, eps)

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	for li := range b.layers {
		l := &b.layers[li]

		// Self-attention (no RoPE): Q,K,V = hWᵀ + b.
		Q := matmulBT(h, l.Wq, L, D, D)
		K := matmulBT(h, l.Wk, L, D, D)
		V := matmulBT(h, l.Wv, L, D, D)
		addBias(Q, l.Bq, L, D)
		addBias(K, l.Bk, L, D)
		addBias(V, l.Bv, L, D)

		ctx := make([]float32, L*D)
		for hd := 0; hd < c.Heads; hd++ {
			qH := make([]float32, L*headDim)
			kH := make([]float32, L*headDim)
			vHT := make([]float32, headDim*L)
			for i := range L {
				src := i*D + hd*headDim
				copy(qH[i*headDim:(i+1)*headDim], Q[src:src+headDim])
				copy(kH[i*headDim:(i+1)*headDim], K[src:src+headDim])
				for d := range headDim {
					vHT[d*L+i] = V[src+d]
				}
			}
			scores := matmulBT(qH, kH, L, headDim, L)
			for i := range scores {
				scores[i] *= scale
			}
			for i := range L {
				softmaxRow(scores[i*L : (i+1)*L])
			}
			ctxHead := matmulBT(scores, vHT, L, L, headDim)
			for i := range L {
				copy(ctx[i*D+hd*headDim:i*D+hd*headDim+headDim], ctxHead[i*headDim:(i+1)*headDim])
			}
		}
		attnOut := matmulBT(ctx, l.Wo, L, D, D)
		addBias(attnOut, l.Bo, L, D)
		for i := range h {
			h[i] += attnOut[i] // residual
		}
		layerNorm(h, l.AttnLNW, l.AttnLNB, L, D, eps)

		// GELU FFN: intermediate → gelu → output, residual, LayerNorm.
		inter := matmulBT(h, l.Wi, L, D, c.Intermediate)
		addBias(inter, l.Bi, L, c.Intermediate)
		gelu(inter)
		ffn := matmulBT(inter, l.Wd, L, c.Intermediate, D)
		addBias(ffn, l.Bd, L, D)
		for i := range h {
			h[i] += ffn[i] // residual
		}
		layerNorm(h, l.OutLNW, l.OutLNB, L, D, eps)
	}
	return h
}

// Embed returns the mean-pooled, L2-normalized sentence embedding for token ids.
func (b *BERT) Embed(ids []int32) []float32 {
	D := b.cfg.Hidden
	h := b.hiddenStates(ids, nil)
	v := poolOne(h, len(ids), D, poolMean) // mean over the L tokens
	return l2norm(v)
}

// gelu applies the exact (erf-based) GELU in place: x·Φ(x) = 0.5x(1+erf(x/√2)).
// transformers' "gelu" activation is the erf form, not the tanh approximation.
func gelu(x []float32) {
	const invSqrt2 = 0.7071067811865476
	for i, v := range x {
		x[i] = float32(0.5 * float64(v) * (1 + math.Erf(float64(v)*invSqrt2)))
	}
}

func l2norm(v []float32) []float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	n := float32(math.Sqrt(s))
	if n == 0 {
		return v
	}
	for i := range v {
		v[i] /= n
	}
	return v
}
