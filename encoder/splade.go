package encoder

import (
	"fmt"
	"math"

	"github.com/townsendmerino/aikit/sparse"
)

// splade.go — SPLADE in-process query/document expansion (roadmap §2.3). A SPLADE
// model is a BERT encoder (LoadBERT) plus a masked-LM head: the expansion projects
// each token's hidden state to vocabulary logits, applies log(1+ReLU), and max-pools
// over the sequence into a sparse vector over the vocab. The result drops straight
// into the sparse package's inverted index — closing the loop so learned-sparse
// retrieval runs end-to-end in-process, no Python at query time.

// SPLADE is a loaded SPLADE model: a BERT encoder plus the BertForMaskedLM head.
type SPLADE struct {
	bert       *BERT
	transformW []float32 // cls.predictions.transform.dense.weight [hidden, hidden]
	transformB []float32 // [hidden]
	transLNW   []float32 // cls.predictions.transform.LayerNorm.weight [hidden]
	transLNB   []float32
	decoderW   []float32 // cls.predictions.decoder.weight [vocab, hidden] (tied to word emb)
	decoderB   []float32 // cls.predictions.bias [vocab]
	vocab      int
}

// LoadSPLADE loads a BertForMaskedLM SPLADE model (config.json + model.safetensors)
// from dir: the BERT encoder (via LoadBERT) plus the masked-LM head.
func LoadSPLADE(dir string) (*SPLADE, error) {
	b, err := LoadBERT(dir)
	if err != nil {
		return nil, err
	}
	D, V := b.cfg.Hidden, b.cfg.VocabSize
	s := &SPLADE{bert: b, vocab: V}

	var e error
	get := func(name string, want ...int) []float32 {
		if e != nil {
			return nil
		}
		var v []float32
		v, e = loadF32(b.st, name, want)
		return v
	}
	s.transformW = get("cls.predictions.transform.dense.weight", D, D)
	s.transformB = get("cls.predictions.transform.dense.bias", D)
	s.transLNW = get("cls.predictions.transform.LayerNorm.weight", D)
	s.transLNB = get("cls.predictions.transform.LayerNorm.bias", D)
	s.decoderB = get("cls.predictions.bias", V)
	if e != nil {
		_ = b.st.Close()
		return nil, fmt.Errorf("encoder: SPLADE MLM head: %w", e)
	}
	// The decoder is usually weight-tied to the word embeddings; load the tensor if
	// the checkpoint stores it, otherwise reuse the already-loaded embeddings.
	if dw, derr := loadF32(b.st, "cls.predictions.decoder.weight", []int{V, D}); derr == nil {
		s.decoderW = dw
	} else {
		s.decoderW = b.wordEmb
	}
	return s, nil
}

// Expand runs the SPLADE expansion for text and returns the sparse term-weight
// vector over the model vocabulary (only positive weights — the natural SPLADE
// sparsity). Feed it to sparse.New (documents) or sparse.Index.Query (queries).
func (s *SPLADE) Expand(text string) (sparse.SparseVec, error) {
	ids, err := s.bert.tok.EncodeWithSpecials(text, s.bert.maxSeq)
	if err != nil {
		return sparse.SparseVec{}, err
	}
	return s.expandIDs(ids), nil
}

func (s *SPLADE) expandIDs(ids []int32) sparse.SparseVec {
	D, V, L := s.bert.cfg.Hidden, s.vocab, len(ids)
	h := s.bert.hiddenStates(ids) // [L, D]

	// MLM transform head: t = LayerNorm(gelu(h·Wᵀ + b)).
	t := matmulBT(h, s.transformW, L, D, D)
	addBias(t, s.transformB, L, D)
	gelu(t)
	layerNorm(t, s.transLNW, s.transLNB, L, D, s.bert.cfg.LNEps)

	// Vocabulary logits = t · decoderWᵀ + decoderB → [L, V].
	logits := matmulBT(t, s.decoderW, L, D, V)
	addBias(logits, s.decoderB, L, V)

	// SPLADE pooling: max over tokens of log(1 + relu(logit)).
	pooled := make([]float32, V)
	for i := 0; i < L; i++ {
		row := logits[i*V : (i+1)*V]
		for v, x := range row {
			if x > 0 {
				if w := float32(math.Log1p(float64(x))); w > pooled[v] {
					pooled[v] = w
				}
			}
		}
	}
	var out sparse.SparseVec
	for v, w := range pooled {
		if w > 0 {
			out.Terms = append(out.Terms, uint32(v))
			out.Weights = append(out.Weights, w)
		}
	}
	return out
}
