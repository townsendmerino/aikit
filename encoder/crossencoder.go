package encoder

import (
	"fmt"
	"math"
)

// crossencoder.go — a BERT cross-encoder reranker (BertForSequenceClassification),
// e.g. cross-encoder/ms-marco-MiniLM-L-6-v2 (hugot's headline CrossEncoders model).
// Unlike a bi-encoder, it scores a (query, document) PAIR jointly: the trunk runs
// over [CLS] query [SEP] document [SEP] (token types 0/1), then the [CLS] hidden
// state goes through the BERT pooler (dense + tanh) and a linear classification head
// to a single relevance logit. Reuses the §2.2 BERT trunk + WordPiece tokenizer; the
// only new weights are the pooler and the classifier.

// CrossEncoder is a loaded BERT cross-encoder reranker.
type CrossEncoder struct {
	bert        *BERT
	poolerW     []float32 // bert.pooler.dense.weight [hidden, hidden]
	poolerB     []float32 // [hidden]
	classifierW []float32 // classifier.weight [labels, hidden]
	classifierB []float32 // classifier.bias [labels]
	labels      int
}

// LoadCrossEncoder loads a BertForSequenceClassification cross-encoder (config.json +
// model.safetensors) from dir: the BERT trunk (via LoadBERT) plus the pooler and the
// classification head. The number of labels is read from the classifier shape (1 for
// a ms-marco-style relevance reranker).
func LoadCrossEncoder(dir string) (*CrossEncoder, error) {
	b, err := LoadBERT(dir)
	if err != nil {
		return nil, err
	}
	D := b.cfg.Hidden
	ce := &CrossEncoder{bert: b}

	ct, err := b.st.Tensor("classifier.weight")
	if err != nil {
		_ = b.st.Close()
		return nil, fmt.Errorf("encoder: cross-encoder classifier.weight: %w", err)
	}
	if len(ct.Shape) != 2 || ct.Shape[1] != D {
		_ = b.st.Close()
		return nil, fmt.Errorf("encoder: cross-encoder classifier.weight shape %v (want [labels,%d])", ct.Shape, D)
	}
	ce.labels = ct.Shape[0]

	var e error
	get := func(name string, want ...int) []float32 {
		if e != nil {
			return nil
		}
		var v []float32
		v, e = loadF32(b.st, name, want)
		return v
	}
	ce.poolerW = get("bert.pooler.dense.weight", D, D)
	ce.poolerB = get("bert.pooler.dense.bias", D)
	ce.classifierW = get("classifier.weight", ce.labels, D)
	ce.classifierB = get("classifier.bias", ce.labels)
	if e != nil {
		_ = b.st.Close()
		return nil, fmt.Errorf("encoder: cross-encoder head: %w", e)
	}
	return ce, nil
}

// Score returns the relevance logit for a (query, document) pair — higher is more
// relevant. Rank a candidate list by descending Score to rerank. (For a model with
// more than one label, this is label 0; use ScoreAll for the rest.)
func (ce *CrossEncoder) Score(query, doc string) (float32, error) {
	all, err := ce.ScoreAll(query, doc)
	if err != nil {
		return 0, err
	}
	return all[0], nil
}

// ScoreAll returns every classification logit for the pair (length = num labels).
func (ce *CrossEncoder) ScoreAll(query, doc string) ([]float32, error) {
	ids, segs := ce.pairIDs(query, doc)
	return ce.scoreIDs(ids, segs), nil
}

// pairIDs builds [CLS] query [SEP] document [SEP] with token-type segments 0/1,
// right-truncating the document (then the query) to the model's max sequence length.
func (ce *CrossEncoder) pairIDs(query, doc string) (ids, segs []int32) {
	cls, _ := ce.bert.tok.SpecialID("[CLS]")
	sep, _ := ce.bert.tok.SpecialID("[SEP]")
	q := ce.bert.tok.Encode(query)
	d := ce.bert.tok.Encode(doc)

	avail := max(ce.bert.maxSeq-3, 0) // room for [CLS] + 2×[SEP]
	if len(q) > avail {
		q = q[:avail]
	}
	if len(q)+len(d) > avail {
		d = d[:avail-len(q)]
	}

	ids = append(ids, cls)
	ids = append(ids, q...)
	ids = append(ids, sep)
	seg1 := len(ids) // document + trailing [SEP] are segment 1
	ids = append(ids, d...)
	ids = append(ids, sep)
	segs = make([]int32, len(ids))
	for i := seg1; i < len(ids); i++ {
		segs[i] = 1
	}
	return ids, segs
}

// scoreIDs runs the trunk on a pre-assembled pair and applies the pooler +
// classification head: classifier(tanh(pooler(CLS))).
func (ce *CrossEncoder) scoreIDs(ids, segs []int32) []float32 {
	D := ce.bert.cfg.Hidden
	h := ce.bert.hiddenStates(ids, segs)
	cls := h[0:D] // the [CLS] token's final hidden state

	pooled := matmulBT(cls, ce.poolerW, 1, D, D) // CLS · poolerWᵀ
	addBias(pooled, ce.poolerB, 1, D)
	for i, v := range pooled {
		pooled[i] = float32(math.Tanh(float64(v)))
	}
	out := matmulBT(pooled, ce.classifierW, 1, D, ce.labels) // pooled · classifierWᵀ
	addBias(out, ce.classifierB, 1, ce.labels)
	return out
}
