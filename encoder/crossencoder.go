package encoder

import (
	"fmt"
	"github.com/townsendmerino/aikit/linalg/elem"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
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
	if ce.labels < 1 {
		_ = b.st.Close()
		return nil, fmt.Errorf("encoder: cross-encoder must have ≥1 label, got %d", ce.labels)
	}
	// pairIDs frames [CLS] q [SEP] d [SEP]; without those specials the tokenizer
	// silently yields id 0 ([PAD]) for them and the pair is malformed. Require
	// them at load rather than mis-score every pair.
	if _, ok := b.tok.SpecialID("[CLS]"); !ok {
		_ = b.st.Close()
		return nil, fmt.Errorf("encoder: cross-encoder tokenizer has no [CLS] token")
	}
	if _, ok := b.tok.SpecialID("[SEP]"); !ok {
		_ = b.st.Close()
		return nil, fmt.Errorf("encoder: cross-encoder tokenizer has no [SEP] token")
	}

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

// Close releases the underlying BERT's mmap-backed weights. Idempotent.
func (ce *CrossEncoder) Close() error {
	if ce.bert == nil {
		return nil
	}
	return ce.bert.Close()
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

// ScoreBatch scores one query against many documents, returning label 0's logit
// per document in the caller's order. concurrency <= 0 means NumCPU.
//
// This is perf-campaign item 28. That item also names "re-tokenizes the query
// per pair" as a cost; measured, it is 0.066% of a 50-document rerank (pairIDs
// totals 40 ms of 60.33 s and the query line does not register at all), so the
// query is hoisted below for tidiness, not for speed. The win is the batch API
// itself: a caller looping over Score runs one forward at a time, each
// parallelizing internally, and document-level parallelism beats intra-op — the
// same result EncodeBatch has.
//
// Documents are dispatched LONGEST FIRST. Cost is roughly linear in pair length,
// and BERT has no padded batch kernel, so unlike EncodeBatch there is no padding
// to bucket away and the only scheduling question left is makespan.
// Longest-processing-time-first is the standard answer and costs one sort.
func (ce *CrossEncoder) ScoreBatch(query string, docs []string, concurrency int) ([]float32, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	if concurrency <= 0 {
		concurrency = runtime.NumCPU()
	}
	concurrency = min(concurrency, len(docs))

	// Tokenize the query ONCE. pairIDsFrom still re-derives the trim per pair,
	// because longest_first depends on that document's length.
	qIDs := ce.bert.tok.Encode(query)

	type pair struct{ ids, segs []int32 }
	pairs := make([]pair, len(docs))
	for i, d := range docs {
		ids, segs := ce.pairIDsFrom(qIDs, d)
		pairs[i] = pair{ids: ids, segs: segs}
	}
	order := make([]int, len(pairs))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		return len(pairs[order[a]].ids) > len(pairs[order[b]].ids)
	})

	out := make([]float32, len(docs))
	var next atomic.Int64
	var wg sync.WaitGroup
	for range concurrency {
		wg.Go(func() {
			for {
				k := int(next.Add(1)) - 1
				if k >= len(order) {
					return
				}
				i := order[k]
				out[i] = ce.scoreIDs(pairs[i].ids, pairs[i].segs)[0]
			}
		})
	}
	wg.Wait()
	return out, nil
}

// pairIDs builds [CLS] query [SEP] document [SEP] with token-type segments 0/1,
// right-truncating the document (then the query) to the model's max sequence length.
func (ce *CrossEncoder) pairIDs(query, doc string) (ids, segs []int32) {
	return ce.pairIDsFrom(ce.bert.tok.Encode(query), doc)
}

// pairIDsFrom is pairIDs with the query already tokenized, so a batch shares one
// encode. It is the single implementation; pairIDs is the one-shot wrapper.
func (ce *CrossEncoder) pairIDsFrom(qTok []int32, doc string) (ids, segs []int32) {
	cls, _ := ce.bert.tok.SpecialID("[CLS]")
	sep, _ := ce.bert.tok.SpecialID("[SEP]")
	// Copy before trimming: every pair in a batch shares qTok, and the trim
	// below reslices it.
	q := append([]int32(nil), qTok...)
	d := ce.bert.tok.Encode(doc)

	avail := max(ce.bert.maxSeq-3, 0) // room for [CLS] + 2×[SEP]
	// longest_first (the HF CrossEncoder default): trim the currently-longer
	// sequence one token at a time until the pair fits, ties trimming the doc.
	// The old scheme gave the query the whole budget first, so a query ≥ avail
	// starved the document to zero tokens and the score became
	// document-independent.
	for len(q)+len(d) > avail {
		if len(q) > len(d) {
			q = q[:len(q)-1]
		} else {
			d = d[:len(d)-1]
		}
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
	// The pooler reads row 0 and nothing else, so the trunk's final layer is run
	// at M=1 (lens doc §4.1). This is the rerank path, where W2R puts 99.8%+ of a
	// query, and the ONE call site that qualifies unconditionally — a
	// cross-encoder is CLS-pooled by construction.
	cls := ce.bert.clsHiddenState(ids, segs) // the [CLS] token's final hidden state

	pooled := matmulBT(cls, ce.poolerW, 1, D, D) // CLS · poolerWᵀ
	addBias(pooled, ce.poolerB, 1, D)
	// linalg's float32 tanh, for consistency with every other activation in the
	// kit rather than for speed: this is the pooler, [1,D] = 384 elements per
	// Score against a ~10 ms forward, so the ~15 µs it costs is below anything
	// the benchmarks can resolve. Same accuracy contract as the rest (≤2 ULP).
	elem.TanhInto(pooled, pooled)
	out := matmulBT(pooled, ce.classifierW, 1, D, ce.labels) // pooled · classifierWᵀ
	addBias(out, ce.classifierB, 1, ce.labels)
	return out
}
