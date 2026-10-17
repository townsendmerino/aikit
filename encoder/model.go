package encoder

import (
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/embed"
)

// Encoder is the surface used by NeuralReranker. Both *Model (f32) and
// *ModelQ8 (int8) implement it; the reranker doesn't care which it
// holds. Lets the CLI / MCP layer pick a precision at startup without
// the rest of the code knowing.
type Encoder interface {
	Encode(text string, isQuery bool) ([]float32, error)
	EncodeBatch(texts []string, isQueries []bool, concurrency int) ([][]float32, error)
	HiddenDim() int
}

// Model is the loaded CodeRankEmbed reranker: weights + tokenizer.
// Goroutine-safe for concurrent Encode calls (all internal state is
// immutable after Load; per-call buffers are stack/heap-local).
type Model struct {
	weights *Weights
	tok     *embed.Tokenizer

	// maxSeqLength caps the wrapped sequence (incl. [CLS]+[SEP]) the
	// forward pass sees. Defaults to DefaultMaxSeqLength (512). Plan §5.
	maxSeqLength int
}

// Load reads a CodeRankEmbed snapshot from dir (config.json,
// model.safetensors, tokenizer.json — the standard HF layout). Cf.
// embed.LoadFromFS for the analogous Model2Vec loader.
//
// Load takes the mmap loader (LoadWeights), so the checkpoint stays in the OS
// page cache instead of a full Go-heap copy that GC scans — the regression M8 was
// written to fix — and Close actually releases the mapping. LoadFromFS keeps the
// fs.FS/embed.FS route (fs.ReadFile, heap) for callers without a real directory.
// Audit #2: previously Load delegated to LoadFromFS, so no caller used the mmap
// path and Close on a Load-built model was a no-op.
// lengthSortedOrder returns indices 0..n-1 ordered by ascending lens, stably.
// Sorting the DISPATCH ORDER rather than the texts keeps each bucket's members
// close in length, so a worker's padded batch wastes less on short rows — the
// number of dispatchable units is unchanged, only their contents. Stable so
// equal-length texts keep caller order, which keeps EncodeBatch deterministic.
func lengthSortedOrder(lens []int, n int) []int {
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return lens[order[a]] < lens[order[b]] })
	return order
}

func Load(dir string) (*Model, error) {
	w, err := LoadWeights(dir)
	if err != nil {
		return nil, err
	}
	tok, err := embed.LoadTokenizer(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("encoder: load tokenizer: %w", err)
	}
	return &Model{weights: w, tok: tok, maxSeqLength: DefaultMaxSeqLength}, nil
}

// LoadFromFS reads from fsys rooted at dir. Same file shape as Load.
func LoadFromFS(fsys fs.FS, dir string) (*Model, error) {
	w, err := LoadWeightsFromFS(fsys, dir)
	if err != nil {
		return nil, err
	}
	tok, err := embed.LoadTokenizerFromFS(fsys, path.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("encoder: load tokenizer: %w", err)
	}
	return &Model{weights: w, tok: tok, maxSeqLength: DefaultMaxSeqLength}, nil
}

// SetMaxSeqLength overrides the per-call truncation cap. Useful for
// unit tests (small L = much faster) and for callers who want to trade
// recall on long inputs for latency. 0 or negative resets to default.
//
// Call it before sharing the Model across goroutines: it mutates internal state
// without synchronization, so a concurrent Encode would race. The Model is
// otherwise immutable-after-Load and safe for concurrent Encode.
func (m *Model) SetMaxSeqLength(n int) {
	if n <= 0 {
		m.maxSeqLength = DefaultMaxSeqLength
		return
	}
	m.maxSeqLength = n
}

// HiddenDim is the output embedding dimension (768 for CodeRankEmbed).
func (m *Model) HiddenDim() int { return m.weights.Cfg.HiddenDim }

// Close releases the mmap-backed weights (the ~547 MB checkpoint lives in the OS
// page cache until then). Call it once, after the last Encode, when a
// long-running server swaps models — otherwise the mapping stays until the process
// exits (Close is the only release; see embed.SafetensorsFile.Close). Idempotent; a
// heap-loaded model has nothing to release.
func (m *Model) Close() error {
	if m.weights == nil || m.weights.st == nil {
		return nil
	}
	return m.weights.st.Close()
}

// Encode tokenizes `text` (prepending the mandatory query prefix iff
// isQuery is true), wraps with [CLS]/[SEP], runs the transformer
// forward pass, and returns the raw (UN-normalized) CLS hidden state.
//
// The caller is responsible for L2-normalizing if the consumer is
// cosine; ranking is invariant to it but the parity goldens compare
// raw vectors so this is the natural Model output.
//
// Goroutine-safe. The Weights and Tokenizer are immutable; every
// per-call buffer (token ids, hidden states, attention scores, …) is
// allocated fresh inside the call.
func (m *Model) Encode(text string, isQuery bool) ([]float32, error) {
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
		return nil, err
	}
	return m.weights.forward(ids), nil
}

// EncodeBatch runs N forward passes in parallel across concurrency
// workers, returning one CLS vector per (text, isQuery) input. Two
// layers of parallelism (M3 + M7):
//
//   - WORKERS: input is statically split into `concurrency` chunks
//     (default NumCPU). Workers run independently — no shared state
//     other than input/output slices.
//   - BATCHED FORWARD: each worker calls forwardBatch on its chunk,
//     processing all its candidates as one padded batch through the
//     12 layers' big matmuls (Wqkv, OutProj, fc11, fc12, fc2). The
//     batched matmuls amortize per-call overhead and keep hidden
//     states in cache across layers — the M7 win.
//
// Static partitioning (vs M3's job-channel) is fine here because each
// worker now does a coalesced batch, not per-input pop-from-queue
// work. Load imbalance can show on adversarial inputs (worker A's
// chunk is all 500-token; worker B's all 5-token), but rerank batches
// are typically uniform-length per M3's measurements.
//
// On error, returns the first error and a nil result slice (no
// partial results). `concurrency` ≤ 0 means runtime.NumCPU().
func (m *Model) EncodeBatch(texts []string, isQueries []bool, concurrency int) ([][]float32, error) {
	return encodeBatch(m.tok, m.maxSeqLength, m.weights.forwardBatch, texts, isQueries, concurrency)
}

// encodeBatch is the shared body of Model.EncodeBatch and ModelQ8.EncodeBatch,
// which differed only in which forwardBatch they called (audit #19: 60 duplicated
// lines, so a fix to the tokenize-error path had to be made twice). fwd is the
// model's batched forward. It fans out over `concurrency` workers, each tokenizing
// its chunk (query/doc prefix per isQueries), running one batched forward, and
// scattering the vectors back; the first tokenize error wins and is returned.
// batchTokenBudget caps B·Lmax for one forward — the quantity the padded batch
// kernel's cost and its scratch arena both scale with. It adapts the batch size
// to the sequence length instead of fixing a document count: 8 sequences at
// Lmax=512, 64 at Lmax=64.
const batchTokenBudget = 4096

// maxBatchSeqs bounds B independently, so a corpus of very short texts does not
// build a 1000-wide batch whose per-sequence bookkeeping outweighs the packing.
const maxBatchSeqs = 64

// bucketByLength groups sequence indices into batches of similar length under
// the token budget.
//
// This is perf-campaign item 14. The previous partition was by INPUT ORDER —
// worker w took a contiguous index range — so a single long document inflated
// Lmax for its whole chunk. Attention skips pad positions, but the projections
// and MLP do not: they run over M = B·Lmax, and attention is only ~8% of a layer
// at L=512/D=768, so ~92% of the FLOPs are exposed to padding. For 50 documents
// uniform on [20,512] that is 48% of all linear-layer work computed on pad.
//
// order must be indices sorted by ascending length; because of that, the longest
// member of a growing bucket is always the one just added, so the budget check
// is a single multiply.
//
// maxPerBucket is the third bound and it is not optional: the token budget alone
// would put a small corpus of short texts into ONE bucket, leaving every worker
// but one idle. Capping at ceil(n/concurrency) reproduces exactly as many
// dispatchable units as the old index-ordered partition produced, so this item
// changes WHICH sequences share a forward without changing how many forwards run
// concurrently. (The existing TestEncodeBatch_speedup caught this: 8 short texts
// collapsed to a single bucket and the measured speedup fell to 1.13×.)
func bucketByLength(order []int, lens []int, maxPerBucket int) [][]int {
	var buckets [][]int
	cur := []int{}
	curMax := 0
	for _, idx := range order {
		l := max(lens[idx], 1)
		nextMax := max(curMax, l)
		if len(cur) > 0 && ((len(cur)+1)*nextMax > batchTokenBudget ||
			len(cur) >= maxBatchSeqs || len(cur) >= maxPerBucket) {
			buckets = append(buckets, cur)
			cur = []int{}
			nextMax = l
		}
		cur = append(cur, idx)
		curMax = nextMax
	}
	if len(cur) > 0 {
		buckets = append(buckets, cur)
	}
	return buckets
}

func encodeBatch(tok *embed.Tokenizer, maxSeq int, fwd func([][]int32) [][]float32, texts []string, isQueries []bool, concurrency int) ([][]float32, error) {
	if len(texts) != len(isQueries) {
		return nil, fmt.Errorf("encoder: EncodeBatch len(texts)=%d != len(isQueries)=%d", len(texts), len(isQueries))
	}
	if len(texts) == 0 {
		return nil, nil
	}
	if concurrency <= 0 {
		concurrency = runtime.NumCPU()
	}
	if concurrency > len(texts) {
		concurrency = len(texts)
	}

	// Tokenize everything first — bucketing needs the lengths, and this also
	// takes tokenization off the forward workers' critical path.
	all := make([][]int32, len(texts))
	lens := make([]int, len(texts))
	var firstErr error
	var errOnce sync.Once
	var twg sync.WaitGroup
	tokChunk := (len(texts) + concurrency - 1) / concurrency
	for w := 0; w < concurrency; w++ {
		start := w * tokChunk
		end := min(start+tokChunk, len(texts))
		if start >= end {
			break
		}
		twg.Add(1)
		go func(start, end int) {
			defer twg.Done()
			for i := start; i < end; i++ {
				var (
					ids []int32
					err error
				)
				if isQueries[i] {
					ids, err = EncodeQuery(tok, texts[i], maxSeq)
				} else {
					ids, err = EncodeDoc(tok, texts[i], maxSeq)
				}
				if err != nil {
					errOnce.Do(func() { firstErr = err })
					return
				}
				all[i], lens[i] = ids, len(ids)
			}
		}(start, end)
	}
	twg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}

	order := lengthSortedOrder(lens, len(texts))
	// Same number of dispatchable units the old partition produced, so worker
	// occupancy is unchanged; only their CONTENTS are length-sorted.
	perBucket := (len(texts) + concurrency - 1) / concurrency
	buckets := bucketByLength(order, lens, perBucket)

	out := make([][]float32, len(texts))
	var next atomic.Int64
	var wg sync.WaitGroup
	for range min(concurrency, len(buckets)) {
		wg.Go(func() {
			// Buckets vary in cost, so workers pull rather than take a static
			// slice — the old index-ordered partition imbalanced them whenever
			// document lengths did.
			for {
				b := int(next.Add(1)) - 1
				if b >= len(buckets) {
					return
				}
				idx := buckets[b]
				idsList := make([][]int32, len(idx))
				for i, gi := range idx {
					idsList[i] = all[gi]
				}
				vecs := fwd(idsList)
				for i, gi := range idx {
					out[gi] = vecs[i] // scatter back to the caller's order
				}
			}
		})
	}
	wg.Wait()
	return out, nil
}

// ── Int8 (M8) model ─────────────────────────────────────────────────

// ModelQ8 is the int8-quantized sibling of Model. Same API surface
// (the Encoder interface is the common contract) but the per-layer
// big linear projections store int8 + per-row scales instead of
// float32 weights, cutting weight bytes ~4× (137M params × 4B = 547MB
// → ~140 MB resident). Forward pass routes through matmulBTQ8 for
// those layers.
//
// Accuracy cost: end-to-end cosine ≥ 0.97 vs the f32 Model (pinned
// by TestModelQ8_cosineMatchesF32). For NDCG: M0's measured CoIR
// lift was +0.165 at β=1; the int8 model is expected to reproduce
// that to within bench noise (~±0.01) because per-matmul ~0.8%
// relative error attenuates by the time it's squashed through 12
// LayerNorms.
type ModelQ8 struct {
	weights      *WeightsQ8
	tok          *embed.Tokenizer
	maxSeqLength int
}

// LoadQ8 reads + quantizes the rerank model at dir. Same disk layout
// as Load (config.json + tokenizer.json + model.safetensors).
func LoadQ8(dir string) (*ModelQ8, error) {
	w, err := LoadWeightsQ8(dir)
	if err != nil {
		return nil, err
	}
	tok, err := embed.LoadTokenizerFromFS(os.DirFS(dir), "tokenizer.json")
	if err != nil {
		_ = w.Close() //nolint:errcheck // a failed load: the tokenizer error is the one to report
		return nil, fmt.Errorf("encoder: load tokenizer: %w", err)
	}
	return &ModelQ8{weights: w, tok: tok, maxSeqLength: DefaultMaxSeqLength}, nil
}

// SetMaxSeqLength mirrors Model.SetMaxSeqLength.
func (m *ModelQ8) SetMaxSeqLength(n int) {
	if n <= 0 {
		m.maxSeqLength = DefaultMaxSeqLength
		return
	}
	m.maxSeqLength = n
}

// HiddenDim implements Encoder.
func (m *ModelQ8) HiddenDim() int { return m.weights.Cfg.HiddenDim }

// Encode implements Encoder.
func (m *ModelQ8) Encode(text string, isQuery bool) ([]float32, error) {
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
		return nil, err
	}
	return m.weights.forward(ids), nil
}

// EncodeBatch implements Encoder. Same static-partition + batched-
// forward-per-worker shape as Model.EncodeBatch; routes through the
// q8 batched forward instead.
func (m *ModelQ8) EncodeBatch(texts []string, isQueries []bool, concurrency int) ([][]float32, error) {
	return encodeBatch(m.tok, m.maxSeqLength, m.weights.forwardBatch, texts, isQueries, concurrency)
}

// Close releases the checkpoint mapping the model's word embeddings are read
// from (see WeightsQ8); the model must not be used afterwards. Idempotent. Like
// Model.Close it satisfies io.Closer, so a caller holding an encoder.Encoder can
// release it via `if c, ok := enc.(io.Closer); ok { c.Close() }` without Close
// being on the (Hard-tier, frozen) Encoder interface (audit #19). Until v1.71.0
// it was a no-op: the model then held no mapping.
func (m *ModelQ8) Close() error { return m.weights.Close() }
