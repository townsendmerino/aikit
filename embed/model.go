package embed

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
)

// StaticModel is a Model2Vec static-embedding model. Goroutine-safe for
// concurrent Encode calls after Load returns — all internal state is
// immutable.
type StaticModel struct {
	tokenizer  *Tokenizer
	embeddings []float32 // [vocab × dim] flat, row-major
	mapping    []int64   // [vocab]
	weights    []float64 // [vocab]
	vocab      int
	dim        int
	normalize  bool

	// Keep the file alive so the unsafe-slice tensor data stays valid.
	st *SafetensorsFile
}

type modelConfig struct {
	Normalize              bool   `json:"normalize"`
	EmbeddingDType         string `json:"embedding_dtype"`
	VocabularyQuantization int    `json:"vocabulary_quantization"`
}

// LoadFromFS reads a Model2Vec model from fsys, rooted at dir. fsys must contain
//
//	<dir>/tokenizer.json
//	<dir>/config.json
//	<dir>/model.safetensors
//
// (the standard HF layout for Model2Vec models — typically a Hugging Face
// snapshot for minishlab/potion-code-16M or another compatible model).
//
// Typical call shapes:
//
//	LoadFromFS(os.DirFS("/path/to/model"), ".")     // load from a directory
//	LoadFromFS(embedFS, "model")                    // load from //go:embed model/*
//	LoadFromFS(mapFS, ".")                          // load from a fstest.MapFS
//
// Paths inside fsys follow the fs.FS slash convention (use the path
// package, not path/filepath). LoadFromFS is the canonical entry point;
// Load is a thin deprecated wrapper.
// parseModelConfig decodes and validates a model's config.json. Only the PARSE
// half is shared: the two callers reach the bytes differently (fs.ReadFile over
// an fs.FS vs os.ReadFile over a directory), and folding that in would mean
// threading a filesystem abstraction through purely to satisfy a helper. The
// error text is carried over unchanged from the two inline copies, so callers
// see exactly the same messages they did before.
func parseModelConfig(cfgBytes []byte) (modelConfig, error) {
	var cfg modelConfig
	if err := json.Unmarshal(cfgBytes, &cfg); err != nil {
		return cfg, fmt.Errorf("parse config.json: %w", err)
	}
	if cfg.EmbeddingDType != "" && cfg.EmbeddingDType != "float32" {
		return cfg, fmt.Errorf("unsupported embedding_dtype %q (only float32 supported)", cfg.EmbeddingDType)
	}
	return cfg, nil
}

func LoadFromFS(fsys fs.FS, dir string) (*StaticModel, error) {
	join := func(name string) string { return path.Join(dir, name) }

	tok, err := LoadTokenizerFromFS(fsys, join("tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	cfgBytes, err := fs.ReadFile(fsys, join("config.json"))
	if err != nil {
		return nil, fmt.Errorf("read config.json: %w", err)
	}
	cfg, err := parseModelConfig(cfgBytes)
	if err != nil {
		return nil, err
	}

	st, err := OpenSafetensorsFromFS(fsys, join("model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("open safetensors: %w", err)
	}
	return buildStaticModel(tok, st, cfg)
}

// LoadMmap is Load for a real directory, memory-mapping model.safetensors
// instead of reading it onto the heap.
//
// WHY THIS EXISTS. OpenSafetensorsMmap has been in this package the whole time
// and no load path reached it: LoadFromFS goes through fs.ReadFile, which by
// definition cannot mmap, and Load is a one-line wrapper around
// LoadFromFS(os.DirFS(dir), ".") that throws away the one thing mmap needs — a
// real path. So every caller heap-read the whole checkpoint.
//
// WHAT IT BUYS, AND WHAT IT COSTS — both measured, because they point opposite
// ways and the choice is the caller's. On the embedded-corpus example (64 MB
// potion-code-16M, docs/internal/perf-amdahl-linux-amd64.md W3):
//
//	load stage    59.5 ms → 48.8 ms   peak 142.5 → 19.5 MiB   alloc 73.9 → 9.6 MB
//	cold start    84.8 ms → 99.2 ms   peak  75.8 → 13.0 MiB   alloc 82.4 → 18.1 MB
//
// Peak heap falls 5.8× end to end and allocation 4.6×, while time-to-first-result
// rises 17%. The load itself is faster — there is no 64 MB read — but mmap defers
// the page faults to whoever first touches the embedding table, which is the
// first Encode, and faulting in 64 MB costs more than reading it sequentially
// from a warm page cache. This is a FOOTPRINT option, not a speed one: take it
// under a memory cap, leave it for a short-lived CLI.
//
// The tensors already alias the file's bytes rather than copying them (see
// reinterpretLE), so the only thing standing between a caller and the page cache
// was where those bytes came from.
//
// LIFETIME. The returned model holds the mapping. It is not unmapped when the
// model becomes unreachable (the safetensors finalizer never runs for a file
// with tensors; see embed's finalizeMmaps), so a long-lived process that loads
// and drops models repeatedly keeps every mapping until exit. Do not retain a
// vector slice past the model itself.
//
// On non-unix targets mmap.MapReadOnly falls back to a heap read, so this is
// identical to Load there rather than unavailable.
func LoadMmap(dir string) (*StaticModel, error) {
	tok, err := LoadTokenizerFromFS(os.DirFS(dir), "tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	cfgBytes, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("read config.json: %w", err)
	}
	cfg, err := parseModelConfig(cfgBytes)
	if err != nil {
		return nil, err
	}
	st, err := OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("open safetensors: %w", err)
	}
	return buildStaticModel(tok, st, cfg)
}

// buildStaticModel is the shared tail of LoadFromFS and LoadMmap: everything
// after the three files are open, which differs between them only in how the
// safetensors bytes were obtained.
func buildStaticModel(tok *Tokenizer, st *SafetensorsFile, cfg modelConfig) (*StaticModel, error) {

	embT, err := st.Tensor("embeddings")
	if err != nil {
		return nil, fmt.Errorf("embeddings tensor: %w", err)
	}
	if len(embT.Shape) != 2 {
		return nil, fmt.Errorf("embeddings tensor: expected 2-D, got shape %v", embT.Shape)
	}
	vocab := embT.Shape[0]
	dim := embT.Shape[1]

	embData, err := embT.Float32s()
	if err != nil {
		return nil, err
	}
	if len(embData) != vocab*dim {
		return nil, fmt.Errorf("embeddings element count %d != vocab*dim (%d*%d)", len(embData), vocab, dim)
	}

	// mapping + weights are optional. The original potion format (vocabulary-
	// quantized, e.g. potion-code-16M) indexes embeddings through mapping[id] and
	// pools with per-token zipf weights[id]. The newer/standard Model2Vec format
	// (e.g. potion-retrieval-32M) bakes both in: embeddings are indexed directly by
	// token id and pooled with uniform weights (plain mean). Absence ⇒ nil ⇒ the
	// direct / uniform path in encodeIDs.
	var mapData []int64
	if mapT, e := st.Tensor("mapping"); e == nil {
		if len(mapT.Shape) != 1 || mapT.Shape[0] != vocab {
			return nil, fmt.Errorf("mapping tensor: expected shape [%d], got %v", vocab, mapT.Shape)
		}
		if mapData, err = mapT.Int64s(); err != nil {
			return nil, err
		}
	}
	var wData []float64
	if wT, e := st.Tensor("weights"); e == nil {
		if len(wT.Shape) != 1 || wT.Shape[0] != vocab {
			return nil, fmt.Errorf("weights tensor: expected shape [%d], got %v", vocab, wT.Shape)
		}
		if wData, err = wT.Float64s(); err != nil {
			return nil, err
		}
	}

	return &StaticModel{
		tokenizer:  tok,
		embeddings: embData,
		mapping:    mapData,
		weights:    wData,
		vocab:      vocab,
		dim:        dim,
		normalize:  cfg.Normalize,
		st:         st,
	}, nil
}

// Load reads a Model2Vec model from a directory containing tokenizer.json,
// config.json, and model.safetensors.
//
// Deprecated: use LoadFromFS(os.DirFS(modelDir), ".") instead. Load is kept
// as a thin wrapper for callers that haven't migrated yet (v0.6.0+).
func Load(modelDir string) (*StaticModel, error) {
	return LoadFromFS(os.DirFS(modelDir), ".")
}

// VocabSize reports the embedding-table vocabulary size.
func (m *StaticModel) VocabSize() int { return m.vocab }

// Dim reports the embedding dimension.
func (m *StaticModel) Dim() int { return m.dim }

// Tokenizer returns the underlying tokenizer (useful for parity tests).
func (m *StaticModel) Tokenizer() *Tokenizer { return m.tokenizer }

// Encode tokenizes and embeds a single string.
//
// Algorithm (verified by golden test against StaticModel.encode()):
//
//	ids = tokenize(text)
//	rows[i] = embeddings[mapping[ids[i]]]      // F32 row, dim values
//	w[i]    = weights[ids[i]]                  // F64 scalar
//	v       = Σ rows[i]·w[i]                   // accumulate in F64
//	v       = v / Σ w[i]                       // F64
//	if normalize: v /= ‖v‖₂                    // F64 sum-of-squares
//	return float32(v)                          // cast at the end
//
// Precision contract: every accumulator (weighted-sum, weight-sum,
// sum-of-squares for L2) is float64. Embeddings stay float32 in memory;
// individual values are widened to float64 only at the multiply-accumulate
// step. Float32 accumulation breaks parity with the Python reference on
// inputs of more than a few dozen tokens. See pool.go for the matching
// implementation and the rationale.
//
// Returns a zero vector for empty inputs and for degenerate cases that
// would otherwise produce NaN (all-UNK on a long word, all-zero weight sum).
func (m *StaticModel) Encode(text string) []float32 {
	ids := m.tokenizer.Encode(text)
	return m.encodeIDs(ids)
}

// EncodeBatch encodes texts concurrently and returns one vector per input, in
// input order. concurrency <= 0 means runtime.NumCPU(), matching
// encoder.Model.EncodeBatch's convention.
//
// This is the bulk path, and until now it did not exist: Encode was the whole
// public encode surface, so every caller — including both shipped examples —
// wrote a serial loop over a corpus. The transformer model got worker fan-out
// years earlier; the model whose own doc comment cites "a 378k chunk corpus" got
// none. StaticModel.Encode is 77.8% of an index run
// (docs/internal/perf-amdahl-linux-amd64.md), so this is the largest single item
// in the campaign, and it is not an optimization of the encode path at all —
// nothing prevented fan-out except that no package offered it.
//
// BIT-IDENTICAL to a serial loop, and the reason is structural rather than
// careful: StaticModel is immutable after load and Encode touches no shared
// mutable state, so a text's vector does not depend on what else is in the batch
// or on which worker takes it. TestEncodeBatch_matchesSerial asserts exact
// equality over the whole corpus, not a sample.
//
// Work is handed out one text at a time through an atomic counter rather than by
// splitting the input into contiguous ranges. Chunk lengths in a real corpus vary
// by more than an order of magnitude, so a contiguous split leaves workers idle
// on whichever range happened to be short; the counter costs one atomic add per
// text against ~156 µs of encoding, which is 0.006%.
//
// Scaling with concurrency is near-linear up to the physical core count, then
// bends: measured on apple-m1pro (6 P-cores + 2 E-cores, no SMT), a batch runs
// ~5.02× at concurrency=6 (84% efficient) and ~5.23× at concurrency=NumCPU=8 (65%
// efficient) — the two E-cores together buy only ~1.04× over the six P-cores. So
// concurrency=NumCPU is the fastest in absolute wall-clock and the right default,
// but a caller optimising throughput-per-watt should pass the P-core count instead.
// (This is a per-machine curve; see docs/internal/perf-amdahl-apple-m1pro.md §3.)
//
// There is deliberately no variant writing into a caller-supplied flat
// [n·dim]float32. The two allocations encodeIDs makes per text are 2 of ~365 —
// the tokenizer makes the rest — and the copy such a variant would save is the
// one ann.NewFlatI8 performs, which is better removed there than worked around
// here.
func (m *StaticModel) EncodeBatch(texts []string, concurrency int) [][]float32 {
	out := make([][]float32, len(texts))
	if len(texts) == 0 {
		return out
	}
	if concurrency <= 0 {
		concurrency = runtime.NumCPU()
	}
	concurrency = min(concurrency, len(texts))
	if concurrency <= 1 {
		for i, t := range texts {
			out[i] = m.Encode(t)
		}
		return out
	}

	var next atomic.Int64
	var wg sync.WaitGroup
	wg.Add(concurrency)
	for range concurrency {
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= len(texts) {
					return
				}
				out[i] = m.Encode(texts[i])
			}
		}()
	}
	wg.Wait()
	return out
}

// encodeIDs is the inner path used by Encode and by tests that supply raw IDs.
// It computes Σ embRow[id]·w[id] / Σ w[id] — the verified Model2Vec runtime mean
// pooling — accumulating directly into one float64 sum in a single pass over ids,
// with no per-call [][]float32 / []float64 staging (audit #18: this is the primary
// dense-retrieval path; the old staging cost ~1.1M slice allocations over a 378k
// chunk corpus and lost sequential access by pooling through a slice-of-slices).
//
// PRECISION CONTRACT (required for golden-test parity, unchanged from the previous
// weightedMeanPoolSafe): the accumulator sum[dim], wsum, and the L2 dot are
// float64; each embedding element (float32) is widened to float64 BEFORE the
// multiply-accumulate; the output is cast to float32 only at the end, matching
// numpy's astype(float64)…astype(float32) in pin_inference.py. Accumulating in
// float32 silently drifts cosine below the 1−1e-5 parity bar on longer inputs — do
// not "optimize" it away. Out-of-range ids contribute zero (no-op tokens); an
// all-pad / Σw==0 input returns a zero vector.
func (m *StaticModel) encodeIDs(ids []int32) []float32 {
	out := make([]float32, m.dim)
	if len(ids) == 0 {
		return out
	}
	sum := make([]float64, m.dim)
	var wsum float64
	for _, id := range ids {
		if id < 0 || int(id) >= m.vocab {
			continue // out-of-range id → zero contribution
		}
		embRow := int64(id) // standard format: token id indexes embeddings directly
		if m.mapping != nil {
			embRow = m.mapping[id] // quantized format: indirect through mapping[]
		}
		if embRow < 0 || int(embRow) >= m.vocab {
			continue
		}
		row := m.embeddings[int(embRow)*m.dim : int(embRow)*m.dim+m.dim]
		ww := 1.0 // uniform (plain mean) when no explicit weights
		if m.weights != nil {
			ww = m.weights[id]
		}
		for j := range m.dim {
			sum[j] += float64(row[j]) * ww
		}
		wsum += ww
	}
	if wsum == 0 {
		return out
	}
	for j := range m.dim {
		out[j] = float32(sum[j] / wsum)
	}
	if m.normalize {
		return L2Normalize(out)
	}
	return out
}
