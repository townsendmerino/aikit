// Package encoder loads and (in subsequent commits) runs the
// nomic-ai/CodeRankEmbed neural reranker as a pure-Go forward pass.
//
// This file implements Milestone 1 of ken's rerank plan:
// config + weight loader with strict shape validation against the
// dumped checkpoint schema. Forward pass arrives in M2. The plan lives at
// https://github.com/townsendmerino/ken/blob/main/docs/internal/results/ken-rerank-plan.md.
//
// The package is a sibling of internal/embed (Model2Vec, the first-stage
// retriever); the two share tokenize.go + safetensors.go but use
// different inference algorithms and load different artifacts, so they
// stay separate per plan §3.
package encoder

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
)

// DefaultMaxSeqLength caps query+candidate token length per plan §5.
// CodeRankEmbed's tokenizer_config.max_length is 512; the model itself
// supports up to n_positions=8192 via RoPE, but rerank candidates are
// chunk-sized and 512 keeps latency bounded. Truncation is right-side
// (tokenizer_config truncation_side=right), preserving the [CLS] prefix.
const DefaultMaxSeqLength = 512

// Config captures the architecture constants from CodeRankEmbed's
// config.json that the forward pass depends on. Loaded from the
// checkpoint rather than hardcoded so a drop-in compatible checkpoint
// can override; ValidateAssumptions then fails loudly on any
// dimension/feature this loader/forward-pass doesn't implement.
type Config struct {
	VocabSize           int     `json:"vocab_size"`
	HiddenDim           int     `json:"n_embd"`
	NumLayers           int     `json:"n_layer"`
	NumHeads            int     `json:"n_head"`
	IntermediateDim     int     `json:"n_inner"`
	MaxPositions        int     `json:"n_positions"`
	MaxTrainedPositions int     `json:"max_trained_positions"`
	TypeVocabSize       int     `json:"type_vocab_size"`
	RoPEBase            float64 `json:"rotary_emb_base"`
	RoPEFraction        float64 `json:"rotary_emb_fraction"`
	RoPEInterleaved     bool    `json:"rotary_emb_interleaved"`
	LayerNormEpsilon    float64 `json:"layer_norm_epsilon"`
	ActivationFunction  string  `json:"activation_function"`
	Prenorm             bool    `json:"prenorm"`
	UseRMSNorm          bool    `json:"use_rms_norm"`
	QKVProjBias         bool    `json:"qkv_proj_bias"`
	MLPFc1Bias          bool    `json:"mlp_fc1_bias"`
	MLPFc2Bias          bool    `json:"mlp_fc2_bias"`
	ScaleAttnWeights    bool    `json:"scale_attn_weights"`
	Causal              bool    `json:"causal"`
	ParallelBlock       bool    `json:"parallel_block"`

	// Mixture-of-experts (nomic-embed-text-v2-moe). MoEEveryNLayers > 0 turns on
	// the MoE FFN for layers where i%MoEEveryNLayers == 1 (the reference's rule);
	// the remaining layers stay dense. Zero across the board for the non-MoE
	// Nomic checkpoints, which keeps their loader/forward path unchanged.
	NumExperts       int  `json:"num_experts"`
	MoETopK          int  `json:"moe_top_k"`
	MoEEveryNLayers  int  `json:"moe_every_n_layers"`
	NumSharedExperts int  `json:"num_shared_experts"`
	ExpertChoice     bool `json:"expert_choice_router"`
	// pooling reduces the per-token hidden states to one vector. Not in
	// NomicBert's config.json (it's a sentence-transformers module setting), so
	// LoadWeightsFromFS reads it from 1_Pooling/config.json — mean for
	// nomic-embed-text, CLS for CodeRankEmbed — defaulting to CLS when absent.
	pooling pooling
}

// HeadDim returns the per-head hidden dimension (HiddenDim / NumHeads).
func (c *Config) HeadDim() int { return c.HiddenDim / c.NumHeads }

// ValidateAssumptions errors if the config contradicts any baked-in
// assumption of the forward pass: post-norm only, standard LayerNorm,
// SwiGLU MLP, no biases on QKV/MLP, bidirectional, sequential block,
// full-rotation NeoX-style RoPE. Fail loudly at load time rather than
// silently produce junk activations. The plan §1 calls every one of
// these out — this is the runtime guard that pins them.
func (c *Config) ValidateAssumptions() error {
	switch {
	case c.Prenorm:
		return fmt.Errorf("encoder: prenorm=true unsupported (post-norm only)")
	case c.UseRMSNorm:
		return fmt.Errorf("encoder: use_rms_norm=true unsupported (LayerNorm only)")
	case c.ActivationFunction != "swiglu" && c.gatedMLP():
		// Gated activations other than swiglu (glu/geglu) would need a different
		// gate function than swigluMLP's SiLU.
		return fmt.Errorf("encoder: activation_function=%q unsupported (swiglu or gelu)", c.ActivationFunction)
	case c.gatedMLP() && (c.MLPFc1Bias || c.MLPFc2Bias):
		// swigluMLP has no bias terms; only the dense GELU MLP carries them.
		return fmt.Errorf("encoder: mlp_fc1_bias/mlp_fc2_bias unsupported with a gated MLP")
	case c.NumSharedExperts > 0:
		return fmt.Errorf("encoder: num_shared_experts=%d unsupported (0 only)", c.NumSharedExperts)
	case c.ExpertChoice:
		return fmt.Errorf("encoder: expert_choice_router=true unsupported (token-choice top-k only)")
	case c.MoEEveryNLayers > 0 && c.NumExperts <= 0:
		return fmt.Errorf("encoder: moe_every_n_layers=%d with num_experts=%d", c.MoEEveryNLayers, c.NumExperts)
	case c.MoEEveryNLayers > 0 && (c.MoETopK <= 0 || c.MoETopK > c.NumExperts):
		return fmt.Errorf("encoder: moe_top_k=%d out of range for %d experts", c.MoETopK, c.NumExperts)
	case c.Causal:
		return fmt.Errorf("encoder: causal=true unsupported (bidirectional only)")
	case c.ParallelBlock:
		return fmt.Errorf("encoder: parallel_block=true unsupported (sequential only)")
	case c.RoPEInterleaved:
		return fmt.Errorf("encoder: rotary_emb_interleaved=true unsupported (rotate_half only)")
	case c.RoPEFraction != 1.0:
		return fmt.Errorf("encoder: rotary_emb_fraction=%v unsupported (1.0 only)", c.RoPEFraction)
	case c.TypeVocabSize < 1:
		return fmt.Errorf("encoder: type_vocab_size must be ≥1, got %d", c.TypeVocabSize)
	case c.LayerNormEpsilon <= 0:
		return fmt.Errorf("encoder: layer_norm_epsilon must be >0, got %v", c.LayerNormEpsilon)
	case c.RoPEBase <= 0:
		return fmt.Errorf("encoder: rotary_emb_base must be >0, got %v", c.RoPEBase)
	case !c.ScaleAttnWeights:
		// selfAttention unconditionally applies 1/√headDim, i.e. it implements
		// scale_attn_weights=true; a checkpoint with it false would silently get
		// scaled anyway (wrong activations). Reject it rather than mislead.
		return fmt.Errorf("encoder: scale_attn_weights=false unsupported (attention is always 1/√headDim scaled)")
	}
	// CodeRankEmbed/NomicBert is RoPE (rotary_emb_base etc. above), so an odd
	// head dim panics in rope.go at the first Encode — catch it at load, same
	// as GTE.
	return validateDims("CodeRankEmbed", c.HiddenDim, c.NumHeads, c.NumLayers, c.IntermediateDim, true)
}

// LayerWeights bundles one transformer block's tensors. Matrices are
// stored in PyTorch's [out, in] row-major layout — matmul is then
// A · Bᵀ without a transpose copy, matching internal/embed's convention.
//
// Lifetime: every []float32 here aliases the underlying SafetensorsFile
// bytes (zero-copy unsafe slice). Do not mutate; do not let the parent
// Weights drop while these are in use.
type LayerWeights struct {
	Wqkv    []float32 // [3*HiddenDim, HiddenDim] fused Q/K/V input projection, no bias
	OutProj []float32 // [HiddenDim, HiddenDim] attention output projection, NO bias (verified against checkpoint)
	Norm1W  []float32 // [HiddenDim] post-attention LayerNorm weight
	Norm1B  []float32 // [HiddenDim] post-attention LayerNorm bias
	Fc11    []float32 // [IntermediateDim, HiddenDim] SwiGLU gate (not fused with Fc12 in the checkpoint)
	Fc12    []float32 // [IntermediateDim, HiddenDim] SwiGLU value (not fused with Fc11 in the checkpoint)
	Fc2     []float32 // [HiddenDim, IntermediateDim] output projection, no bias
	Norm2W  []float32 // [HiddenDim] post-MLP LayerNorm weight
	Norm2B  []float32 // [HiddenDim] post-MLP LayerNorm bias

	// Optional attention biases (nil for the bias-free v1.5/CodeRankEmbed
	// checkpoints; set when config qkv_proj_bias is true, as in v2-moe).
	WqkvB    []float32 // [3*HiddenDim]
	OutProjB []float32 // [HiddenDim]

	// Dense GELU MLP (nomic-embed-text-v2-moe's non-MoE layers). Mutually
	// exclusive with the SwiGLU trio above: Fc1 != nil selects this path.
	Fc1  []float32 // [IntermediateDim, HiddenDim]
	Fc1B []float32 // [IntermediateDim]
	Fc2B []float32 // [HiddenDim]

	// Mixture-of-experts FFN (set when IsMoE). Router is [NumExperts, HiddenDim];
	// ExpW1 is the experts stacked as [NumExperts*IntermediateDim, HiddenDim].
	// ExpW2 is stored TRANSPOSED per-expert — [NumExperts*HiddenDim, IntermediateDim],
	// each expert's [I,D] weight held as [D,I] — so the FFN's second projection uses
	// the SIMD A·Bᵀ matmul (audit #10); the on-disk tensor is [E*I, D].
	// ExpBias is a single shared [HiddenDim] row added after combining experts.
	IsMoE   bool
	Router  []float32
	ExpW1   []float32
	ExpW2   []float32
	ExpBias []float32
}

// Weights is the immutable per-checkpoint bundle returned by Load*.
// Multiple concurrent forward passes share one Weights instance.
//
// Plan §1 correction: the checkpoint stores the SwiGLU gate and value
// as TWO separate tensors (mlp.fc11 [3072,768] and mlp.fc12 [3072,768]),
// not a fused mlp.fc1 [6144,768]. There is also no out_proj bias and no
// final encoder LayerNorm beyond the last block's norm2.
type Weights struct {
	// be is the compute backend for f32 matmuls; nil = the pure-Go path (default).
	be Backend

	Cfg          Config
	WordEmb      []float32 // [VocabSize, HiddenDim]
	TokenTypeEmb []float32 // [TypeVocabSize, HiddenDim] — only row 0 used (single segment)
	EmbLN_W      []float32 // [HiddenDim]
	EmbLN_B      []float32 // [HiddenDim]
	Layers       []LayerWeights

	// Retained so the alias-backed []float32 fields stay valid for the
	// lifetime of Weights. Same lifetime contract as embed.StaticModel.
	st *embed.SafetensorsFile

	// ropeCache holds the rotary table across forwards instead of rebuilding it
	// per call — the same prefix-property cache GTE already uses (a longer table
	// contains every shorter one as a bit-identical prefix). Grows to the longest
	// sequence seen; zero value ready to use; safe for concurrent forwards.
	ropeCache ropeCache
}

// LoadWeights reads config.json + model.safetensors from a real on-disk
// directory. As of M8 the .safetensors blob is mmapped (not heap-copied)
// so the 547 MB CodeRankEmbed checkpoint stays in the OS page cache
// instead of dominating Go heap RSS. config.json (small) still goes
// through fs.ReadFile.
//
// Use LoadWeightsFromFS for fs.FS-backed (MapFS, embed.FS) paths — that
// route stays heap-backed because fs.FS doesn't expose a file descriptor.
func LoadWeights(dir string) (*Weights, error) {
	cfg, err := loadConfig(os.DirFS(dir), "config.json")
	if err != nil {
		return nil, err
	}
	if err := cfg.ValidateAssumptions(); err != nil {
		return nil, err
	}
	// Pooling is a declared per-model property — read it here too, not only in
	// LoadWeightsFromFS. Without this the mmap loader (and LoadQ8, which wraps it)
	// left cfg.pooling "" and poolOne silently treats "" as CLS, so a mean-pooled
	// checkpoint (nomic-embed-text-v1.5) returned CLS vectors with no error.
	// poolCLS is the absent-file fallback, matching LoadWeightsFromFS.
	if cfg.pooling, err = poolingFromConfig(dir, poolCLS); err != nil {
		return nil, err
	}
	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("encoder: open safetensors: %w", err)
	}
	return buildWeightsFromSafetensors(cfg, st)
}

// LoadWeightsFromFS reads config.json + model.safetensors from fsys/dir,
// validates every tensor's shape against Cfg, and returns the bundle.
// Returns the first error encountered (tensor name included) so the
// failure mode is one clear "tensor X has shape Y, want Z" rather than
// silent activation drift.
//
// fs.FS-backed (heap copy via fs.ReadFile). For mmap-backed loads from
// a real directory, use LoadWeights (M8 path).
func LoadWeightsFromFS(fsys fs.FS, dir string) (*Weights, error) {
	cfg, err := loadConfig(fsys, path.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	if err := cfg.ValidateAssumptions(); err != nil {
		return nil, err
	}
	// Pooling is a sentence-transformers module setting, not in NomicBert's
	// config.json. Read it (mean for nomic-embed-text, CLS for CodeRankEmbed);
	// fall back to CLS — CodeRankEmbed's mode and this loader's prior default.
	if cfg.pooling, err = poolingFromFS(fsys, dir, poolCLS); err != nil {
		return nil, err
	}
	st, err := embed.OpenSafetensorsFromFS(fsys, path.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("encoder: open safetensors: %w", err)
	}
	return buildWeightsFromSafetensors(cfg, st)
}

// buildWeightsFromSafetensors fills a *Weights from an already-opened
// SafetensorsFile. Factored out of LoadWeightsFromFS so both the
// heap-loaded (fs.FS) and mmap-loaded (LoadWeights) paths share the
// tensor-name + shape-validation contract — a future schema change is
// one edit, not two.
// tensorLoader accumulates the FIRST error across a run of tensor loads so a
// caller can write the loads as plain assignments instead of wrapping each one in
// a three-line error block.
//
// WHY: buildWeightsFromSafetensors loads 23 tensors, and every one of them used to
// cost the same four lines — `if X, err = loadF32(...); err != nil { return nil,
// err }`. That is ~69 of the function's 103 lines spent on identical punctuation,
// and it is what repowise flagged the file for (nesting five deep, the worst
// nesting score in the package).
//
// SEMANTICS ARE UNCHANGED, deliberately. f32 short-circuits once err is set, so no
// load runs after the first failure and the error returned is the SAME error value
// the immediate-return version returned. The only difference is that the caller
// walks to the end of its loop doing nothing before returning it — an error path,
// where a few no-op iterations cost nothing.
//
// THE HAZARD, because it is not obvious: a short-circuited f32 returns nil, and a
// nil is harmless only while nobody READS it. Storing it is fine; passing it to
// something that indexes is not. buildWeightsFromSafetensors has exactly one such
// consumer (transposeExpertsW2) and drains the accumulator immediately before it.
// Any future consumer of a loaded value inside this run needs the same guard.
type tensorLoader struct {
	st  *embed.SafetensorsFile
	err error
}

// f32 loads one tensor, or returns nil if an earlier load in this run already
// failed. The shape is variadic to match loadF32's own `want ...int` at the call
// site without a []int literal at each of the 23 of them.
func (l *tensorLoader) f32(name string, want ...int) []float32 {
	if l.err != nil {
		return nil
	}
	v, err := loadF32(l.st, name, want)
	if err != nil {
		l.err = err
	}
	return v
}

func buildWeightsFromSafetensors(cfg *Config, st *embed.SafetensorsFile) (*Weights, error) {
	w := &Weights{Cfg: *cfg, st: st, Layers: make([]LayerWeights, cfg.NumLayers)}
	ld := &tensorLoader{st: st}

	// Embeddings + emb_ln.
	w.WordEmb = ld.f32("embeddings.word_embeddings.weight", cfg.VocabSize, cfg.HiddenDim)
	w.TokenTypeEmb = ld.f32("embeddings.token_type_embeddings.weight", cfg.TypeVocabSize, cfg.HiddenDim)
	w.EmbLN_W = ld.f32("emb_ln.weight", cfg.HiddenDim)
	w.EmbLN_B = ld.f32("emb_ln.bias", cfg.HiddenDim)

	// Per-layer (9 tensors × 12 layers = 108, plus 4 above = 112 total).
	for i := 0; i < cfg.NumLayers; i++ {
		pfx := fmt.Sprintf("encoder.layers.%d.", i)
		l := &w.Layers[i]
		l.Wqkv = ld.f32(pfx+"attn.Wqkv.weight", 3*cfg.HiddenDim, cfg.HiddenDim)
		l.OutProj = ld.f32(pfx+"attn.out_proj.weight", cfg.HiddenDim, cfg.HiddenDim)
		l.Norm1W = ld.f32(pfx+"norm1.weight", cfg.HiddenDim)
		l.Norm1B = ld.f32(pfx+"norm1.bias", cfg.HiddenDim)
		// Optional attention biases (present iff the checkpoint has them).
		if cfg.QKVProjBias {
			l.WqkvB = ld.f32(pfx+"attn.Wqkv.bias", 3*cfg.HiddenDim)
			l.OutProjB = ld.f32(pfx+"attn.out_proj.bias", cfg.HiddenDim)
		}

		switch {
		case cfg.isMoELayer(i):
			// MoE FFN: router + experts stacked into two [E*I, D] tensors.
			l.IsMoE = true
			E, I, D := cfg.NumExperts, cfg.IntermediateDim, cfg.HiddenDim
			l.Router = ld.f32(pfx+"mlp.router.layer.weight", E, D)
			l.ExpW1 = ld.f32(pfx+"mlp.experts.mlp.w1", E*I, D)
			l.ExpW2 = ld.f32(pfx+"mlp.experts.mlp.w2", E*I, D)
			// Store each expert's W2 transposed [I,D] → [D,I] so the FFN's second
			// projection can run through the SIMD A·Bᵀ matmul instead of a scalar
			// triple-loop (audit #10). One transpose per expert, once at load.
			//
			// THE ONE PLACE THE ACCUMULATOR MUST BE CHECKED MID-FUNCTION. Every other
			// statement here just stores what ld.f32 returned, and a nil on the error
			// path is harmless because it is never read. This one CONSUMES ExpW2, and
			// transposeExpertsW2 over a nil slice with non-zero E/I/D would index out
			// of range. So the accumulator is drained here, before the only consumer.
			if ld.err != nil {
				return nil, ld.err
			}
			l.ExpW2 = transposeExpertsW2(l.ExpW2, E, I, D)
			l.ExpBias = ld.f32(pfx+"mlp.experts.bias", D)
		case cfg.gatedMLP():
			l.Fc11 = ld.f32(pfx+"mlp.fc11.weight", cfg.IntermediateDim, cfg.HiddenDim)
			l.Fc12 = ld.f32(pfx+"mlp.fc12.weight", cfg.IntermediateDim, cfg.HiddenDim)
			l.Fc2 = ld.f32(pfx+"mlp.fc2.weight", cfg.HiddenDim, cfg.IntermediateDim)
		default:
			// Dense two-matrix GELU MLP.
			l.Fc1 = ld.f32(pfx+"mlp.fc1.weight", cfg.IntermediateDim, cfg.HiddenDim)
			l.Fc2 = ld.f32(pfx+"mlp.fc2.weight", cfg.HiddenDim, cfg.IntermediateDim)
			if cfg.MLPFc1Bias {
				l.Fc1B = ld.f32(pfx+"mlp.fc1.bias", cfg.IntermediateDim)
			}
			if cfg.MLPFc2Bias {
				l.Fc2B = ld.f32(pfx+"mlp.fc2.bias", cfg.HiddenDim)
			}
		}
		l.Norm2W = ld.f32(pfx+"norm2.weight", cfg.HiddenDim)
		l.Norm2B = ld.f32(pfx+"norm2.bias", cfg.HiddenDim)
	}
	if ld.err != nil {
		return nil, ld.err
	}
	return w, nil
}

// isMoELayer reports whether layer i uses the MoE FFN. Mirrors the reference
// NomicBertEncoder: with moe_every_n_layers = n > 0, layer i is MoE iff
// i%n == 1 — so for n=2 the ODD layers (1,3,5,…) are MoE and the even ones stay
// dense. Always false when the checkpoint declares no experts.
func (c *Config) isMoELayer(i int) bool {
	return c.MoEEveryNLayers > 0 && c.NumExperts > 0 && i%c.MoEEveryNLayers == 1
}

// gatedMLP reports whether the dense layers use the gated (SwiGLU/GLU) MLP —
// two input projections plus a gate — rather than the plain two-matrix GELU MLP.
// CodeRankEmbed and nomic-embed-text-v1.5 are "swiglu"; v2-moe is "gelu".
// Defaults to the gated form for an empty/unknown value: every Nomic checkpoint
// aikit supported before v2-moe was SwiGLU and the loader read fc11/fc12
// unconditionally, so only an explicit plain-GELU activation switches paths.
func (c *Config) gatedMLP() bool {
	switch c.ActivationFunction {
	case "gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh":
		return false
	}
	return true
}

// geluTanh reports whether the dense GELU MLP (gatedMLP() == false) should use
// the tanh approximation instead of the exact erf form. "gelu_new", "gelu_fast",
// and "gelu_pytorch_tanh" are three HF names for the identical tanh formula (see
// elem.GELUTanhF32) — a DIFFERENT function from plain "gelu"'s erf form, not an
// approximation safe to collapse into it. Substituting one for the other is a
// silent model change (see vision.geluTanh's identical caveat for SigLIP).
func (c *Config) geluTanh() bool {
	switch c.ActivationFunction {
	case "gelu_new", "gelu_fast", "gelu_pytorch_tanh":
		return true
	}
	return false
}

func loadConfig(fsys fs.FS, p string) (*Config, error) {
	b, err := fs.ReadFile(fsys, p)
	if err != nil {
		return nil, fmt.Errorf("encoder: read %s: %w", p, err)
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("encoder: parse %s: %w", p, err)
	}
	return &c, nil
}

// loadF32 reads a shape-checked f32 weight. It delegates to the shared
// embed.SafetensorsFile.TensorF32 (which also widens BF16/F16 — CodeRankEmbed's
// weights are F32, so the F32 path is taken); kept as a thin alias so the call sites
// don't churn.
func loadF32(st *embed.SafetensorsFile, name string, want []int) ([]float32, error) {
	return st.TensorF32(name, want...)
}

// transposeExpertsW2 returns the stacked expert W2 tensor rewritten from
// [E*I, D] (each expert [I,D], row-major) to [E*D, I] (each expert [D,I]).
// This lets moeMLP apply W2 with the SIMD A·Bᵀ matmul — out_j = Σ_i x1_i·W2[i,j]
// becomes a matmul of x1 against W2ᵀ, where W2ᵀ[j,i] = W2[i,j] — instead of a
// scalar triple-loop (audit #10). Run once per load; the per-expert block size is
// unchanged (I·D), so moeMLP's offset arithmetic is identical.
//
// It allocates a fresh owned buffer rather than transposing in place: the loaded
// tensor is a zero-copy view aliasing the read-only mmap (LayerWeights: "Do not
// mutate"), so an in-place write would fault. The MoE experts are a small slice
// of the model, so owning this one transposed copy is a negligible memory cost.
func transposeExpertsW2(w2 []float32, E, I, D int) []float32 {
	out := make([]float32, len(w2))
	for e := range E {
		src := w2[e*I*D : (e+1)*I*D]  // expert e, [I, D] row-major
		dst := out[e*I*D : (e+1)*I*D] // expert e, [D, I] row-major
		for i := range I {
			row := src[i*D : (i+1)*D]
			for j := range D {
				dst[j*I+i] = row[j]
			}
		}
	}
	return out
}
