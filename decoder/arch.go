package decoder

// Architecture is the resolved, family-agnostic description of a decoder LLM's
// structure. ONE generic forward pass (runLayers/forward/causalAttention/
// gatedMLP) reads it; per-family adapters (registry.go) populate it from that
// family's config.json. Gemma 3 is currently the only family — it's expressed
// as a descriptor here so adding Llama/Mistral/Qwen (multi-model-plan G2) is
// descriptor population, not new forward code.
//
// Every field below is consumed by the forward pass. Behavioural knobs that
// need new tensors (untied LM head, QKV bias, MoE) are deferred to later G-
// milestones; the forward pass rejects descriptor values it doesn't implement
// rather than silently mis-running.
type Architecture struct {
	Name string // family name, for logs/errors ("gemma3")

	// Dims (mirrors config.json; the loader also reads these for tensor shapes).
	HiddenDim, NumLayers, NumHeads, NumKVHeads, HeadDim int
	IntermediateDim, VocabSize                          int

	// Norm.
	Norm          NormKind
	RMSAddOne     bool // Gemma's (1+w) scaling; false for Llama/Qwen
	NormEps       float64
	NormPlacement NormPlacement // Pre2 (Llama) | Sandwich4 (Gemma)

	// MLP.
	Act ActKind

	// Attention.
	QKNorm        bool             // RMSNorm on Q and K per head before RoPE (Gemma3, Qwen3)
	AttnScale     float64          // explicit q·k multiplier (resolved: query_pre_attn_scalar^-0.5 or 1/sqrt(headDim))
	SlidingWindow int              // 0 = none
	layerIsGlobal func(i int) bool // per-layer global(full) vs local(sliding) attention

	// RoPE (dual base for Gemma's local/global layers; equal bases = single-base).
	RoPELocalBase, RoPEGlobalBase float64

	// Embedding / head.
	EmbedScale float64 // 0 or 1 = none; Gemma = sqrt(hidden)
	TiedLMHead bool    // tied embeddings as the LM head vs a separate lm_head

	// Output soft-capping (Gemma 2; 0 = none, which is Gemma 3).
	FinalLogitSoftcap float64
	AttnLogitSoftcap  float64
}

// NormKind selects the normalization. Only RMSNorm is implemented today;
// LayerNorm (GPT-2/NeoX) is multi-model-plan G5.
type NormKind int

const (
	NormRMS NormKind = iota
	NormLayer
)

// NormPlacement selects where norms sit relative to the residual adds. Pre2 is
// the Llama/Mistral/Qwen norm-before-each-sublayer; Sandwich4 is Gemma's
// pre+post norm on both attention and MLP.
type NormPlacement int

const (
	NormPre2 NormPlacement = iota
	NormSandwich4
)

// ActKind selects the MLP activation. GeluTanh = Gemma's GeGLU; SiLU = the
// SwiGLU used by Llama/Mistral/Qwen. Gelu/ReLU2/non-gated MLPs are later G's.
type ActKind int

const (
	ActGeluTanh ActKind = iota
	ActSiLU
)

// isGlobalLayer reports whether layer i uses full (global) attention vs local
// (sliding-window). Defaults to global when no per-layer function is set.
func (a *Architecture) isGlobalLayer(i int) bool {
	if a.layerIsGlobal != nil {
		return a.layerIsGlobal(i)
	}
	return true
}

// ropeBase returns the RoPE base for layer i (Gemma uses a smaller base on the
// local layers; single-base families set both equal).
func (a *Architecture) ropeBase(i int) float64 {
	if a.isGlobalLayer(i) {
		return a.RoPEGlobalBase
	}
	return a.RoPELocalBase
}
