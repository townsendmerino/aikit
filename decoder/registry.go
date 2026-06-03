package decoder

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// archAdapter resolves a parsed config.json into the family-agnostic
// Architecture descriptor + the family's tensor-name schema, both consumed by
// the generic forward pass / loader.
type archAdapter func(*Config) (*Architecture, *tensorSchema, error)

// registry maps config.json model_type → its adapter. Adding a family
// (multi-model-plan G2+) is a new entry here plus its tensor schema — the
// forward pass itself doesn't change.
var registry = map[string]archAdapter{
	"gemma3":      gemma3Architecture,
	"gemma3_text": gemma3Architecture, // the 270M/1B text checkpoints
	"qwen3":       qwen3Architecture,  // Qwen3 dense (0.6B/1.7B/4B/8B/…)
}

// resolveArchitecture picks the adapter for cfg.ModelType and builds the
// descriptor + schema. An unknown model_type is a loud error, not a silent
// wrong load.
func resolveArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	adapter, ok := registry[cfg.ModelType]
	if !ok {
		return nil, nil, fmt.Errorf("decoder: unsupported model_type %q (have: %s)", cfg.ModelType, knownModelTypes())
	}
	return adapter(cfg)
}

func knownModelTypes() string {
	keys := make([]string, 0, len(registry))
	for k := range registry {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return strings.Join(keys, ", ")
}

// gemma3Architecture expresses Gemma 3 as a descriptor: RMSNorm with (1+w),
// the 4-norm sandwich, GeGLU, QK-norm, the query_pre_attn_scalar attention
// scale, dual-base RoPE (local/global per layer), √hidden embedding scale, and
// a tied LM head with no soft-capping. ValidateAssumptions pins the bits this
// forward pass can't vary (Gemma-2 soft-capping, unsupported activation).
func gemma3Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.ValidateAssumptions(); err != nil {
		return nil, nil, err
	}
	return &Architecture{
		Name:              "gemma3",
		HiddenDim:         cfg.HiddenDim,
		NumLayers:         cfg.NumLayers,
		NumHeads:          cfg.NumHeads,
		NumKVHeads:        cfg.NumKVHeads,
		HeadDim:           cfg.HeadDim,
		IntermediateDim:   cfg.IntermediateDim,
		VocabSize:         cfg.VocabSize,
		Norm:              NormRMS,
		RMSAddOne:         true,
		NormEps:           cfg.RMSNormEps,
		NormPlacement:     NormSandwich4,
		Act:               ActGeluTanh,
		QKNorm:            true,
		AttnScale:         math.Pow(cfg.QueryPreAttnScalar, -0.5),
		SlidingWindow:     cfg.SlidingWindow,
		layerIsGlobal:     cfg.IsGlobalLayer,
		RoPELocalBase:     cfg.RoPELocalBase,
		RoPEGlobalBase:    cfg.RoPEGlobalBase,
		EmbedScale:        math.Sqrt(float64(cfg.HiddenDim)),
		TiedLMHead:        true,
		FinalLogitSoftcap: cfg.FinalLogitSoftcap, // 0 (ValidateAssumptions rejects nonzero)
		AttnLogitSoftcap:  cfg.AttnLogitSoftcap,
	}, &gemma3TensorSchema, nil
}

// qwen3Architecture expresses Qwen3 dense (multi-model-plan G2): RMSNorm
// without the (1+w) offset, the Pre2 norm placement (no post-sublayer norms),
// SwiGLU, QK-norm (Qwen3 keeps it), 1/√head_dim attention scale, single-base
// RoPE, no embedding scale, and a separate lm_head (untied). No QKV bias
// (Qwen3 dropped Qwen2's). The tensor schema is qwen3TensorSchema.
func qwen3Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateQwen3(); err != nil {
		return nil, nil, err
	}
	return &Architecture{
		Name:            "qwen3",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         cfg.HeadDim,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKNorm:          true,
		AttnScale:       math.Pow(float64(cfg.HeadDim), -0.5),
		SlidingWindow:   0,                  // Qwen3 dense: full attention
		layerIsGlobal:   nil,                // all-global
		RoPELocalBase:   cfg.RoPEGlobalBase, // single base (rope_theta)
		RoPEGlobalBase:  cfg.RoPEGlobalBase,
		EmbedScale:      0,     // none
		TiedLMHead:      false, // finalized from lm_head.weight presence at load
	}, &qwen3TensorSchema, nil
}
