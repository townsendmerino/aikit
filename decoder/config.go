package decoder

import (
	"encoding/json"
	"fmt"
	"io/fs"
)

// Config captures the Gemma 3 architecture constants the forward pass
// depends on. Field tags follow the HF config.json schema so a checkpoint's
// config drives the loader rather than hardcoded constants — the same
// config-driven approach encoder.Config uses, which is what lets one code
// path serve both 270M and 1B (and beyond) unchanged.
//
// Values that vary per layer (the 5:1 local:global attention pattern) are
// derived from SlidingWindowPattern at load time, not stored per layer.
type Config struct {
	VocabSize            int     `json:"vocab_size"`              // 262144
	HiddenDim            int     `json:"hidden_size"`             // 640 (270M)
	NumLayers            int     `json:"num_hidden_layers"`       // 18 (270M)
	NumHeads             int     `json:"num_attention_heads"`     // 4 (270M)
	NumKVHeads           int     `json:"num_key_value_heads"`     // 1 (270M) — GQA
	HeadDim              int     `json:"head_dim"`                // 256 (270M); note heads*headDim != hidden
	IntermediateDim      int     `json:"intermediate_size"`       // 2048 (270M) — GeGLU
	MaxPositions         int     `json:"max_position_embeddings"` // 32768
	RMSNormEps           float64 `json:"rms_norm_eps"`
	RoPELocalBase        float64 `json:"rope_local_base_freq"`   // 10000
	RoPEGlobalBase       float64 `json:"rope_theta"`             // 1000000
	SlidingWindow        int     `json:"sliding_window"`         // 512 (270M)
	SlidingWindowPattern int     `json:"sliding_window_pattern"` // 6 → 5 local : 1 global
	QueryPreAttnScalar   float64 `json:"query_pre_attn_scalar"`  // 256 (270M)
	UseQKNorm            bool    `json:"use_qk_norm"`            // true in Gemma 3
	HiddenActivation     string  `json:"hidden_activation"`      // "gelu_pytorch_tanh"

	// LayerTypes is the per-layer attention kind ("sliding_attention" /
	// "full_attention"). Gemma 3's checkpoints carry the local:global pattern
	// here explicitly; sliding_window_pattern is often null in config.json, so
	// this is the authoritative source when present (see IsGlobalLayer).
	LayerTypes []string `json:"layer_types"`

	// Gemma 2 fields that MUST be absent/zero in a Gemma 3 checkpoint.
	// ValidateAssumptions rejects a checkpoint that still sets them so we
	// fail loudly rather than silently skip soft-capping.
	FinalLogitSoftcap float64 `json:"final_logit_softcapping"`
	AttnLogitSoftcap  float64 `json:"attn_logit_softcapping"`
}

// IsGlobalLayer reports whether layer i is a global (full) attention layer
// vs a local (sliding-window) one. Gemma 3 carries this per-layer in
// LayerTypes ("full_attention" vs "sliding_attention"), which is the
// authoritative source. Only if LayerTypes is absent do we fall back to the
// SlidingWindowPattern arithmetic (a global layer as the last of each group;
// pattern=6 → layers 5, 11, 17 → 3 global of 18 in the 270M).
//
// Getting this right is load-bearing: local and global layers use different
// RoPE bases (10k vs 1e6), so a misclassified layer silently corrupts logits.
func (c *Config) IsGlobalLayer(i int) bool {
	if i >= 0 && i < len(c.LayerTypes) {
		return c.LayerTypes[i] == "full_attention"
	}
	p := c.SlidingWindowPattern
	if p <= 0 {
		return true // no pattern configured → all-global (degenerate)
	}
	return (i+1)%p == 0
}

// ValidateAssumptions fails loudly on any config the scaffolded forward pass
// is not built to honor. Mirrors encoder.Config.ValidateAssumptions: pin the
// assumptions at load time rather than produce junk logits at run time.
func (c *Config) ValidateAssumptions() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.HeadDim == 0:
		return fmt.Errorf("decoder: missing required dim (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.HeadDim)
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder: num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder: vocab_size is zero")
	case c.HiddenActivation != "" && c.HiddenActivation != "gelu_pytorch_tanh":
		return fmt.Errorf("decoder: hidden_activation=%q unsupported (gelu_pytorch_tanh / GeGLU only)", c.HiddenActivation)
	case c.FinalLogitSoftcap != 0 || c.AttnLogitSoftcap != 0:
		return fmt.Errorf("decoder: soft-capping set (final=%v attn=%v) — that's a Gemma 2 checkpoint; this path is Gemma 3 (QK-norm) only",
			c.FinalLogitSoftcap, c.AttnLogitSoftcap)
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder: rms_norm_eps must be >0, got %v", c.RMSNormEps)
	case c.RoPELocalBase <= 0 || c.RoPEGlobalBase <= 0:
		return fmt.Errorf("decoder: rope base must be >0 (local=%v global=%v)", c.RoPELocalBase, c.RoPEGlobalBase)
	case len(c.LayerTypes) > 0 && len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("decoder: layer_types has %d entries, want num_hidden_layers=%d", len(c.LayerTypes), c.NumLayers)
	}
	for i, lt := range c.LayerTypes {
		if lt != "sliding_attention" && lt != "full_attention" {
			return fmt.Errorf("decoder: layer_types[%d]=%q unsupported (want sliding_attention/full_attention)", i, lt)
		}
	}
	return nil
}

// loadConfig reads and parses config.json from fsys.
func loadConfig(fsys fs.FS, name string) (*Config, error) {
	b, err := fs.ReadFile(fsys, name)
	if err != nil {
		return nil, fmt.Errorf("decoder: read %s: %w", name, err)
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("decoder: parse %s: %w", name, err)
	}
	return &c, nil
}
