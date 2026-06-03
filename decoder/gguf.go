package decoder

import (
	"fmt"

	"github.com/townsendmerino/aikit/embed"
)

// GGUF loading (multi-model-plan G7) — read a quantized llama.cpp checkpoint and
// run it through the generic forward. The GGUF file carries both the
// architecture config (metadata) and the weights (dequantized from the mmap,
// then optionally re-quantized to resident int8/int4 per the quant mode), so no
// separate config.json/safetensors is needed. Two layout quirks vs the HF
// safetensors path: tensors use llama.cpp's blk.N.* names, and the q/k
// projections are stored in llama.cpp's interleaved-RoPE permutation, which is
// inverted here to match this package's HF-convention rotate_half RoPE.
//
// Scope: the llama architecture; F32/F16/Q8_0/Q4_0 and the K-quants Q4_K/Q6_K
// (so Q4_K_M files load end-to-end). Other architectures and the remaining quant
// types (Q5_K/Q3_K/IQ*) are follow-ups and fail loudly.

// ggufConfig synthesizes a Config from GGUF metadata. Only llama is supported.
func ggufConfig(g *embed.GGUFFile) (*Config, error) {
	arch, _ := g.Str("general.architecture")
	if arch != "llama" {
		return nil, fmt.Errorf("decoder(gguf): architecture %q unsupported (have: llama)", arch)
	}
	u := func(k string) int {
		v, _ := g.Uint(k)
		return int(v)
	}
	cfg := &Config{
		ModelType:       "llama",
		HiddenDim:       u("llama.embedding_length"),
		NumLayers:       u("llama.block_count"),
		NumHeads:        u("llama.attention.head_count"),
		NumKVHeads:      u("llama.attention.head_count_kv"),
		IntermediateDim: u("llama.feed_forward_length"),
		HiddenAct:       "silu",
	}
	if eps, ok := g.Float("llama.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float("llama.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	} else {
		cfg.RoPEGlobalBase = 10000 // llama.cpp default
	}
	// Vocab size from the embedding table (ne1 = rows = vocab).
	if dims, _, err := g.Tensor("token_embd.weight"); err == nil && len(dims) == 2 {
		cfg.VocabSize = dims[1]
	}
	// EOS for generation stop (resolveEOSIDs falls back to this when there's no
	// generation_config.json next to the .gguf).
	if eos, ok := g.Uint("tokenizer.ggml.eos_token_id"); ok {
		cfg.EOSTokenID = []byte(fmt.Sprintf("%d", eos))
	}
	return cfg, nil
}

// loadGGUFWeights parses a .gguf file and builds the weight bundle, mapping
// llama.cpp tensor names to the descriptor and un-permuting q/k.
func loadGGUFWeights(path string, quant quantMode) (*Weights, error) {
	// mmap, not heap-read: the raw quantized bytes stay in reclaimable page
	// cache while we dequantize tensor-by-tensor. The weights end up as fresh
	// (f32 or int8) copies, so the mapping is unneeded once the build returns.
	g, err := embed.OpenGGUFMmap(path)
	if err != nil {
		return nil, err
	}
	defer g.Close()
	cfg, err := ggufConfig(g)
	if err != nil {
		return nil, err
	}
	arch, _, err := resolveArchitecture(cfg) // llama descriptor + finalizeRoPE
	if err != nil {
		return nil, err
	}
	return buildWeightsFromGGUF(cfg, arch, g, quant)
}

// buildWeightsFromGGUF dequantizes the GGUF tensors into the weight bundle.
// When quant is set, each matmul tensor is re-quantized (per-row int8 or
// group-wise int4) right after it is dequantized (and un-permuted) and its f32
// is freed — so a Q4/Q8 GGUF lands resident as int8/int4 (~¼ / ~⅛ f32) without
// ever materializing the whole model in f32 (see loadWeights). The GGUF's own
// quant is lossy and so is the re-quant, but it captures nearly all of what a
// Q4_K_M file carries.
func buildWeightsFromGGUF(cfg *Config, arch *Architecture, g *embed.GGUFFile, quant quantMode) (*Weights, error) {
	hidden, hd := arch.HiddenDim, arch.HeadDim
	w := &Weights{Cfg: *cfg, arch: arch, Layers: make([]LayerWeights, arch.NumLayers)}

	maybeQuant := func(m weightMat) weightMat {
		m.quantize(quant)
		return m
	}
	// mat loads a tensor as a [out, in] weightMat, shape-checked, quantized when
	// requested.
	mat := func(name string, out, in int) (weightMat, error) {
		dims, data, err := g.Tensor(name)
		if err != nil {
			return weightMat{}, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return weightMat{}, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		return maybeQuant(newWeightMat(data, out, in)), nil
	}
	// embMat loads the embedding / LM head, which is logit-critical — quantize
	// it with the embedding policy (int8 even in int4 mode), not the projection
	// mode.
	embMat := func(name string, out, in int) (weightMat, error) {
		dims, data, err := g.Tensor(name)
		if err != nil {
			return weightMat{}, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return weightMat{}, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		m := newWeightMat(data, out, in)
		m.quantize(quant.embedding())
		return m, nil
	}
	// vec loads a 1-D tensor (norm).
	vec := func(name string, n int) ([]float32, error) {
		dims, data, err := g.Tensor(name)
		if err != nil {
			return nil, err
		}
		if len(dims) != 1 || dims[0] != n {
			return nil, fmt.Errorf("decoder(gguf): %q dims %v, want [%d]", name, dims, n)
		}
		return data, nil
	}
	// permMat loads a q/k projection and inverts llama.cpp's RoPE permutation
	// (on the f32 data, before any int8 quantization).
	permMat := func(name string, out, in, nHead int) (weightMat, error) {
		dims, data, err := g.Tensor(name)
		if err != nil {
			return weightMat{}, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return weightMat{}, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		data = ggufInvPermute(data, out, in, nHead)
		return maybeQuant(newWeightMat(data, out, in)), nil
	}

	var err error
	if w.Embed, err = embMat("token_embd.weight", cfg.VocabSize, hidden); err != nil {
		return nil, err
	}
	if w.FinalNorm, err = vec("output_norm.weight", hidden); err != nil {
		return nil, err
	}
	// Separate output head when present; else tied to the embedding.
	arch.TiedLMHead = true
	if g.Has("output.weight") {
		if w.LMHead, err = embMat("output.weight", cfg.VocabSize, hidden); err != nil {
			return nil, err
		}
		arch.TiedLMHead = false
	}

	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	for i := 0; i < arch.NumLayers; i++ {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		if l.PreAttnNorm, err = vec(p+"attn_norm.weight", hidden); err != nil {
			return nil, err
		}
		if l.QProj, err = permMat(p+"attn_q.weight", qDim, hidden, arch.NumHeads); err != nil {
			return nil, err
		}
		if l.KProj, err = permMat(p+"attn_k.weight", kvDim, hidden, arch.NumKVHeads); err != nil {
			return nil, err
		}
		if l.VProj, err = mat(p+"attn_v.weight", kvDim, hidden); err != nil {
			return nil, err
		}
		if l.OProj, err = mat(p+"attn_output.weight", hidden, qDim); err != nil {
			return nil, err
		}
		if l.PreMLPNorm, err = vec(p+"ffn_norm.weight", hidden); err != nil {
			return nil, err
		}
		if l.GateProj, err = mat(p+"ffn_gate.weight", arch.IntermediateDim, hidden); err != nil {
			return nil, err
		}
		if l.UpProj, err = mat(p+"ffn_up.weight", arch.IntermediateDim, hidden); err != nil {
			return nil, err
		}
		if l.DownProj, err = mat(p+"ffn_down.weight", hidden, arch.IntermediateDim); err != nil {
			return nil, err
		}
	}
	return w, nil
}

// ggufInvPermute inverts llama.cpp's q/k weight permutation. llama.cpp stores
// the projection rows in interleaved-pair order for its RoPE; this package uses
// HF's rotate_half (first-half / second-half) order. For head h, the HF row
// (s*half + j) — s∈{0,1} selecting the half, j the position within it — comes
// from the GGUF row (2*j + s). w is [out, in] row-major; out = nHead*headDim.
func ggufInvPermute(w []float32, out, in, nHead int) []float32 {
	hd := out / nHead
	half := hd / 2
	res := make([]float32, len(w))
	for h := 0; h < nHead; h++ {
		for s := 0; s < 2; s++ {
			for j := 0; j < half; j++ {
				hfRow := h*hd + s*half + j
				ggufRow := h*hd + 2*j + s
				copy(res[hfRow*in:hfRow*in+in], w[ggufRow*in:ggufRow*in+in])
			}
		}
	}
	return res
}
