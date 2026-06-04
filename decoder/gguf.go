package decoder

import (
	"encoding/json"
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

// ggufConfig synthesizes a Config from GGUF metadata, dispatching on
// general.architecture. The resolved Config feeds resolveArchitecture, so a
// GGUF reuses the same per-family adapter as the safetensors path.
func ggufConfig(g *embed.GGUFFile) (*Config, error) {
	arch, _ := g.Str("general.architecture")
	switch arch {
	case "llama":
		return ggufLlamaConfig(g)
	case "mellum":
		return ggufMellumConfig(g)
	default:
		return nil, fmt.Errorf("decoder(gguf): architecture %q unsupported (have: llama, mellum)", arch)
	}
}

// ggufVocabSize reads the vocab size from the embedding tensor's dims (no
// dequant) or, failing that, the tokens metadata array length.
func ggufVocabSize(g *embed.GGUFFile) int {
	if dims, ok := g.Dims("token_embd.weight"); ok && len(dims) == 2 {
		return dims[1] // [in=hidden, out=vocab]
	}
	if toks, ok := g.Metadata["tokenizer.ggml.tokens"].([]any); ok {
		return len(toks)
	}
	return 0
}

// ggufEOS sets cfg.EOSTokenID from the GGUF tokenizer metadata (resolveEOSIDs
// falls back to this when there's no generation_config.json next to the .gguf).
func ggufEOS(g *embed.GGUFFile, cfg *Config) {
	if eos, ok := g.Uint("tokenizer.ggml.eos_token_id"); ok {
		cfg.EOSTokenID = []byte(fmt.Sprintf("%d", eos))
	}
}

func ggufLlamaConfig(g *embed.GGUFFile) (*Config, error) {
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
		VocabSize:       ggufVocabSize(g),
	}
	if eps, ok := g.Float("llama.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float("llama.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	} else {
		cfg.RoPEGlobalBase = 10000 // llama.cpp default
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufMellumConfig builds a Mellum2 Config from the mellum.* metadata: dims,
// the MoE counts (incl. the narrower expert_feed_forward_length), the sliding/
// full layer pattern, and the YaRN rope scaling — synthesized into the same
// shapes the mellum adapter consumes (LayerTypes + a rope_parameters JSON), so
// resolveArchitecture runs the identical descriptor build as the safetensors path.
func ggufMellumConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("mellum." + k)
		return int(v)
	}
	gf := func(k string) float64 { // float-or-int metadata
		if v, ok := g.Float("mellum." + k); ok {
			return v
		}
		v, _ := g.Uint("mellum." + k)
		return float64(v)
	}
	normTopK := true
	cfg := &Config{
		ModelType:           "mellum",
		HiddenDim:           u("embedding_length"),
		NumLayers:           u("block_count"),
		NumHeads:            u("attention.head_count"),
		NumKVHeads:          u("attention.head_count_kv"),
		HeadDim:             u("attention.key_length"),
		IntermediateDim:     u("feed_forward_length"),        // dense (vestigial)
		MoeIntermediateSize: u("expert_feed_forward_length"), // expert FFN width
		NumExperts:          u("expert_count"),
		NumExpertsPerTok:    u("expert_used_count"),
		SlidingWindow:       u("attention.sliding_window"),
		NormTopKProb:        &normTopK,
		HiddenAct:           "silu",
		VocabSize:           ggufVocabSize(g),
		RMSNormEps:          gf("attention.layer_norm_rms_epsilon"),
	}
	// Per-layer attention type: sliding_window_pattern[i] true ⇒ sliding (local),
	// false ⇒ full (global).
	pat, ok := g.Metadata["mellum.attention.sliding_window_pattern"].([]any)
	if !ok || len(pat) != cfg.NumLayers {
		return nil, fmt.Errorf("decoder(gguf-mellum): sliding_window_pattern missing or wrong length")
	}
	for _, p := range pat {
		if b, _ := p.(bool); b {
			cfg.LayerTypes = append(cfg.LayerTypes, "sliding_attention")
		} else {
			cfg.LayerTypes = append(cfg.LayerTypes, "full_attention")
		}
	}
	// Synthesize rope_parameters: YaRN on the full layers (freq_base + scaling),
	// plain RoPE on the sliding layers (freq_base_swa). llama.cpp truncates the
	// attention_factor to f32; the mscale tolerance absorbs that.
	base := gf("rope.freq_base")
	baseSwa := gf("rope.freq_base_swa")
	if baseSwa == 0 {
		baseSwa = base
	}
	cfg.RopeParameters = json.RawMessage(fmt.Sprintf(
		`{"full_attention":{"rope_type":"yarn","rope_theta":%g,"factor":%g,"original_max_position_embeddings":%g,"beta_fast":%g,"beta_slow":%g,"attention_factor":%g},`+
			`"sliding_attention":{"rope_type":"default","rope_theta":%g}}`,
		base, gf("rope.scaling.factor"), gf("rope.scaling.original_context_length"),
		gf("rope.scaling.yarn_beta_fast"), gf("rope.scaling.yarn_beta_slow"),
		gf("rope.scaling.yarn_attn_factor"), baseSwa))
	ggufEOS(g, cfg)
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

	// streamMat builds a [out, in] weightMat by streaming the tensor's rows through
	// the GGUF dequantizer and quantizing each row directly into the resident
	// arrays — no whole-tensor f32 intermediate (see streamQuantized). rowSrc maps
	// a destination row index to its source element offset in the tensor (identity
	// for a plain load; a permutation for the RoPE-permuted q/k projections).
	streamMat := func(name string, out, in int, mode quantMode, rowSrc func(r int) int) (weightMat, error) {
		dims, into, err := g.RowDequantizer(name)
		if err != nil {
			return weightMat{}, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return weightMat{}, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		return streamQuantized(out, in, mode, func(r int, dst []float32) error {
			return into(rowSrc(r), dst)
		})
	}
	// mat loads a tensor as a [out, in] weightMat, shape-checked, quantized when
	// requested.
	mat := func(name string, out, in int) (weightMat, error) {
		return streamMat(name, out, in, quant, func(r int) int { return r * in })
	}
	// embMat loads the embedding / LM head, which is logit-critical — quantize
	// it with the embedding policy (int8 even in int4 mode), not the projection
	// mode.
	embMat := func(name string, out, in int) (weightMat, error) {
		return streamMat(name, out, in, quant.embedding(), func(r int) int { return r * in })
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
	// permMat loads a q/k projection and inverts llama.cpp's RoPE row permutation.
	// Because the permutation is a pure reorder of whole rows (output features), it
	// commutes with per-row quantization: rather than permute an f32 buffer then
	// quantize, we dequant the rows straight into HF order — destination row hfRow
	// pulls from GGUF source row (h*hd + 2*j + s) — and quantize in place. See
	// ggufInvPermute for the index derivation.
	permMat := func(name string, out, in, nHead int) (weightMat, error) {
		hd := out / nHead
		half := hd / 2
		return streamMat(name, out, in, quant, func(hfRow int) int {
			h, rem := hfRow/hd, hfRow%hd
			ggufRow := h*hd + 2*(rem%half) + rem/half
			return ggufRow * in
		})
	}
	// stackedExperts loads a GGUF MoE expert tensor — a 3-D [in, out, nExpert]
	// (fastest-first) blob where each expert occupies a contiguous [out, in]
	// row-major slice — and returns one quantized weightMat per expert.
	stackedExperts := func(name string, out, in, nExpert int) ([]weightMat, error) {
		dims, into, err := g.RowDequantizer(name)
		if err != nil {
			return nil, err
		}
		if len(dims) != 3 || dims[0] != in || dims[1] != out || dims[2] != nExpert {
			return nil, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d, experts=%d]", name, dims, in, out, nExpert)
		}
		// Each expert occupies a contiguous [out, in] row-major slice; stream its
		// rows directly into a per-expert quantized weightMat (no whole-tensor f32).
		res := make([]weightMat, nExpert)
		for e := range nExpert {
			m, err := streamQuantized(out, in, quant, func(r int, dst []float32) error {
				return into((e*out+r)*in, dst)
			})
			if err != nil {
				return nil, err
			}
			res[e] = m
		}
		return res, nil
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
	// Load the layers in parallel: each is independent (its own weightMat slots
	// over the read-only mmap), and the per-tensor dequant + re-quant is the load's
	// cost — fanning it out across cores turns a 12B GGUF's ~2 min load into seconds.
	loadLayer := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		var err error
		if l.PreAttnNorm, err = vec(p+"attn_norm.weight", hidden); err != nil {
			return err
		}
		if l.QProj, err = permMat(p+"attn_q.weight", qDim, hidden, arch.NumHeads); err != nil {
			return err
		}
		if l.KProj, err = permMat(p+"attn_k.weight", kvDim, hidden, arch.NumKVHeads); err != nil {
			return err
		}
		if l.VProj, err = mat(p+"attn_v.weight", kvDim, hidden); err != nil {
			return err
		}
		if l.OProj, err = mat(p+"attn_output.weight", hidden, qDim); err != nil {
			return err
		}
		// QK-norm (Mellum, Qwen3): per-head RMSNorm over head_dim, before RoPE.
		// llama.cpp permutes the q/k weights for its RoPE, so the matching
		// per-head-dim norm weights are un-permuted the same way.
		if arch.QKNorm {
			qn, kerr := vec(p+"attn_q_norm.weight", hd)
			if kerr != nil {
				return kerr
			}
			kn, kerr := vec(p+"attn_k_norm.weight", hd)
			if kerr != nil {
				return kerr
			}
			l.QNorm, l.KNorm = ggufInvPermuteVec(qn), ggufInvPermuteVec(kn)
		}
		if l.PreMLPNorm, err = vec(p+"ffn_norm.weight", hidden); err != nil {
			return err
		}
		if arch.MoE != nil {
			// Sparse MoE (Mellum): router + stacked per-expert SwiGLU at the
			// narrower moe_intermediate_size.
			expInter := arch.MoE.IntermediateDim
			if l.Router, err = mat(p+"ffn_gate_inp.weight", arch.MoE.NumExperts, hidden); err != nil {
				return err
			}
			gate, gerr := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, arch.MoE.NumExperts)
			up, uerr := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, arch.MoE.NumExperts)
			down, derr := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, arch.MoE.NumExperts)
			if gerr != nil || uerr != nil || derr != nil {
				return fmt.Errorf("decoder(gguf): mellum experts layer %d: %v / %v / %v", i, gerr, uerr, derr)
			}
			l.Experts = make([]expertWeights, arch.MoE.NumExperts)
			for e := range arch.MoE.NumExperts {
				l.Experts[e] = expertWeights{Gate: gate[e], Up: up[e], Down: down[e]}
			}
			return nil
		}
		if l.GateProj, err = mat(p+"ffn_gate.weight", arch.IntermediateDim, hidden); err != nil {
			return err
		}
		if l.UpProj, err = mat(p+"ffn_up.weight", arch.IntermediateDim, hidden); err != nil {
			return err
		}
		if l.DownProj, err = mat(p+"ffn_down.weight", hidden, arch.IntermediateDim); err != nil {
			return err
		}
		return nil
	}
	if err := parallelLayers(arch.NumLayers, loadLayer); err != nil {
		return nil, err
	}
	return w, nil
}

// ggufInvPermuteVec inverts llama.cpp's q/k RoPE permutation for a single
// head_dim-wide vector (a QK-norm weight, shared across heads): HF position
// (s*half + j) comes from GGUF position (2*j + s). Mirrors ggufInvPermute with
// nHead=1 and no input dimension.
func ggufInvPermuteVec(v []float32) []float32 {
	hd := len(v)
	half := hd / 2
	res := make([]float32, hd)
	for s := 0; s < 2; s++ {
		for j := 0; j < half; j++ {
			res[s*half+j] = v[2*j+s]
		}
	}
	return res
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
