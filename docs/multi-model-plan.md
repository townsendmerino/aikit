# aikit multi-model plan — from "runs Gemma 3" to "runs almost any open decoder LLM"

Status: **proposal**. Companion to [`docs/gemma-decoder-plan.md`](gemma-decoder-plan.md).
That plan got us a faithful, parity-checked Gemma 3 forward pass (M1–M6). This
plan generalizes the `decoder` package so the *same* code path serves Llama,
Mistral, Qwen, Phi, DeepSeek, Yi, SmolLM, and friends — without becoming a pile
of `if family == …` branches.

It runs **parallel to** the M7–M9 perf/quant/GPU track: generalization is about
*which checkpoints load and produce correct logits*; the perf track is about
*how fast*. They meet at the `Backend` seam and the quant work.

---

## 0. Where we are

Today the package is Gemma-3-only, and tightly so:

- `Config.ValidateAssumptions` **rejects** anything that isn't Gemma 3 (only
  `gelu_pytorch_tanh`; soft-capping forbidden).
- `runLayers` hardcodes Gemma's **4-norm "sandwich"** (pre+post norm on both
  attention and MLP), the **×√hidden embedding scale**, **QK-norm always on**,
  **GeGLU**, **tied** LM head, and **dual-base RoPE**.
- `causalAttention` scales by `query_pre_attn_scalar^-0.5` and assumes no QKV
  bias.
- The loader opens a **single `model.safetensors`** — no sharding.
- The tokenizer is Gemma SentencePiece-style: `▁` space normalizer, `Type ==
  "BPE"` only, **hardcoded** `<bos>`/`<eos>`/`<start_of_turn>` lookups.

The key insight: every popular open decoder LLM is the **same skeleton** —
embed → N×(norm, causal-GQA-attention, norm, gated-MLP) → final norm → LM head
— differing only on a bounded set of **knobs**. So generalization is mostly
*parameterization + two infra pieces + (for the MoE class) one new FFN path*.

---

## 1. What already generalizes (no work)

The safetensors loader + BF16/F16 decode, the KV cache, GQA, the sampler,
RoPE rotate_half, RMSNorm/LayerNorm math, and the `Backend` seam are all
family-agnostic. Keep them; build the descriptor on top.

---

## 2. The `Architecture` descriptor

Replace the Gemma-specific assumptions in `Config`/`runLayers` with a struct of
knobs, resolved once at load time from `config.json` (`model_type` /
`architectures`) via a small per-family adapter, then consumed by a single
generic forward pass.

```go
// Architecture is the resolved, family-agnostic description of a decoder
// LLM's structure. One generic forward pass reads it; per-family adapters
// (see §4) populate it from that family's config.json + tensor names.
type Architecture struct {
    // Dims
    HiddenDim, NumLayers, NumHeads, NumKVHeads, HeadDim int
    IntermediateDim, VocabSize, MaxPositions           int

    // Norm
    Norm        NormKind // RMSNorm | LayerNorm
    RMSAddOne   bool     // Gemma's (1+w); false for Llama/Qwen
    NormEps     float64
    NormPlacement NormPlacement // Pre2 (Llama) | Sandwich4 (Gemma)

    // MLP
    Act    ActKind // SiLU(SwiGLU) | GeluTanh(GeGLU) | Gelu | ReLU2
    MoE    *MoEConfig // nil = dense FFN; set = sparse experts (§6)

    // Attention
    QKVBias       bool    // Qwen2, GPT-2
    OutBias       bool
    QKNorm        bool    // Gemma3, Qwen3
    AttnScale     float64 // 0 ⇒ 1/sqrt(HeadDim); else explicit (Gemma scalar)
    SlidingWindow int     // 0 = none; >0 with pattern below
    LayerIsGlobal func(i int) bool // sliding/global per layer (Gemma 5:1; Mistral all-local)

    // RoPE
    RoPE RoPEConfig // base(s), scaling (linear/NTK/YaRN/llama3), rotaryDim fraction, interleaved

    // Embedding / head
    EmbedScale float64 // 0 or 1 = none; Gemma = sqrt(hidden)
    TiedLMHead bool    // tied embeddings vs separate lm_head.weight

    // Output
    FinalLogitSoftcap float64 // Gemma 2 (0 = none)
    AttnLogitSoftcap  float64
}
```

`runLayers` becomes one generic loop that branches on the descriptor:

```
h = embed[id]; if EmbedScale>0 { h *= EmbedScale }
for l in layers:
    res = h
    h = norm(h, preAttnNorm)               // always
    h = attention(l, h, …)                  // GQA, optional QKNorm/bias/sliding/softcap
    if NormPlacement == Sandwich4 { h = norm(h, postAttnNorm) }
    h = res + h
    res = h
    h = norm(h, preMLPNorm)
    h = MoE==nil ? gatedMLP(h, Act) : moeMLP(h, …)
    if NormPlacement == Sandwich4 { h = norm(h, postMLPNorm) }
    h = res + h
h = norm(h, finalNorm)
logits = h · (TiedLMHead ? embed : lmHead)ᵀ
if FinalLogitSoftcap>0 { logits = softcap(logits) }
```

Gemma 3 is then just one descriptor (Sandwich4 + GeluTanh + RMSAddOne +
QKNorm + dual-RoPE + EmbedScale + tied). The existing M1–M6 parity goldens
become the **regression guard** that the refactor didn't change Gemma's
numbers.

---

## 3. The knobs, by axis (and which models flip them)

| Axis | Descriptor field | Gemma 3 | Llama 2/3 | Mistral | Qwen2 | Qwen3 | Phi-3 | GPT-2 |
|---|---|---|---|---|---|---|---|---|
| Norm type | `Norm` | RMS | RMS | RMS | RMS | RMS | RMS | **LayerNorm** |
| RMS (1+w) | `RMSAddOne` | **yes** | no | no | no | no | no | — |
| Norm placement | `NormPlacement` | **Sandwich4** | Pre2 | Pre2 | Pre2 | Pre2 | Pre2 | Pre2 |
| Activation | `Act` | **GeGLU** | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | **GELU (no gate)** |
| Embed scale | `EmbedScale` | **√h** | — | — | — | — | — | — |
| QK-norm | `QKNorm` | **yes** | no | no | no | **yes** | no | no |
| QKV bias | `QKVBias` | no | no | no | **yes** | no | no | **yes** |
| Attn scale | `AttnScale` | scalar | 1/√hd | 1/√hd | 1/√hd | 1/√hd | 1/√hd | 1/√hd |
| LM head | `TiedLMHead` | tied | **untied** (7B+) | untied | tied (small) | tied (small) | untied | tied |
| RoPE | `RoPE` | dual base | single (+llama3 scaling on 3.1+) | single | single | single | single (+su/longrope) | — (learned pos) |
| Sliding window | `SlidingWindow` | 5:1 alt | no | **all** | no | no | no | no |
| FFN | `MoE` | dense | dense | dense (Mixtral=**MoE**) | dense (A14B=**MoE**) | dense (MoE variants) | dense | dense |

Reading the table: **Llama / Mistral / Qwen2 are the cheapest wins** — they
flip only `RMSAddOne=false`, `NormPlacement=Pre2`, `Act=SiLU`,
`EmbedScale=0`, `AttnScale=1/√hd`, plus untied head / QKV bias. No new
algorithms. They're also the largest bucket of community checkpoints.

---

## 4. Per-family adapters (config + tensor names)

Two things vary per family and need a tiny adapter each:

1. **Config field mapping.** Most HF configs share names (`hidden_size`,
   `num_attention_heads`, `num_key_value_heads`, `rope_theta`,
   `rms_norm_eps`), so a base decoder maps 90%. Family quirks
   (`query_pre_attn_scalar`, `rope_local_base_freq`, `rope_scaling{type,factor}`,
   `partial_rotary_factor`, `sliding_window`) are handled in the adapter.

2. **Tensor-name scheme.** q/k/v/o/gate/up/down are standardized across HF
   (`model.layers.N.self_attn.q_proj.weight`, `mlp.gate_proj.weight`), but the
   **norm tensors differ**: Llama has `input_layernorm` + `post_attention_layernorm`
   (2); Gemma adds `pre_feedforward_layernorm` + `post_feedforward_layernorm`
   (4); GPT-2 uses an entirely different `transformer.h.N.*` scheme. So
   `gemma3TensorSchema` generalizes to a per-family `tensorSchema`.

Design: a `registry` keyed by `model_type` (`"gemma3"`, `"llama"`, `"mistral"`,
`"qwen2"`, `"qwen3"`, `"phi3"`, …) → `func(rawConfig) (*Architecture, tensorSchema, error)`.
Unknown `model_type` → a clear "unsupported architecture %q (have: …)" error,
not a silent wrong load. `ValidateAssumptions` stops being "is this Gemma 3"
and becomes "is every knob in this descriptor one the forward pass implements."

---

## 5. Infra gaps (needed regardless of family)

### 5.1 Sharded safetensors — the #1 practical blocker

`LoadWeights` opens one `model.safetensors`. Anything above ~2B params (incl.
Gemma 3 4B/12B/27B, every Llama ≥7B) ships as
`model-00001-of-0000N.safetensors` + a `model.safetensors.index.json`
`weight_map` (tensor name → shard file). Add:

- parse `model.safetensors.index.json` when present (fall back to the single
  file when absent),
- mmap each shard once (extend `embed` to hold N mapped files),
- resolve each tensor name through the weight map to the right shard.

Without this the generalization is academic for any model you'd actually want
to run. Do it first.

### 5.2 Tokenizer families

The current tokenizer is Gemma SentencePiece-style. Two more families cover
nearly everything:

- **Byte-level BPE (GPT-2 / Llama-3 / Qwen / Mistral-v3).** Different from
  Gemma: a **byte-level pretokenizer** with the GPT-2 split regex, no `▁`
  marker (spaces become `Ġ`), `add_prefix_space` semantics. The merge/rank
  machinery you already have is reusable; the pre/post-processing differs.
- **Unigram (T5 / older SP).** Viterbi over piece scores — a different segment
  algorithm.

Also: stop hardcoding `<bos>`/`<start_of_turn>`. Read special-token ids and the
add-BOS/EOS policy from `tokenizer_config.json` / `generation_config.json` so
each model contributes its own. Gate parity per family against HF
`tokenizers` (the M2 discipline, per family).

---

## 6. The one structural addition: MoE

Mixtral, Qwen-MoE (A14B/A3B), DeepSeek-V2/V3, and others replace the dense FFN
with a **router + sparse experts**: a small gate picks top-k of E expert MLPs
per token, runs only those, and combines them weighted by the gate. This is a
genuine new code path (`moeMLP`), not a knob:

- `MoEConfig{NumExperts, TopK, NormTopKProb, SharedExperts}`,
- per-layer expert weights (E× gate/up/down) — big, so the sharded loader (§5.1)
  and quant (M8) matter even more here,
- router matmul → top-k select → run chosen experts → weighted sum.

It unlocks a meaningful slice of frontier open models, but it's optional for
"most" models — sequence it after the dense families land.

---

## 7. Milestones

Each ends green with a per-family logit/greedy golden (the M3/M4 discipline,
new fixture per family). Parallel to M7–M9.

- **G0 — refactor to the descriptor.** ✅ **DONE 2026-06-02.** `Architecture`
  descriptor + registry (`model_type` → adapter) + generic
  `runLayers`/`forward`/`attention`/`gatedMLP`; Gemma 3 is one descriptor. All
  M1–M9 Gemma goldens pass **byte-identical** (cosine unchanged). See
  [`milestones/G0-descriptor.md`](milestones/G0-descriptor.md).
- **G1 — sharded safetensors loader** (§5.1). Acceptance: load a multi-shard
  checkpoint (e.g. Gemma 3 4B) and reproduce M1-style tensor checksums.
- **G2 — Llama / Mistral / Qwen2 family.** Knobs: RMS no-offset, Pre2 norms,
  SwiGLU, no embed scale, 1/√hd scale, untied head, optional QKV bias, optional
  all-sliding (Mistral). Acceptance: logit parity vs HF for one checkpoint each
  (e.g. Llama-3.2-1B, Mistral-7B-v0.3, Qwen2.5-0.5B).
- **G3 — byte-level BPE tokenizer** (§5.2). Acceptance: HF-exact id parity for
  Llama-3 and Qwen tokenizers.
- **G4 — RoPE scaling + partial rotary + QK-norm-on-others.** llama3 / linear /
  NTK / YaRN scaling; `partial_rotary_factor` (Phi); Qwen3 QK-norm. Acceptance:
  long-context Llama-3.1 and Phi-3 parity.
- **G5 — LayerNorm + non-gated MLP** (GPT-2/NeoX class) and untie the last
  Gemma-only assumptions. Acceptance: GPT-2 small parity.
- **G6 — MoE** (§6). Acceptance: Mixtral or Qwen-MoE greedy parity.
- **G7 — quantized weights** (GGUF / GPTQ / AWQ): the format that makes the big
  ones runnable on a laptop. Couples with M8.

Order rationale: **G0→G1→G2→G3 unlocks the majority of community checkpoints**
(the Llama/Mistral/Qwen universe). G4–G5 mop up the long tail. G6–G7 are the
"truly almost all" stretch.

---

## 8. Honest scope statement

After G0–G5 + G7, aikit runs *almost all* dense open decoder LLMs that fit in
memory (with quant, that's most ≤~14B on a laptop, larger on a workstation).
"All LLMs" is never literally true — closed weights aren't downloadable,
encoder-decoder models (T5, Whisper) and diffusion/SSM/Mamba architectures are
out of this skeleton's scope, and the very largest MoEs (DeepSeek-V3 class) are
a hardware problem, not a code one. The realistic target is **"almost any open,
dense or MoE, decoder-only transformer checkpoint, at a speed set by the
backend (M7) and quant (M8/G7)."**

---

## 9. How this meets the perf track

The descriptor refactor (G0) is independent of M7. Once both land, the generic
forward pass calls the SIMD/parallel/WebGPU `Backend` exactly as the Gemma path
does now — no per-family perf work. Quant (M8) and quantized-format loading
(G7) are the same machinery viewed from two directions (runtime precision vs
on-disk format); build M8's int4 group-quant first, then G7 reads the
equivalent on-disk layout.
