# Milestone G5 — GPT-2 (LayerNorm + learned positions + non-gated MLP)

Parent plan: [`docs/multi-model-plan.md`](../multi-model-plan.md) §3, §7 (G5).
Touches: `decoder/arch.go` (knobs), `rmsnorm.go` (`layerNorm`), `mlp.go`
(`nonGatedMLP`), `model.go` (`normalize` dispatch + learned-pos embed),
`attention.go` (skip-RoPE + output bias), `config.go` (`validateGPT2` + n_* keys),
`registry.go` (gpt2 adapter), `weights.go` (`buildGPT2Weights` + Conv1D transpose),
`tokenizer/sentencepiece.go` (accept GPT-2's typeless tokenizer.json).

Status: **DONE & validated on Linux 2026-06-03.** GPT-2 small runs through the
generic forward at **cosine 1−7e-14** vs the HF float32 oracle (argmax `' the'`),
the Go byte-level tokenizer reproduces its ids, and it generates end-to-end
(`demo/gemma --model testdata/gpt2` → "The capital of France is the capital of the
French Republic…"). All prior goldens unchanged.

## What it proves

GPT-2 is the first family that genuinely breaks the Llama mold — it's the
"non-RoPE, non-RMSNorm, non-gated" class. Five new axes, all expressed as
descriptor knobs feeding the SAME forward pass:

| axis | Llama/Qwen/Gemma | GPT-2 |
|---|---|---|
| norm | RMSNorm | **LayerNorm** (mean-centered, weight + bias) |
| positions | RoPE | **learned absolute** (wpe), RoPE skipped |
| MLP | gated (Ge/SwiGLU) | **non-gated** up→gelu→down, with biases |
| attention bias | none / q,k,v | **q,k,v AND output** bias |
| weight layout | Linear `[out,in]` | **Conv1D `[in,out]`** (transposed on load) |

The forward pass stayed singular: `normalize()` dispatches RMS vs LayerNorm,
`mlp()` dispatches gated vs non-gated, attention gates RoPE on `LearnedPosEmbed`
and adds an output bias on `OutBias`. GPT-2's activation `gelu_new` is exactly
the tanh GELU Gemma already had, so no new activation.

## What changed

- **`layerNorm`** (`rmsnorm.go`): mean-centered normalize with weight + bias,
  float64 accumulation (rmsNorm's parity discipline). `normalize()` in
  `model.go` picks it by `arch.Norm`.
- **Learned positions** (`model.go`): `arch.LearnedPosEmbed` adds `wpe[pos]` at
  embed time; `attention.go` skips RoPE for those families. `finalizeRoPE` no-ops
  (no base).
- **`nonGatedMLP`** (`mlp.go`): up→gelu→down with `UpBias`/`DownBias`.
- **Biases**: `QKVBias` (already from Qwen2) + new `OutBias` on the attention
  output projection; `LayerWeights` gained `OBias`, `UpBias`, `DownBias`, and the
  norm biases (`PreAttnNormBias`/`PreMLPNormBias`, `Weights.FinalNormBias`).
- **`buildGPT2Weights`** (`weights.go`): a dedicated loader, because GPT-2's
  layout doesn't fit the per-suffix schema — the fused `c_attn [hidden,3*hidden]`
  is split into q/k/v thirds, every projection is `conv1DTransposed` from Conv1D
  `[in,out]` to the `[out,in]` the matmul expects, and it loads `wte`/`wpe`/`ln_f`
  under GPT-2's flat `h.N.*` names. `buildWeightsFromSafetensors` dispatches to it
  on `arch.Name == "gpt2"`.
- **gpt2 adapter + `validateGPT2`** + Config's `n_embd`/`n_head`/`n_layer`/
  `n_positions`/`n_inner`/`layer_norm_epsilon` keys; the adapter canonicalizes the
  standard Config dims so `Model.Config()` reports them uniformly.
- **Tokenizer** (`tokenizer/sentencepiece.go`): GPT-2's `tokenizer.json` omits
  `model.type`; accept an empty type (its merges + ByteLevel pipeline are the
  same BPE machinery), so GPT-2 tokenizes in pure Go too.

## Validation

- `decoder/gpt2_test.go` (`-short`-gated): loads real GPT-2 small, asserts the
  five knobs + tied head, checks the Go tokenizer reproduces the HF ids, then
  matches the float32 oracle — argmax identical, sample/top-k ≤ 5e-3, cosine
  1−7e-14.
- `decoder/gpt2_unit_test.go`: `TestResolveArchitecture_gpt2` (descriptor) and
  `TestLayerNorm` (zero-mean/unit-var + affine weight·bias).
- `scripts/pin_llama_forward.py testdata/gpt2 gpt2` is the oracle (committed
  `testdata/gpt2_forward_golden.json`).

## Next

- **GPT-2 medium/large/XL** and **DistilGPT2** are the same code, larger.
- **GPT-NeoX / Pythia / GPT-J**: LayerNorm + (often) parallel attention/MLP
  blocks and rotary — a parallel-residual knob on this base.
- **Phi-2**: parallel blocks + partial rotary (already at the RoPE layer) +
  fused QKV — close to GPT-2 plus rotary.
- **G6 MoE** remains the one structural FFN addition.
