# Milestone G2b — Qwen2 / Qwen2.5 family (QKV bias)

Parent plan: [`docs/multi-model-plan.md`](../multi-model-plan.md) §3 (the QKV-bias
axis), §7 (G2 follow-up). Touches: `decoder/arch.go` (`QKVBias` knob),
`weights.go` (bias tensors + `qwen2TensorSchema`), `attention.go` (`addBias`),
`config.go` (`validateQwen2`, `use_sliding_window`), `registry.go` (qwen2 adapter).

Status: **DONE & validated on Linux 2026-06-03.** Qwen2.5-0.5B-Instruct runs
through the generic forward at **cosine 1−1e-12** vs the HF float32 oracle
(argmax `' Paris'`), the Go byte-level tokenizer reproduces its ids, and it
generates end-to-end (`demo/gemma --model testdata/qwen2.5-0.5b`). All prior
goldens unchanged.

## What it proves

Qwen2 is the **llama descriptor + one knob**: an additive bias on the q/k/v
projections. Adding it was a knob (`QKVBias`) + three tensors in the schema +
one `addBias` call — no change to the attention algorithm. This was the last
"cheap win" axis from the plan's §3 table (the Llama/Mistral/Qwen2 bucket).

| knob | Llama | Qwen2 | Qwen3 |
|---|---|---|---|
| QK-norm | no | no | **yes** |
| QKV bias | no | **yes** | no |
| head_dim | derived | derived | explicit |
| RoPE | single (+llama3 scaling) | single | single |

So Qwen2 sits exactly between Llama (no bias, no QK-norm) and Qwen3 (no bias,
QK-norm) — three families, one forward pass.

## What changed

- **`QKVBias` descriptor knob** (`arch.go`) + **`QBias/KBias/VBias`** in
  `LayerWeights` and **`QBias/KBias/VBias` suffixes** in `tensorSchema`. Empty
  suffix = no bias, so every other family is unaffected.
- **Loader** (`weights.go`): loads the three `[out]` bias vectors when the schema
  names them, shape-checked against qDim/kvDim. Biases stay f32 (not matmul
  weights), so int8 quant (M8) is unaffected — the bias adds to the f32 matmul
  output either way.
- **`addBias`** (`attention.go`): elementwise add after the q/k/v projection,
  gated on `arch.QKVBias`, before QK-norm/RoPE.
- **qwen2 adapter + `validateQwen2`** + registry `model_type: "qwen2"`. Reuses
  `validateLlama` and rejects `use_sliding_window=true` (Qwen2's optional
  sliding window is a follow-up; the common checkpoints leave it off).

## Validation

- `decoder/qwen2_test.go` (`-short`-gated): loads real Qwen2.5-0.5B, asserts the
  arch resolved `qwen2` with `QKVBias` and no QK-norm and that the biases
  loaded, checks the Go tokenizer reproduces the HF ids, then matches the
  float32 oracle — argmax identical, sample/top-k ≤ 5e-3, cosine 1−1e-12.
- `decoder/llama_test.go`: `TestResolveArchitecture_qwen2` (descriptor + schema +
  sliding-window rejection) and `TestAddBias`.
- `scripts/pin_llama_forward.py` is family-agnostic now — `… testdata/qwen2.5-0.5b qwen2`
  writes the committed `testdata/qwen2_forward_golden.json`.

## Next

- **Mistral**: llama + sliding-window (the KV-cache window machinery already
  exists from Gemma's M5); enable `use_sliding_window` / `sliding_window` on the
  llama path.
- **Qwen2-MoE / Qwen3-MoE**: a separate `model_type` and the G6 MoE FFN.
