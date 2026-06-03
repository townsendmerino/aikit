# Milestone G2 — Qwen3 dense family (a second model runs)

Parent plan: [`docs/multi-model-plan.md`](../multi-model-plan.md) §3, §4, §7 (G2).
Touches: `decoder/registry.go` (qwen3 adapter), `weights.go` (per-family tensor
schema + untied head), `config.go` (qwen3 validation), `model.go` (untied head).

Status: **DONE & validated on Linux 2026-06-02.** Qwen3-1.7B runs through the
**same generic forward pass** as Gemma — next-token logits match the HF float32
oracle at **cosine 0.999999999999** (argmax `' Paris'`). The Gemma M1–M9 goldens
are byte-identical (regression intact). **Qwen3 dense is the demo target** for a
real coding-capable model (pending G3, the tokenizer).

## What it proves

The G0 descriptor abstraction holds: adding a whole new family was **descriptor
population + a tensor schema + an adapter** — *zero* changes to the forward
math. Qwen3 flips a different set of knobs than Gemma:

| knob | Gemma 3 | Qwen3 |
|---|---|---|
| RMS (1+w) | yes | **no** (`RMSAddOne=false`) |
| norm placement | Sandwich4 (4 norms) | **Pre2** (2 norms) |
| activation | GeGLU (gelu-tanh) | **SwiGLU (silu)** |
| embed scale | √hidden | **none** |
| attn scale | query_pre_attn_scalar | **1/√head_dim** |
| RoPE | dual base | **single base** |
| QK-norm | yes | yes (already supported) |
| LM head | tied | **untied** (`lm_head.weight`) |
| sharded | no (270M) | **yes** (2 shards → uses G1) |

None of these needed new attention/MLP code — QK-norm came free from G0.

## What changed

- **Per-family tensor schema** (`weights.go`): `gemma3TensorSchema` became a
  typed `tensorSchema`; added `qwen3TensorSchema`. Empty suffix = absent tensor,
  so the loader skips Pre2's missing Post*Norms and loads the untied `lm_head`.
  The **same HF tensor name means different roles per family** — Gemma's
  `post_attention_layernorm` is a post-attn norm; Qwen's is the pre-MLP norm —
  which is exactly why the schema is per-family.
- **Untied LM head** (`weights.go`/`model.go`): `Weights.LMHead`; the loader
  finalizes `TiedLMHead` from `lm_head.weight` *presence* (robust to checkpoints
  that tie despite the family default); `forward` uses the head accordingly.
  `LMHead` is also int8-quantizable (M8).
- **qwen3 adapter + `validateQwen3`** (`registry.go`/`config.go`); registry maps
  `model_type: "qwen3"`. Added `Config.HiddenAct` (Qwen uses `hidden_act`, Gemma
  uses `hidden_activation`).
- The generic `runLayers` Pre2 branch and `gatedMLP` SwiGLU branch (both from
  G0) are now actually exercised.

## Validation

- `decoder/qwen3_test.go` (`TestQwen3_forwardParity`, `-short`-gated): loads the
  real 2-shard Qwen3-1.7B, feeds the golden's **HF token ids** (Go tokenizer is
  G3), and matches the float32 oracle — argmax identical, sample/top-k ≤ 5e-3,
  full cosine ≥ 1 − 1e-4 (got 1 − 1e-12). Asserts the arch resolved to `qwen3`
  with an untied head.
- `scripts/pin_qwen3_forward.py` is the oracle (committed
  `testdata/qwen3_forward_golden.json`; full dump gitignored).
- Full decoder suite (19 tests) green: Gemma goldens unchanged.

## Next

- **G3 — byte-level BPE tokenizer** (Qwen/Llama-3 family): the last piece before
  `gemma-web --model qwen3-…` chats in the GUI in pure Go. The M2 merge
  machinery is reusable; the pre/post-processing (GPT-2 split regex, `Ġ` spaces)
  differs. Until then the forward is parity-proven but needs HF-supplied ids.
- Qwen2 is then a small delta (same schema minus QK-norm, plus QKV bias).
