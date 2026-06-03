# Milestone G2c — Mistral family (all-layer sliding window)

Parent plan: [`docs/multi-model-plan.md`](../multi-model-plan.md) §3 (the sliding
window axis), §7 (G2 follow-up). Touches: `decoder/registry.go` (mistral adapter).

Status: **DONE & validated on Linux 2026-06-03.** TinyMistral-248M runs through
the generic forward at **cosine 1−4e-14** vs the HF float32 oracle — over a
**67-token prompt against a 32-token sliding window**, so the last position
genuinely attends to only the trailing 32 tokens. That makes this a real test of
the all-layer sliding-window path, not just the llama-equivalent forward.

## What it proves

Mistral is the **llama descriptor with sliding-window attention on every layer**
— and the window machinery already existed (Gemma's M5: `KVCache.WindowStart`,
the `layerIsGlobal` per-layer hook). Gemma uses a 5:1 local:global pattern;
Mistral is all-local. So the adapter is just llama + `layerIsGlobal = always
local` + the window size. Zero new forward code.

`WindowStart(pos) = pos - window + 1` (clamped) is exactly Mistral's convention
(token i attends to `(i-window, i]`), which is why the existing Gemma-validated
code matched HF Mistral to 1e-14 without change.

## What changed

- **mistral adapter** (`registry.go`) + registry `model_type: "mistral"`. Reuses
  `validateLlama` and `parseRopeScaling`. Sets `SlidingWindow = cfg.sliding_window`
  and `layerIsGlobal = func() bool { return false }` (all-local) when a window is
  configured; a null/0 window (Mistral-v0.2+) falls back to full attention.
- Shares `llamaTensorSchema` (Mistral and Llama have identical tensor names and
  no QK-norm / no bias).

## Validation

- `decoder/mistral_test.go` (`-short`-gated): asserts arch `mistral`, a positive
  window, and that **every layer is local**; requires the prompt to exceed the
  window; then matches the float32 oracle (cosine 1−4e-14).
- `decoder/llama_test.go`: `TestResolveArchitecture_mistral` (all-local + window,
  and the full-attention fallback when sliding_window=0).
- Oracle: `PIN_PROMPT="<long text>" scripts/pin_llama_forward.py testdata/tinymistral-248m mistral`.

## Next

- **Mistral-7B/Mixtral**: 7B is the same dense code at scale; Mixtral adds the
  G6 MoE FFN.
