# Milestone M3 — Gemma 3 forward pass (logit parity, the correctness gate)

Parent plan: [`docs/gemma-decoder-plan.md`](../gemma-decoder-plan.md) §3, §6 (M3), §10.
Scaffold this fills in: `decoder/model.go` (forward), `attention.go`, `mlp.go`,
`rope.go` (new).

Status: **DONE & validated on Linux 2026-06-02.** Next-token logits match the
HF float32 oracle with **cosine = 0.999999999999** (≈1 − 1e-12, vs the 1 − 1e-4
bar) and identical argmax — `' Paris'` for "The capital of France is".

## Goal

One token, bit-faithful. Embed → 18 Gemma 3 blocks → final norm → tied LM head,
asserting the last-position logit vector matches HF. Get the math exactly right
before any optimization or multi-token decode (M4) — a subtle error here poisons
everything downstream.

## What was implemented

- **`decoder/rope.go`** — single-position NeoX RoPE (rotate_half) with the
  per-layer base: local `rope_local_base_freq` (10k) on sliding layers, global
  `rope_theta` (1e6) on full-attention layers.
- **`attention.go` `causalAttention`** — GQA (4 q-heads → 1 kv-head): project
  q/k/v, **QK-norm** (RMSNorm(1+w) over head_dim, before RoPE), RoPE, append to
  the KV cache, scaled-dot-product over the causal range
  (`scale = query_pre_attn_scalar^-0.5`), softmax, value-weighted sum, output
  projection.
- **`mlp.go` `geGLU`** — `down(geluTanh(gate(x)) ⊙ up(x))`, no biases.
- **`model.go`** — `runLayers` (embed×√hidden → sandwich-norm blocks) and
  `forward` (final norm → `logits = h·Embedᵀ`, tied weights, no soft-cap).
  Prefill skips the vocab-sized LM head on all but the last token.

## The bug the parity gate caught

First run: argmax already correct (`' Paris'`) but cosine only **0.994** with a
systematic distortion. Root cause: **`sliding_window_pattern` is `null` in this
checkpoint's config.json**, so the pattern-arithmetic fallback classified *every*
layer as global and applied the 1e6 RoPE base everywhere. Gemma 3 actually
carries the per-layer kind in a **`layer_types`** array (`full_attention` at
layers 5/11/17, `sliding_attention` elsewhere). Parsing `layer_types` and using
it in `IsGlobalLayer` (config.go) — with a length/value check in
`ValidateAssumptions` — fixed it: cosine 0.994 → 0.999999999999, max sampled
logit Δ 1.9 → 3e-5. Exactly the silent-corruption case the cosine gate exists to
surface; argmax alone would have passed it.

## The oracle

`scripts/pin_gemma_forward.py` runs HF in **float32** (so the reference math is
f32 end-to-end — no bf16-normalizer ambiguity) and writes:
- `testdata/gemma_forward_golden.json` (committed, ~15 KB): prompt, ids, argmax,
  top-32, full-vector stats, a seeded 256-index sample.
- `testdata/gemma_forward_full.json` (gitignored, ~5 MB): every logit, for an
  exact full-vector cosine when present.

## Acceptance criteria — all met

- [x] `go build ./...` / `go vet ./...` / `gofmt` clean; default checkout green
      (M3 test **skips** when checkpoint/golden absent).
- [x] `TestForward_logitParity`: argmax identical; top-k + sampled logits within
      5e-3; sum/sum_sq within 1e-3 rel; full cosine ≥ 1 − 1e-4 (got 1 − 1e-12).
- [x] `ValidateAssumptions` rejects a malformed `layer_types`.

## Known follow-ups (not blocking M3)

- **Perf (M7).** Naive f32 backend (`cpuBackend.MatmulBT`) + per-call RoPE
  recompute. The LM head dominates; ~1.2 s for the 6-token prefill. Wire the
  SIMD/parallel linalg (plan §1) and precompute RoPE tables.
- **Sliding window (M5).** The prompt (L=6) is far below the 512 window, so
  local==causal here. `cache.WindowStart` is wired but unverified past 512.
- **Multi-token decode (M4).** The cache-based single-step path is built and
  exercised on the prefill; M4 adds the greedy-continuation string golden.
