# Milestone G0 — the Architecture descriptor (multi-model foundation)

Parent plan: [`docs/multi-model-plan.md`](../multi-model-plan.md) §2, §4, §7 (G0).
Touches: new `decoder/arch.go`, `decoder/registry.go`; refactors `model.go`,
`attention.go`, `mlp.go`, `rmsnorm.go`, `weights.go`, `config.go`.

Status: **DONE & validated on Linux 2026-06-02.** Pure refactor — the entire
Gemma 3 forward pass now reads a family-agnostic `Architecture` descriptor, and
**every M1–M9 Gemma golden passes with byte-identical numbers** (the regression
guard). No new model loads yet; that's G2+.

## Goal

Replace the hardcoded Gemma assumptions in the forward pass with a struct of
knobs resolved once at load time, so adding Llama/Mistral/Qwen later is
descriptor population, not new forward code. Acceptance (per the plan): the
existing Gemma goldens still pass unchanged.

## What changed

- **`Architecture`** (`arch.go`) — the resolved descriptor: dims, `Norm`/
  `RMSAddOne`/`NormEps`/`NormPlacement`, `Act`, `QKNorm`/`AttnScale`/
  `SlidingWindow`/`layerIsGlobal`, dual-base RoPE, `EmbedScale`, `TiedLMHead`,
  and soft-cap fields. Enums `NormKind`/`NormPlacement`/`ActKind`.
- **Registry** (`registry.go`) — `model_type → adapter`. `gemma3` /
  `gemma3_text` are registered; an unknown `model_type` is a loud
  `"unsupported model_type %q (have: …)"` error, not a silent wrong load. The
  `gemma3` adapter builds the descriptor from `Config` and runs
  `ValidateAssumptions`.
- **Generic forward** — `runLayers`/`forward` read the descriptor: embedding
  scale, norm placement (Gemma's 4-norm `Sandwich4` vs the `Pre2` branch),
  `(1+w)` RMS offset, activation (`gatedMLP` dispatches GeGLU/SwiGLU), tied vs
  untied head, optional final soft-cap. `causalAttention` reads `QKNorm`,
  `AttnScale`, `ropeBase`, and the sliding window from the descriptor.
- **Loader** — `LoadWeights` resolves the architecture via the registry and
  stores it on `Weights`; the tied-embedding/4-norm **tensor schema is still
  Gemma-specific** (generalizing it is G2).

Gemma 3 is now exactly one descriptor + one adapter. The branches the other
families will flip (`Pre2`, `SiLU`, `RMSAddOne=false`, `EmbedScale=0`,
`AttnScale=1/√hd`) exist but are unexercised until G2.

## Validation

- **Parity unchanged (the acceptance gate):** M3 logit cosine
  `0.9999999999997645`, M5 window `0.999999999996809`, M4 greedy continuation,
  M1 checksums, M8 int8, M9 GPU forward — all **identical** to pre-refactor.
- **New unit coverage** for the generic branches Gemma's goldens don't hit:
  `resolveArchitecture` (gemma3 fields + unknown-`model_type` error), `silu` vs
  `x·sigmoid(x)`, and the RMS `(1+w)` vs `w` branch.
- `go build`/`vet`/`gofmt` clean (default + `-tags gpu`).

## Next (multi-model-plan)

- **G1** — sharded safetensors (`model.safetensors.index.json`) for ≥~2B.
- **G2** — Llama/Mistral/Qwen2 adapters + tensor schema + QKV bias + untied head.
- **G3** — byte-level BPE tokenizer (the Qwen/Llama-3 family) for the coding-model
  demo target (Qwen2.5-Coder).
