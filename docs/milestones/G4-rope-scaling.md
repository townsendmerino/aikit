# Milestone G4 — RoPE scaling + partial rotary (long-context Llama runs)

Parent plan: [`docs/multi-model-plan.md`](../multi-model-plan.md) §3, §7 (G4).
Touches: `decoder/ropescale.go` (new — scaling math), `arch.go` (RotaryDim +
precomputed inv-freq tables), `rope.go` (`applyRoPE` takes the table), `config.go`
(`rope_scaling` / `partial_rotary_factor` parse), `registry.go` (llama adapter
wires scaling).

Status: **DONE & validated on Linux 2026-06-03.** Llama-3.2-1B — which uses
`llama3` rope_scaling (factor 32) — runs through the generic forward at **cosine
1−8e-13** vs the HF float32 oracle (argmax `' Paris'`). This is also the first
**full Llama-3 pure-Go end-to-end**: the byte-level BPE tokenizer (G3) produces
HF-identical ids, then the scaled-RoPE forward reproduces the logits. Gemma /
Qwen3 / TinyLlama goldens are byte-identical (the refactor moved no numbers).

## What it proves

RoPE frequency scaling is a **descriptor knob baked into a precomputed inv-freq
table**, not new forward code. The adapter resolves `rope_scaling` into the
table once at load; `applyRoPE` reads it. The same seam carries Gemma's dual
base, Llama-3's llama3 scaling, and (the knob, not yet a family) Phi's partial
rotary.

## What changed

- **`ropescale.go`** (new): `parseRopeScaling` decodes HF's `rope_scaling`
  object (accepts both the `rope_type` and legacy `type` keys);
  `computeInvFreq(base, rotaryDim, scaling)` builds `base^(-2d/rotaryDim)` and
  applies the transform. Implemented: **linear** (inv_freq /= factor) and
  **llama3** (HF's piecewise wavelength interpolation — high freqs kept, low
  freqs / factor, middle smoothly blended). Unsupported types (yarn, longrope,
  dynamic) are a **loud error**, not a silent wrong load.
- **`arch.go`**: added `RotaryDim` and an unexported `ropeScaling`; `finalizeRoPE`
  (called by `resolveArchitecture` for every family) precomputes the local/global
  inv-freq tables; `ropeInvFreq(layer)` returns the right one. So pow/scaling run
  **once at load**, not per token (also resolves an M7 TODO).
- **`rope.go`**: `applyRoPE` now takes the precomputed `invFreq []float64` and
  rotates `2*len(invFreq)` dims — full head_dim by default, fewer for partial
  rotary (trailing dims pass through). Math is unchanged for full rotary.
- **`config.go`**: `rope_scaling` parsed (was a blanket-reject guard in G2's
  llama adapter); added `partial_rotary_factor` + `Config.rotaryDim()`. The
  llama adapter drops the "reject scaled RoPE" guard and wires the parsed
  scaling + rotary dim into the descriptor.

## Validation

- `decoder/llama32_forward_test.go` (`-short`-gated): loads real Llama-3.2-1B,
  asserts the arch resolved llama3 scaling + a tied head, **checks the Go
  byte-level tokenizer reproduces the golden's HF ids**, then matches the
  float32 oracle — argmax identical, sample/top-k ≤ 5e-3, cosine 1−8e-13.
- `decoder/ropescale_test.go`: `computeInvFreq` with no scaling equals the
  classic `base^(-2d/dim)` table (refactor is numerically inert); linear divides
  by factor; llama3 matches an independent reimpl over the real params and
  exercises all three branches; partial rotary leaves the trailing dims untouched.
- `scripts/pin_llama_forward.py testdata/llama3.2-1b llama32` is the oracle
  (committed `testdata/llama32_forward_golden.json`; full dump gitignored).
- Full suite green: Gemma/Qwen3/TinyLlama forward + sliding-window goldens
  unchanged.

## Next

- **YaRN / longrope / dynamic NTK** scaling (currently rejected) — for the
  Phi-3-128k and some long-context Qwen/Yi checkpoints. Same `computeInvFreq`
  seam; YaRN also needs an attention temperature (mscale).
- **Phi family adapter** (G5-ish): `partial_rotary_factor` is implemented and
  unit-tested at the RoPE layer, but Phi also brings QKV bias, a fused
  `gate_up_proj`, and (Phi-2) parallel blocks / LayerNorm — a family adapter, not
  just a knob.
- **Qwen2** (`attention_bias` q/k/v/o bias) and **Mistral** (llama + sliding
  window) remain small deltas on the existing schema.
