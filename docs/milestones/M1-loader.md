# Milestone M1 — Gemma 3 loader (BF16 + config + shape-validated weights)

Parent plan: [`docs/gemma-decoder-plan.md`](../gemma-decoder-plan.md) §2–§4, §6 (M0/M1), §10.
Scaffold this fills in: `decoder/weights.go`, `embed/safetensors.go`.

## Goal

Load a real **Gemma 3 270M** checkpoint (`config.json` + `model.safetensors`)
from disk into a fully-populated `decoder.Weights`, with every tensor
shape-validated against `Config`, and prove the bytes are correct against a
pinned golden. No forward pass — just "the weights are in memory and provably
right." This unblocks M3 (the forward pass).

## Why this is the first coding milestone

It has zero dependency on the §1 linalg refactor or the tokenizer, it is
unit-testable in isolation, and a wrong loader silently poisons every
milestone after it. Get the bytes right first.

## Scope (do)

1. **M0 prerequisite — parity oracle.** Add `scripts/pin_gemma.py` (mirror
   `scripts/pin_encoder.py`): load `google/gemma-3-270m` via HF
   `transformers`, dump to `testdata/gemma_golden.json`:
   - the full `config.json` it parsed,
   - for a fixed handful of tensors (e.g. `model.embed_tokens.weight`,
     `model.layers.0.self_attn.q_proj.weight`, `model.norm.weight`): shape,
     dtype, and a float64 checksum (sum + sum-of-squares) computed **after**
     widening bf16→f32, so the Go side can reproduce it exactly.
2. **BF16/F16 decode in `embed/safetensors.go`.**
   - `func (t Tensor) BFloat16sToF32() ([]float32, error)` — bf16 is the top
     16 bits of f32, so widen by `math.Float32frombits(uint32(lohi)<<16)` per
     element (exact; NaN/Inf/subnormals come along for free). Guard `DType ==
     "BF16"` and `len(raw)%2 == 0`.
   - `func (t Tensor) Float16sToF32() ([]float32, error)` — real f16→f32
     (5-bit exponent rebias + subnormal handling). Guard `DType == "F16"`.
   - Unit-test both against a tiny hand-built byte slice with known values
     (incl. a negative, a zero, and one subnormal for f16).
3. **`decoder.LoadWeights` / `LoadWeightsFromFS`.** Replace the stubs:
   - `loadConfig` + `ValidateAssumptions` (already implemented),
   - open safetensors via `embed.OpenSafetensorsMmap` (mmap path, like
     `encoder.LoadWeights`),
   - decode every tensor in `gemma3TensorSchema` (already in `weights.go`) to
     f32 with a shape check, using a `loadF32` helper that dispatches on
     `DType` (F32 passthrough, BF16/F16 widen) — model after
     `encoder.loadF32`. One clear `tensor %q shape %v != want %v` error.
   - Populate `Weights{Embed, FinalNorm, Layers[...]}`; retain `st` so the
     alias-backed slices stay valid (same lifetime contract as
     `encoder.Weights`).
   - **Decision to make & document:** widen-on-load (simple, 2× RAM) vs keep
     bf16 resident. For M1 do widen-on-load; leave a `// TODO(M8)` note.

## Out of scope (don't)

Forward pass, RoPE tables, attention, tokenizer, sampler, the linalg refactor.

## Acceptance criteria

- [ ] `go build ./...` and `go vet ./...` clean.
- [ ] `TestBFloat16sToF32` / `TestFloat16sToF32` pass against hand-built bytes.
- [ ] New `decoder/weights_test.go`: with the 270M checkpoint present, every
      tensor loads with the expected shape and the checksums for the sampled
      tensors match `testdata/gemma_golden.json` to ≤ 1e-6 relative.
- [ ] The test **skips cleanly** when the checkpoint/golden are absent (the
      `encoder` convention — a fresh checkout stays green).
- [ ] `ValidateAssumptions` rejects a config with `final_logit_softcapping`
      set (covered by a table test with a synthetic config).

## Pointers

- Reference loader: `encoder/weights.go` (`buildWeightsFromSafetensors`,
  `loadF32`, `shapeEqual`) and `encoder/model.go` (`Load`).
- Reference parity script: `scripts/pin_encoder.py`.
- Tensor key schema is already encoded in `decoder/weights.go`
  (`gemma3TensorSchema`, `tensorName`).
- Get the checkpoint:
  `huggingface-cli download google/gemma-3-270m --local-dir testdata/gemma-3-270m`.
