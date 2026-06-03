# Milestone G7 — GGUF (quantized checkpoints run on a laptop)

Parent plan: [`docs/multi-model-plan.md`](../multi-model-plan.md) §7 (G7), §9
(couples with M8 quant). Touches: `embed/gguf.go` (new — container parser +
block dequant), `decoder/gguf.go` (new — metadata→config, name mapping, q/k
un-permutation), `decoder/weights.go` (`.gguf` dispatch in `LoadWeights`).

Status: **DONE & validated on Linux 2026-06-03.** Quantized TinyLlama GGUFs run
through the generic forward and match the f32 oracle: **Q8_0 cosine 0.99996**,
**Q4_0 0.9944**, **Q4_K_M 0.9975** — all argmax `' Paris'`. Q4_K_M is the
single most-downloaded laptop quant, so this is real-world coverage, not a toy.

## What it proves

aikit reads the format the open-weights ecosystem actually ships for local
inference. A GGUF file is self-describing — its metadata carries the whole
architecture config and its tensors carry the weights — so a single `.gguf`
path loads with no separate config.json/safetensors/tokenizer files. The
quantized weights dequantize to f32 on load and feed the *exact same* forward
pass as every other family (G7 is a loader, not a new compute path — it couples
with M8's runtime quant story from the on-disk side).

## What changed

- **`embed/gguf.go`** (new): a GGUF v2/v3 container parser — magic/version,
  the typed metadata key-values (all 13 GGUF value types, incl. arrays), and the
  tensor directory — plus block dequant for the types that matter:
  - **F32 / F16** — trivial.
  - **Q8_0 / Q4_0** — 32-element blocks (f16 scale + int8 / packed nibbles).
  - **Q6_K / Q4_K** — the 256-element K-quant super-blocks (6-bit quants with
    int8 sub-scales; 4-bit quants with 6-bit packed scales/mins), mirroring
    ggml's `dequantize_row_q6_K` / `q4_K` exactly. These are required for *any*
    real 4-bit file: even a "Q4_0" GGUF keeps `output.weight` in Q6_K, and
    Q4_K_M is Q4_K + Q6_K + F32.
  - Typed metadata accessors (`Str`/`Uint`/`Float`).
- **`decoder/gguf.go`** (new): `ggufConfig` builds a `Config` from the
  `llama.*` metadata; `buildWeightsFromGGUF` maps llama.cpp's `blk.N.*` /
  `token_embd` / `output` names and dequantizes each tensor into a `weightMat`.
  The one subtlety is **`ggufInvPermute`**: llama.cpp stores q/k projection rows
  in interleaved-RoPE order, so they're un-permuted back to this package's
  HF-convention rotate_half layout (verified row-for-row against the original HF
  f32 weights — residual = pure Q8_0 quant error). EOS comes from
  `tokenizer.ggml.eos_token_id`.
- **`LoadWeights`** dispatches a `.gguf` path to the GGUF loader; everything
  downstream (the descriptor, forward, sampler, demos) is unchanged.

## Validation

- `decoder/gguf_forward_test.go` (`-short`-gated): loads TinyLlama Q8_0 / Q4_0 /
  Q4_K_M GGUFs and compares to the committed f32 llama golden — argmax must
  still match, cosine clears a per-type floor (Q8_0 ≥0.999; 4-bit ≥0.99).
- `embed/gguf_test.go`: hand-built Q8_0 / Q4_0 block dequant (exact), and a real
  header/metadata/Q6_K smoke test pinned to the Python `gguf` reference.
- `decoder/gguf_unit_test.go`: the q/k un-permutation on a known case + a
  bijection check.
- The Go Q8_0 dequant was confirmed bit-identical to Python `gguf` before wiring
  the forward.

## Scope / next

- **Implemented quant types**: F32, F16, Q8_0, Q4_0, Q4_K, Q6_K — covers Q8_0,
  Q4_0, and Q4_K_M files end-to-end. Q5_K / Q3_K / Q2_K / IQ* are more block
  formats on the same seam (each a `dequant*` + a size entry).
- **GGUF tokenizer**: the vocab/merges live in metadata; wiring them would let a
  `.gguf` chat with no sidecar tokenizer (today the parity tests feed ids).
- **Other GGUF architectures** (qwen2/gemma/…): map their metadata keys + names;
  the per-family descriptors already exist.
- **GPTQ / AWQ** (safetensors-resident int4): the other half of the plan's G7 —
  different packing (`qweight`/`qzeros`/`scales`/`g_idx`), same dequant-to-f32
  idea, and our safetensors loader already handles the container.
- **Memory**: dequant-to-f32 on load loses the quant's RAM win; pairing GGUF
  with M8-style resident int8/int4 (dequant per-tile in matmul) is the way to
  actually run the big quantized models in small memory.
