# Milestone M8 — int8 weight quantization

Parent plan: [`docs/gemma-decoder-plan.md`](../gemma-decoder-plan.md) §8, §6 (M8).
Touches: `internal/linalg/quant.go`, `decoder/weightmat.go`, `weights.go`,
`model.go`, `attention.go`, `mlp.go`, `demo/gemma/main.go`.

Status: **DONE (int8) & validated on Linux 2026-06-02.** `--quant int8` runs the
270M generating coherent text; the int8 forward keeps the f32 argmax (`' Paris'`)
at **cosine 0.9996**, and the matmul weight footprint drops **3.98×** (270 MB
vs 1072 MB f32). int4 group-quant is left as a documented follow-up.

## What was implemented

- **`internal/linalg/quant.go`** — per-row symmetric int8 (`QuantizeRowsInt8`,
  `DequantizeRowInt8`) and a weight-only int8 matmul `MatmulBTQ8` (int8 weights,
  f32 activations, int8→f32 widen in the inner loop, scaled per row, parallel
  over N — sharing the `parallelCols` helper with the f32 `MatmulBT`).
- **`weightMat`** (`decoder/weightmat.go`) — a weight matrix that is f32 *or*
  per-row int8, hiding the precision behind `matmul` (used at all 8 projection
  sites + the tied LM head) and `embedRow` (the embedding lookup dequantizes one
  row on the fly when int8). The forward pass is precision-agnostic.
- **`Weights.quantizeInt8`** converts every matmul matrix (the 7 projections per
  layer + the tied embedding — ~170M of the 270M params live in the embedding)
  to int8 in place, freeing the f32 backing.
- **`Options.Quant`** ("" | "int8") + a `--quant` demo flag.

## Validation

- `internal/linalg/quant_test.go`: round-trip rel-L2 ≤ 1e-2; `MatmulBTQ8` within
  2e-2 rel-L2 of the exact f32 matmul.
- `decoder/quant_test.go` (`TestQuantInt8_accuracy`): with the real checkpoint
  loaded int8 — argmax unchanged vs the f32 oracle, full-logit cosine ≥ 0.999
  (got 0.9996), Embed's f32 backing freed, and the weight footprint ≥ 3.5×
  smaller (measured 3.98×).
- `--quant int8` demo: `"The capital of France is Paris. It is the most visited
  city in the world. …"` — coherent.
- The f32 path is byte-identical after the `weightMat` refactor (M3/M4/M5 parity
  tests unchanged).

## Honest limitations

- **Peak RSS during load** — FIXED by streaming quantize-at-load (see the
  follow-ups). The loader now widens *one* tensor to f32, quantizes it to int8,
  and frees the f32 before the next tensor, so the transient footprint is the
  int8 model + one tensor's f32 rather than the whole checkpoint in f32. This
  also covers the GGUF path (a quantized `.gguf` lands resident as int8 without
  materializing the whole model in f32 first). The retained-weight win (3.98×)
  is unchanged.
- **Speed:** the quantized matmuls now widen to a scratch row/group and run the
  SIMD `dotF32` kernel (see the follow-ups), so they match the AVX2/NEON f32
  path rather than the old scalar loop. The multiply is still f32 (widen-then-
  dot); an int8×int8→int32 fixed-point SIMD kernel would cut it further and is
  the remaining speed lever.

## Known follow-ups (not blocking M8)

- **int4 group-quant** (plan §8) — DONE. `Load(…, Quant:"int4")` quantizes the
  projections to group-wise symmetric 4-bit (group 32: a per-group f32 scale,
  two nibbles/byte; `linalg.QuantizeGroupsInt4` + `MatmulBTQ4` dequant-per-tile),
  ~⅛ f32 on the projections (~6.4× there; less overall when a big embedding is
  kept int8). The embedding **and** LM head stay int8 (the `quantMode.embedding`
  policy) — they are the tied head, so 4-bit there flips the argmax; this mirrors
  GGUF Q4_K_M keeping `token_embd`/`output` at Q6_K. Validated by
  `TestGGUF_int4_resident` (TinyLlama 1.1B: argmax preserved, cosine 0.994 vs
  f32 — on par with Q4_K_M). int4 is a big-model tool: on the 270M it is lossy
  enough to move the top token (`TestQuantInt4_accuracy` checks structure +
  footprint + loose correlation only).
- **int4 + int8 SIMD matmuls** — DONE. Both quantized matmuls now widen each
  weight row/group into a reused scratch buffer and run the SIMD `dotF32` kernel
  (AVX2/NEON — the primitive `MatmulBT` uses) over it, applying the scale at
  write-back; only the cheap unpack/widen stays scalar. `MatmulBTQ4` does it per
  group (32) then scales per group; `MatmulBTQ8` widens the whole row (per-row
  scale). On a decode shape (M=1, K=N=2048): int4 **6.7×** (8.3 → 1.2 ms), int8
  **6.9×** (3.0 → 0.43 ms) over the prior scalar loops (`BenchmarkMatmulBTQ{4,8}`
  vs `…Scalar`). Outputs unchanged within float reassociation; decoder quant
  accuracy identical. (int8×int8→int32 fixed-point is a further speed option.)
- **Streaming quantize-at-load** — DONE. `Load(…, Quant:"int8")` quantizes each
  matmul tensor as it is read and frees the f32 immediately, for the safetensors,
  GPT-2, and GGUF paths. No whole-model f32 spike; a quantized `.gguf` loads
  resident as int8. Validated by `TestGGUF_int8_resident` (argmax + cosine vs the
  f32 oracle) and the unchanged `TestQuantInt8_accuracy`.
- **int8 SIMD kernel** (int8×int8→int32) for a speed win, not just memory.
- **Dedup:** `encoder/quant.go` + `linalg_q8.go` still hold the encoder's own
  copy; unifying onto `internal/linalg` is a cleanup (the kernels legitimately
  differ — encoder blocked for M>1, decoder column-parallel for M=1).
