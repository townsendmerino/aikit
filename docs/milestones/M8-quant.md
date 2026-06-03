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

- **Peak RSS during load doesn't drop** (it's slightly higher): the loader
  widens bf16→f32 for the whole checkpoint, *then* quantizes per-matrix, so f32
  and int8 coexist transiently, and Go's GC returns the freed f32 to the OS
  lazily. The win is the **retained weight bytes** (3.98×), realized in steady
  state. A streaming quantize-at-load (widen one tensor → quantize → free,
  before the next) would also cut peak RSS — the follow-up below.
- **Speed:** int8 here is memory-oriented; the multiply is still f32 (int8→f32
  widen in the loop), so it isn't faster than the AVX2 f32 path on the 270M.
  An int8×int8 SIMD kernel (int32 accumulate) is where the speed win lives.

## Known follow-ups (not blocking M8)

- **int4 group-quant** (plan §8) — the real unlock for 1B/4B: group-size 32–128,
  per-group scale, packed nibbles, on the embedding + projections. `MatmulBTQ8`
  and this `weightMat` seam are the template.
- **Streaming quantize-at-load** to cut peak RSS (see above).
- **int8 SIMD kernel** (int8×int8→int32) for a speed win, not just memory.
- **Dedup:** `encoder/quant.go` + `linalg_q8.go` still hold the encoder's own
  copy; unifying onto `internal/linalg` is a cleanup (the kernels legitimately
  differ — encoder blocked for M>1, decoder column-parallel for M=1).
