# Milestone M7 — perf: shared SIMD linalg, interactive decode

Parent plan: [`docs/gemma-decoder-plan.md`](../gemma-decoder-plan.md) §1, §6 (M7).
Touches: new `internal/linalg/`, `encoder/linalg.go`, `decoder/backend.go`.

Status: **DONE & validated on Linux 2026-06-02.** Decode is now ~**18 tok/s**
on the Ryzen 7 3700X (270M, 128-token run) — above the plan's >10 tok/s
interactive target, up from ~5 tok/s on the naive backend (~3.5×). All HF
parity tests still pass (identical math, just faster).

## What was done (the §1 refactor, scoped)

- **Shared kernel package `internal/linalg`.** Moved the hand-written dot
  kernels — the AVX2/FMA `dot_amd64.{go,s}`, NEON `dot_arm64.{go,s}`, the scalar
  `dot_generic.go`/`dot_other.go`, the `dot.go` dispatcher, and their tests —
  out of `encoder/` into one package, so there is a **single copy of the
  assembly** shared by the encoder and the Gemma decoder (plan §1's core goal:
  no duplicated SIMD asm). Exposed `Dot`, `Dot4x4`, `Dot8x4` (the register-
  blocked micro-kernels) and a new row-parallel `MatmulBT`.
- **`linalg.MatmulBT`** — `dst[M,N] = a·bᵀ` via the SIMD `Dot` kernel,
  parallelized across the N output columns (the dimension that's always large
  and the only one with parallelism on the M=1 single-token decode path).
  Serial under a MAC threshold so tiny matmuls skip the goroutine fan-out.
- **Encoder** imports the shared kernels (two call sites in `encoder/linalg.go`);
  its public API and cache-blocked matmul driver are unchanged.
- **Decoder** `cpuBackend.MatmulBT` now dispatches to `linalg.MatmulBT` — a
  one-line body swap; the interface and every caller stayed put.

## Validation

- **Parity unchanged.** Every decoder HF-parity test (M3 logits, M4 greedy
  continuation, M5 sliding window) still passes — the math is identical.
- **`internal/linalg/matmul_test.go`** checks `MatmulBT` against a naive
  reference across M=1 / M>1, the LM-head shape, the serial/parallel threshold,
  and non-multiple-of-4 K (the scalar tail).
- **`-race` clean** on `internal/linalg` and `decoder` — the column-parallel
  workers write disjoint `dst` slices.
- **Encoder unaffected** — its dot/matmul/parity tests are green after the move.

## Speedup (270M, Ryzen 7 3700X, 16 threads)

| Test / run                | naive backend | shared linalg | speedup |
|---------------------------|--------------:|--------------:|--------:|
| M5 window (748-tok prefill) | 62.6 s      | 26.8 s        | 2.3×    |
| M4 decode (48 tok)        | 11.3 s        | 3.5 s         | 3.2×    |
| 128-tok generation        | ~5 tok/s      | ~18 tok/s     | 3.5×    |

## Known follow-ups (not blocking M7)

- **Goroutine reuse.** `MatmulBT` spawns a fresh worker set per call (~126/token);
  on the smaller layer projections that fan-out caps the win. A persistent pool
  or fusing Q/K/V (and gate/up) into single matmuls would push further.
- **Unify the matmul driver.** The encoder keeps its own cache-blocked
  `matmulBTInto` (tuned for M>1 batches); the decoder uses the M=1-friendly
  `linalg.MatmulBT`. Sharing one driver is a future cleanup — only the kernels
  are shared today. `parallelThreshold` tuning (cpu-acceleration §B) still open.
- **Not moved:** `SoftmaxRow`/`LayerNorm`/`RMSNorm`/`GeluTanh` (plan §1 listed
  them) — cheap, not the bottleneck, and each package has its own. Optional.
- **GPU (M9)** and **quant (M8)** remain for the larger checkpoints.
