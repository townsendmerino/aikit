# Task: goinfer → aikit kernel moves (2026-09)

**Premise.** aikit owns kernels (raw-bit gated, asm allowed, no model names); goinfer owns model
semantics. Five things in goinfer sit on the wrong side of that line. Each moves the way A1's
attention kernels and the W4A8 tile moved (v1.25.0 / v1.26.0): land in aikit with a reference
implementation and a raw-bit gate, tag, then bump goinfer, swap the call site, delete the local
copy, and prove goinfer's goldens did not move. **Never the other order**, and goinfer never ships
against an untagged aikit.

## Summary

| move | aikit tag | goinfer commit | before/after cell | gate |
|---|---|---|---|---|
| **M5** duplicate inventory | *(none — no aikit change)* | *(see below)* | n/a — not a kernel swap | `TestZZM5_*` probes, mutation-checked |
| M1 fusedattn | *pending* | — | — | — |
| M2 MXFP4 | *pending* | — | — | — |
| M3 W4A8 device kernels | *pending* | — | — | — |
| M4 sequence mixers | *Phase 0 only, no move* | — | — | — |

**Status 2026-09-06: M5 done. M1–M4 not started.**

---

## M5 · duplicate inventory — DONE

Walk goinfer for bodies that duplicate an aikit export. **Bit-identity was measured, not read** —
every row below was decided by a raw-bit probe (`math.Float32bits`, never a tolerance), and the
probe was mutation-checked once by perturbing a single lane and confirming it went red.

| goinfer symbol | aikit symbol | bit-identical? | action |
|---|---|---|---|
| `dequantHeads` (`decoder/kvcache.go`) | `linalg.DequantizeRowsInt8Into` | **YES** — measured | **CALL + delete the arithmetic** ✅ done |
| `matvec` (`decoder/deltanet.go`) | `linalg.MatmulBT` | n/a — *already delegates* | keep (allocating wrapper, no duplicated arithmetic) |
| `matvecWM` (`decoder/deltanet.go`) | via `matmul` → linalg | n/a — *already delegates* | keep |
| `fakeQuantInt4("sym")` (`decoder/fakequant.go`) | `QuantizeGroupInt4Row`+`DequantizeRowInt4` | **NO** — 7/256 differ, all signed-zero | keep — see below |
| `fakeQuantInt4("symmse"/"affine")` | *none* | n/a | keep — no aikit equivalent exists |
| `silu` (`decoder/rmsnorm.go`) | `linalg.SiLUF32` | **NO** — 1/18 probes differ | **DECISION** (S-06 step 2) — not taken here |
| `geluErf` (`decoder/rmsnorm.go`) | `linalg.GELUF32` | **NO** — 2/18 differ | **DECISION** — not taken here |
| `geluTanh` (`decoder/rmsnorm.go`) | `linalg.GELUTanhF32` | **NO** — 2/18 differ | **DECISION** — not taken here |
| `softmaxStable` (`decoder/sampler.go`) | *none* (returns `[]float64`) | n/a | keep — different type contract |

### What moved, and what the gate actually proved

`dequantHeads` was a per-KV-head int8 expansion — `dst[o+c] = float32(q[o+c]) * s` — which is
exactly `DequantizeRowsInt8Into` with `rows=nKV`, `cols=headDim`. The two were gated raw-bit equal
over int8's full range **including −128** (the asymmetric extreme), per-head scales spanning max
normal, min normal, denormal, ±0, Inf and NaN, and the tail shapes the cache actually uses
(`8×7`, `3×33`, `5×1`, plus head dims 64/128/256). Mutation check: negating one output lane turned
the gate red with the raw bits printed (`c1880000` vs `41880000`); removing the perturbation
returned it to green. goinfer's nine call sites and their argument order are unchanged; only the
duplicated loop is gone. **No aikit change, therefore no tag** — which is the point of doing M5
first.

### Two rows that look movable and are not

**`fakeQuantInt4("sym")` — differs only in signed zero, and only because of an integer round trip.**
Its own comment calls the `sym` branch "aikit's runtime int4 (the control)", which invites exactly
the swap this inventory exists to check. Measured: **7 of 256 elements differ**, every one of them
`−0` vs `+0`. goinfer computes `float32(q) * s` where `q` came from `math.Round`, so a small negative
input yields `−0.0`; aikit routes the code through an integer nibble (`byte(q+8)`, 8 = zero) and the
sign cannot survive. The values are numerically equal (`−0 == +0`) and the difference is invisible
to any tolerance test — which is why a raw-bit gate is the right instrument and why "it's the
control" was not sufficient grounds to swap. Nor is it a duplicate in the useful sense: the file is
a *scheme comparator* (`sym` / `symmse` / `affine`) for quantizer-quality work, and two of its three
schemes have no aikit counterpart. It stays.

**The elementwise trio is a parity decision, not a move, and this task does not take it.**
`docs/task-simd-audit.md` S-06 step 2 is explicit that swapping goinfer's `silu`/`gelu*` for
aikit's changes bits. The inventory measures *how much*, so the decision is made against a number
rather than a hunch:

- `geluErf(1)`: goinfer `3f57625f`, aikit `3f57625e` — **1 ULP**.
- `geluTanh(-1)`: goinfer `be229e91`, aikit `be229e90` — **1 ULP**.
- `geluErf(-7)` / `geluTanh(-7)`: goinfer keeps a tiny denormal-scale value (`ad1d9a40`,
  `a7280000`), aikit returns `−0` (`80000000`) — a **flush-to-zero** difference at magnitudes where
  the activation is negligible but the bits are not equal.

Under goinfer's bit-identity contract (decode == batched prefill == speculative verify; a reused
prefix == a cold prefill) a 1-ULP change is a goldens regeneration, and that is goinfer's owner's
call. **S-06 step 1** — parallelising the elementwise loops over goinfer's worker pool, identical
bits — is untouched by this and remains available as a goinfer-only change.

### What was deliberately not done in M5

- No aikit code was written or tagged. M5 is an inventory plus one deletion that needed no aikit
  change; that is the whole design of doing it first.
- `quantizeWM` / the `.giw` quantization path were not swept — they are weight-*format* code
  (goinfer semantics), not kernels, so they are out of the premise rather than merely unfinished.
- Audit P-08's `WeightMat`/W8A8 precision question is listed in the audit and is **a precision
  decision, not a duplicate**; it is not taken here.
