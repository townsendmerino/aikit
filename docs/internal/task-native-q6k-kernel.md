# task: fused native-GGUF K-quant matmul kernel (Q6_K×Q8_K) — DEFERRED

> Status: **DEFERRED (not started in code).** Drafted 2026-06-13. This is the
> "feature A" spike from the goinfer cross-repo review: let aikit run matmuls
> directly on native GGUF K-quant blocks (start Q6_K) so goinfer can skip the
> dequant→requant-to-int8/int4 step at GGUF load. The two near-free numbers were
> front-loaded (below) and they **narrow the justification** before any kernel was
> written — hence deferred pending the decisions in §7. The convenience win it was
> partly meant to enable (**D — the transparent .giw cache**) is **independent, needs
> no aikit change, and should ship regardless**; do not let this become its critical path.

---

## 1. What it is

A tiled, fused dequant-dot kernel: GEMM directly against GGUF K-quant weight
blocks, with no f32 materialization and no requant. Pays off three ways:
(1) native-GGUF-direct inference; (2) higher-fidelity `.giw` bundles (store native
K-quant blocks, not a reconvert); (3) no requant-at-load for the in-RAM case.

Scope if it proceeds: **Q6_K first**, then (only on a go) the other mid-bit
K-quants **Q5_K, Q4_K_M**. Q8_0≈int8 and Q4_0≈int4 map to existing `WeightMat`
kinds with no fidelity/footprint gain — **out of scope**.

## 2. Front-loaded findings (measured 2026-06-13, before any kernel)

These were measured first precisely because they can stop the spike. Treat the
original "~22% footprint / ~1.4% aggregate fidelity" as **outputs to confirm** — and
they came back weaker than the estimate.

**Footprint (pure arithmetic):**

| storage | bits/weight | vs native Q6_K |
|---|---|---|
| **Q6_K (native)** | **6.5625** (210 B / 256) | — |
| int8-requant | 8.0 | Q6_K is **18% smaller** ✅ |
| int4-requant (group 32) | 5.0 | Q6_K is **31% larger** ❌ |

The footprint win exists **only vs int8**. Against int4-requant, native Q6_K
*costs* ~31% more memory — so vs int4 it can only justify itself on fidelity.

**Requant divergence on a real Q6_K LM head** (`output.weight`, [32000×2048], from
`testdata/tinyllama-gguf/…Q4_K_M.gguf` — the logit-producing, rare-token-critical
tensor), error of each requant vs native Q6_K over 64 random activations:

| requant | relL2 | top-1 flip | top-5 overlap |
|---|---|---|---|
| int8 | 0.012 | 2/64 (3%) | 0.975 |
| int4 (g32) | 0.104 | **15/64 (23%)** | 0.812 |

*Caveat: random Gaussian activations, not real hidden states — absolute flip rates
are overstated (real states concentrate logits). The int8-vs-int4 **contrast** is
the real signal and is directional.* Repro: a throwaway `main` opening the GGUF via
`embed.OpenGGUF`, requanting `output.weight` with `linalg.QuantizeRowsInt8` /
`QuantizeGroupsInt4`, comparing logits over random activations (deleted; ~40 lines).

## 3. Strategic read (why this narrows the case)

- int4-requant **materially** hurts the logit tensor (8× the relL2, ~7× the flip
  rate of int8) — which is exactly why this GGUF ships the head as Q6_K and the body
  as Q4_K. The model's own quant policy says "the head needs > 4 bits."
- **But goinfer already keeps logit-critical tables at int8 in int4 models** (see the
  `WeightMat` doc comment in `linalg/weightmat.go`). int8-requant tracks native closely
  (3% flip, relL2 0.012), with **no new kernel**. So the rare-token fidelity goinfer
  cares about is **mostly already captured** by the selective-int8-head it does today.
- Net, vs the realistic goinfer baseline (int4 body + **int8 head**): native Q6_K's
  marginal win is ~18% footprint on the tables that are int8 today + a small tail
  bonus — i.e. the **native-direct + `.giw`-native-block** case, **not** a
  "fidelity beats int4" or "smaller than int4" case. Matches the honest
  ~1.4%-aggregate framing from goinfer's data.
- ⇒ The fused kernel's real justification is **native-GGUF-direct inference + higher-
  fidelity `.giw` bundles**. The compute number (NEON prefill vs W8A8, §5) only matters
  *if that capability is the goal*. The fidelity-vs-int4 and footprint-vs-int4 arguments
  do not carry it.

## 4. Design (pick-up-ready)

Architecture confirmed against the real code (2026-06-13):

- **`linalg` and `embed` are independent siblings** (neither imports the other; `linalg`
  is stdlib-only). So the fused kernel + Q6_K block-unpack live **in `linalg`**, where
  `WeightMat` and all kernels already are. `embed` only exposes raw bytes. **No new edge.**
- **Q6_K unpack lives in `linalg`, mirrored from `embed/gguf.go`'s `dequantQ6KBlock`, with
  a shared golden** so the two copies can't drift. If the duplication ever grows past
  constants, extract a tiny leaf package both import (same pattern as the mmap leaf) —
  **not** an `embed→linalg` edge.

**Q6_K block layout** (from `embed/gguf.go:dequantQ6KBlock`, 210 B / 256 weights):
`ql[128]` (low 4 bits) · `qh[64]` (high 2 bits) · `sc[16]` int8 sub-scales (one per 16) ·
`d` (f16 super-scale). Weight `q = (low4 | high2<<4) − 32 ∈ [−32,31]`, value `= d · sc[is] · q`.

**Q8_K activation quantizer** (the prerequisite — flat per-row int8 cannot do it):
mirror llama.cpp `block_q8_K`: per-256-block f32 scale `d`, 256 int8 `qs`, **16 int16
`bsums`** (Σ of each consecutive-16 activations).
```go
type Q8K struct { d []float32; qs []int8; bsums []int16 } // d[nb], qs[nb*256], bsums[nb*16]
func QuantizeActQ8K(a []float32, M, K int) Q8K            // K % 256 == 0
```

**Fused dot** — the bsums fold Q6_K's −32 offset + per-16 sub-scales cheaply, keeping it
integer/bandwidth-bound:
```
dot = d_w·d_a · [ Σ_sub sc[sub]·(Σ_16 code·q8)  −  32 · Σ_sub sc[sub]·bsum[sub] ]
```
(`code∈[0,63]`; the −32 and sub-scales never recomputed in the inner loop — the whole
reason a Q8_K-style quantizer is on the critical path, not flat per-row int8.)

**Tiled, panel-resident GEMM (NOT per-column):**
```
matmulBTGGUF(a, blocks, ggmlType, dst, M, K, N):
  QuantizeActQ8K(a) once
  for nPanel in chunks of N (panel fits L2):
      stream/unpack the panel's Q6_K blocks once
      for m in 0..M: fused-dot(Q8K[m], panel) → dst   # GEMM over all M, panel resident
```
Naive on-the-fly (re-dequant per column) pays dequant M times and looks terrible at
prefill — the spike must be tiled or the go/no-go is a strawman.

**Standalone first — do NOT touch `WeightMat` dispatch until go/no-go clears** (adjustment #1).
All three deciding numbers (compute, footprint, parity) are measurable with a standalone
`matmulBTGGUF` + scalar reference + `QuantizeActQ8K`, called directly in the bench. The
`WeightMat` native-GGUF kind (`gguf []byte` + `ggmlType uint32` field, `WrapGGUF`,
`embed.TensorRawBytes(name)->(bytes, ggmlType)`, dispatch case, goinfer wiring) is the
**first productionization step AFTER a go**, not part of the spike. Don't add permanent
surface to a widely-used type for a spike that might say no. (Sketch, for later: new field
mutually exclusive with f32/q8/q4, checked first in the `MatmulBT` switch; nothing in
`WeightMat` is serialized, so the tag perturbs nothing on disk.)

## 5. Methodology guardrails (the go/no-go is a strawman without these)

1. **Tiled, panel-resident** — never per-column re-dequant (§4).
2. **Measure M=1 AND prefill M≥512.** Go/no-go is **prefill**, not decode. At M=1 it's
   weight-bandwidth-bound and a fused dot is nearly free; the dequant ALU cost only
   competes at large M, where the risk lives.
3. **Both arches.** amd64 (AVX2 — note whether an AVX-512-VNNI `VPDPBUSD` path is worth
   gating) and **arm64 NEON is the decision-flipper** (no VNNI, weaker integer dotprod):
   a Q6_K bandwidth-bound on AVX2 could be compute-bound on NEON and flip the verdict for
   every Apple-silicon user. **A one-arch result is half an answer.**
4. **Q8_K activations, not flat per-row int8** (§4).
5. **Baseline = the existing `dot_i8dp` W8A8 path.** The deciding number:
   **fused Q6_K×Q8_K throughput vs `MatmulBTW8A8`, at prefill-M, on NEON.**

## 6. Parity gates — keep the two comparisons DISTINCT (adjustment #2)

- **Kernel correctness:** fused-Q6_K vs the f32-dequant reference — argmax + cosine.
  Should be ~exact (same arithmetic). *Necessary, but NOT the decision gate.*
- **Decision gate:** native-Q6_K end-to-end vs **int4-requant** (and vs int8-requant for
  footprint) on a **rare-token continuation set**, scored separately from aggregate — a
  goinfer end-to-end measurement (aggregate top-1 hides tail regressions; goinfer saw an
  embed case −2.3 overall / −3.2 on rare / 0 on frequent). Make it an explicit harness row;
  don't let it hide inside "parity vs f32." aikit can supply the kernel-level proxy (§2);
  the token-level rare gate is goinfer's.

## 7. Open decisions (resolve before resuming)

1. **Proceed to the kernel at all?** Given footprint loses to int4 and the int8 head
   already captures most head-fidelity, the kernel's case narrows to native-direct +
   `.giw` fidelity. If that capability is wanted → proceed. If "ship D + keep the int8
   head" covers the need → this stays shelved (zero asm spent). **Owner: goinfer side
   (has the real rare-token data).**
2. **If proceeding: NEON-first or AVX2-now?** NEON is the deciding arch and can't be
   measured on the amd64 dev box. If the Mac is available soon → get the NEON go/no-go
   first; build the arch-independent core here meanwhile (§8). If the Mac is blocked a
   while → AVX2-on-amd64 now is fine idle-capacity work, held **conditional** (wasted if
   NEON says no-go and the desktop-Linux big-model case isn't independently worth it).

## 8. Suggested resume order (if it proceeds)

Arch-independent core first — useful on every branch (amd64-only, both, or even for D's
`.giw` native-block path), wastes nothing:
1. `QuantizeActQ8K` (Q8_K quantizer) + unit test.
2. Scalar tiled `matmulBTGGUF` (Q6_K×Q8_K) — the parity oracle.
3. Parity gate (fused vs f32-dequant; §6 comparison 1) using the real Q6_K tensor in
   `testdata/tinyllama-gguf/…Q4_K_M.gguf` (`output.weight`).
4. Both-arch bench harness (`linalg/q6k_perf_test.go`): fused vs `MatmulBTW8A8` at M=1 and
   M≥512, shapes ~2048×2048 and 1536×8960.
5. **Gate the asm on §7.** Then NEON kernel + the decision bench (Mac), and/or AVX2 (amd64).
6. **Only after a go:** `WeightMat` native-GGUF kind + `WrapGGUF` + `embed.TensorRawBytes`
   + goinfer wiring (§4).

## 9. Anchors

- `linalg/weightmat.go` — `WeightMat` type, kind dispatch, `Wrap*` (where the post-go kind lands).
- `linalg/quant.go` — `QuantizeRowsInt8`/`QuantizeGroupsInt4`, `MatmulBTW8A8` (the baseline), `MatmulBTQ4`.
- `embed/gguf.go` — `dequantQ6KBlock` (the layout to mirror), `Tensor`/`Dims` (and where
  `TensorRawBytes` would go).
- `linalg/dot_i8dp_arm64.s`, `dot_amd64.s` (`dotI8AVX2`), `dot_w4a8_{amd64,arm64}.s` —
  the int8 dot bodies the fused kernel reuses; the W4A8 kernels are the closest precedent
  (nibble-unpack prologue + reused SDOT/VPMADDWD body).
- Real test tensor: `testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` →
  21 Q6_K tensors, incl. `output.weight` [32000×2048].
