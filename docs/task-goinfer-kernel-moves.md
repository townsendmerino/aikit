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
| **M1** fusedattn | **v1.35.0** ⚠️ *perfgate skipped — see CHANGELOG* | *pending* | *not measured* | `TestAttendTileFused_bitIdenticalToGoinferRef`, mutation-checked ×2 |
| **M2** MXFP4 | **v1.36.0** (perfgate PASS) | goinfer `25a65447` | **identical** — argmax 244/244, cosine 0.999058 both sides, real 20B | `TestDequantMXFP4Split_bitIdenticalToGoinferRef` + `TestMXFP4Orders_areNotInterchangeable`, mutation-checked ×3 |
| M3 W4A8 device kernels | *pending* | — | — | — |
| M4 sequence mixers | *Phase 0 only, no move* | — | — | — |

**Status 2026-09-06: M5 done. M1 landed (v1.35.0), goinfer half pending. M2 COMPLETE both halves (v1.36.0 → goinfer `25a65447`). M3–M4 not started.**

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

---

## M2 · MXFP4 — aikit half DONE, goinfer half pending

**What duplicated.** goinfer's `decoder/mxfp4.go` and this repo's `embed/gguf_dequant.go` both
carry the MXFP4 (OCP FP4, ggml type 39) e2m1 value table and e8m0 scale conversion, **byte for
byte** — same 16 doubled values, same bit formula, same lineage comment pointing at
`gguf/quants.py`. goinfer additionally has one thing aikit did not: the **safetensors layout**,
where the packed nibbles and the block scales arrive as two separate tensors. aikit's copy was
unexported, so nothing could reuse it either way.

**What the gate had to pin first: the two layouts are not one function.** They share the block
size, the scale encoding and the value table, so they read as the same kernel with different
addressing. They are not:

| layout | shape | byte j packs |
|---|---|---|
| GGUF / GGML | contiguous 17-byte blocks | elements **j and j+16** |
| safetensors | two tensors (`*_blocks`, `*_scales`) | elements **2j and 2j+1** |

goinfer measured this rather than assuming it, and the assumption it replaced was wrong: its
Phase 0 had recorded *"no new numerics, only the addressing differs"*. Dequantizing a real gpt-oss
expert both ways and diffing against the same weight read through the already-validated GGUF path
gave **cosine 0.081 for GGML order and 1.000000 for sequential**. Routing safetensors data through
the GGML core yields finite, plausibly-scaled, completely wrong weights — it does not error and it
does not look broken. So `DequantMXFP4Blocks` and `DequantMXFP4Split` stay separate on purpose,
and `TestMXFP4Orders_areNotInterchangeable` goes red if they are ever unified, carrying that
measurement in its own failure text.

**New aikit surface** (`embed/mxfp4.go`): `MXFP4Scale`, `DequantMXFP4Blocks`, `DequantMXFP4Split`,
`MXFP4BlockElems` / `MXFP4BlockBytes`. Split writes into a caller-owned `dst` because the caller is
streaming — gpt-oss-20b's experts are ~76 GB dequantized to f32 across all layers — and checks its
shapes **exactly** rather than as lower bounds, since `blocks` and `scales` are independently
shaped tensors whose mismatch is the corruption worth refusing loudly.

**The gate, written before the code** (it did not compile until the API existed, which is the
point). Raw bits via `math.Float32bits`, never a tolerance, against frozen copies of goinfer's
bodies at `4f5da73c`: all **256** e8m0 bytes — the whole domain, subnormals included — block counts
1/2/3/17/256 so an off-by-one in the stride cannot be averaged away, and every one of the 256 byte
values appearing as a nibble pair. No Python: the vectors are generated in Go from the source
alone.

**Mutation-checked three ways.** Swapping the split order to GGML's j/j+16 — the exact historical
bug — reds 22 of 32 values on the first block. Perturbing one lane reds exactly one, printing
`00400000` vs `00600000`. Shifting the e8m0 exponent field by one reds the scale test at x=2.

### A fourth mutation stayed green, and chasing it found a false comment

Replacing the bit formula with `float32(math.Pow(2, x-128))` changed **nothing**. That reads at
first like a hole in the gate; it is not. Measured over all 256 inputs, the two forms are
bit-identical **everywhere** — 2^-128 and 2^-127 are exactly representable as float32 subnormals,
which reach 2^-149, and Go's `math.Pow` is exact for powers of two.

So the comment *both* repos carried — "the exact bit formula (not 2^(x-128)) keeps the x∈{0,1}
subnormals bit-identical to the reference" — was **over-claiming**. The formula is a fine choice
(exact by construction, no libm call), but the stated *reason* was false, and it would have told
anyone simplifying the function that they had broken subnormals they had not touched. Corrected in
aikit with the measurement recorded; goinfer's copy carries the same wording and gets the same fix
when its half lands. This is the `A DOC COMMENT CLAIMING COVERAGE IS NOT COVERAGE` rule one step
sideways: a comment claiming a *necessity* that no test asserts and no measurement supports.

### M2's goinfer half — DONE, and the golden did not move

aikit **v1.36.0** was tagged first (releasegate 4/4, perfgate PASS, vulncheck clean 15/15, root CI
green), then goinfer bumped onto it, swapped `decoder/gptoss_safetensors.go:86` from
`mxfp4DequantSplitInto` to `embed.DequantMXFP4Split`, and deleted `decoder/mxfp4.go`. Never the
other order, and goinfer never built against an untagged aikit.

**The paired cell**, same box and same session, on the real 20B checkpoint —
`TestGptOssSafetensors_vsGGUF` under `-tags realckpt` with both checkpoints read from local NVMe:

| | argmax | logit cosine | wall |
|---|---|---|---|
| **before** — goinfer's own `mxfp4DequantSplitInto`, aikit v1.35.0 | 244 vs 244 | **0.999058** | 262.06 s |
| **after** — `embed.DequantMXFP4Split`, aikit v1.36.0 | 244 vs 244 | **0.999058** | 262.19 s |

Every printed digit identical. That test is the right one for this move rather than a convenient
one: it diffs the safetensors loader against the already-T3-validated GGUF path on the same model,
and the split nibble order is exactly what it exists to catch.

**goinfer keeps its `decoder/mxfp4_test.go`**, repointed at aikit rather than deleted with the
implementation. It holds what aikit's gate structurally cannot: a fixture from a real gpt-oss:20b
tensor dequantized by the reference `gguf` **Python** library. aikit has no Python, so its vectors
are Go-generated and self-referential by construction; goinfer's is an independent oracle, and
deleting it would have removed the only check that the packing matches what the reference actually
emits for bytes off a real checkpoint. This is the general shape: a move deletes duplicated
ARITHMETIC, not the consumer-side test that has an oracle the kernel repo cannot reach.
