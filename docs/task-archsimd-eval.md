# Task: should aikit adopt Go's `simd`/`archsimd` for `linalg.Dot` and the encoder int8 kernels? (2026-09-10)

> **BLUF. No, for both named kernels — and the three reasons are independent, so none
> of them is a "we could fix that later".**
>
> 1. **`ann.Flat.Query` is memory-bandwidth-bound by an order of magnitude, measured.**
>    A dot product reads each operand once and reuses nothing, so the f32 cosine scan has a
>    fixed arithmetic intensity of **0.25 MAC/byte at every dimension** — a property of the
>    operation, not of the kernel. This box's ridge point is **2.74 MAC/byte** (305.7 GMAC/s
>    of f32 FMA against 111.5 GB/s of DRAM read, both measured below). The scan sits
>    **11.0× on the memory side of it.** In production — where `Flat.Query` already shards
>    across every core — the scan reaches **100.7 GB/s, i.e. 90% of the measured DRAM read
>    ceiling, while using 8.2% of the machine's arithmetic.** A wider multiply optimises the
>    92% that is already idle.
> 2. **`archsimd` cannot express the int8 kernels aikit already has.** The brief predicted
>    the int8 reranker path was "VPDPBUSD territory" and the likelier win. It is VPDPBUSD
>    territory — aikit's `dot_i8_avx512vnni_amd64.s` issues that instruction 9 times — and
>    **Go 1.27's `simd` and `archsimd` do not expose it at all.** The only dot-product
>    primitives in `archsimd` are `Int16xN.DotProductPairs` (VPMADDWD) and
>    `Uint8xN.DotProductPairsSaturated` (VPMADDUBSW). On arm64 `archsimd` exposes `Int8x16`
>    with **zero** dot-product methods — no SDOT — so the arm64 int8 kernels
>    (`dotI8SDOT`, `dotI8Tile4x4` at 163.5 GMAC/s) could not be ported either. Rewriting in
>    `archsimd` would be a **downgrade on both architectures**.
> 3. **aikit already ran this campaign, and already shipped the parts that pay.** Go 1.27's
>    `simd` landed in v1.23.0 and again in v1.24.0 for nine kernels — softmax, exp, SiLU,
>    GELU/erf, tanh, RoPE, SPLADE's log1p — at 1.4×–5.6×, behind `//go:build goexperiment.simd`
>    with scalar fallbacks and a CI leg on both arm64 and amd64 runners. Those kernels won
>    because they are **compute-bound elementwise passes with no incumbent assembly and ops the
>    portable API actually has.** The two kernels in this brief fail all three tests. The
>    infrastructure the brief asks to build already exists; the question is only whether these
>    kernels belong behind it, and they do not.
>
> **Recommendation: NO-GO on both. No code was written beyond the measurement harness**
> (`ann/flat_roofline_bench_test.go`, retained as the evidence). **The lever the roofline
> actually points at is bytes per candidate, and aikit has already built every rung of that
> ladder** — measured at d=768/N=200k: f32 → int8 is **4.44×** (almost exactly the 4× byte
> ratio, the signature of a bandwidth-bound scan) and f32 → binary is **11.6×**. `HNSW`,
> `FlatI8` and `FlatBinary` are all shipped. Nothing here is a scaling fix and nothing here
> needed one.

**Box.** `apple-m1pro` — Apple M1 Pro, 6 P + 2 E cores, Go 1.27.0 darwin/arm64, aikit at
`c214070`. Every number below was measured on it during this task unless labelled otherwise.
amd64 figures are quoted from `docs/task-simd-audit.md` with their original box labels; **no
amd64 hardware was available for this task**, which is called out again in §6 because it
bounds one of the conclusions and not the others.

---

## 1. Phase 1 — inventory: what the brief assumed vs what is in the tree

The brief's framing assumed a mostly-scalar library about to adopt its first vector code. That
is not this repo. Correcting it changes the question:

| brief's assumption | actual state |
|---|---|
| `linalg.Dot` is "SIMD-f32" from v1.4, implementation unspecified | **Hand-written assembly on both arches.** arm64: `dot_arm64.s` (4-lane VFMLA). amd64: `dot_amd64.s` (AVX2+FMA) behind hand-rolled CPUID/XGETBV detection in `dot_amd64.go` — *exactly* the Stapelberg dispatch pattern the brief proposes, already built, in `.s` rather than `archsimd`. |
| the encoder int8 kernels might benefit from AVX512-VNNI | **AVX512-VNNI is already implemented**: `dot_i8_avx512vnni_amd64.s` and `dot_w4a8_avx512vnni_amd64.s`, measured +1.23–1.28× on a cloud Xeon. Plus AVX2 (`dotI8AVX2`), NEON SDOT (`dot_i8dp_arm64.s`), and register-blocked tiles on both arches (`dotI8Tile4x4`, 3.5–3.9× at M≥4). 25 assembly files in `linalg/` total. |
| aikit cannot require `GOEXPERIMENT=simd`, so this would be new build-tag machinery | **The machinery exists and is in CI.** `linalg/exp_simd.go`, `encoder/rope_simd.go`, `encoder/log1p_simd.go` are `//go:build goexperiment.simd`, each paired with a `_scalar.go` default-build fallback. `.github/workflows/ci.yml:151-184` runs a `GOEXPERIMENT=simd` leg on **two** runners (arm64 and amd64) plus a `GODEBUG=simd=0` emulation leg. |
| `archsimd` is amd64-only, so Apple Silicon gets nothing | **Half right, and the half that matters is the other one.** aikit uses the *portable* `simd` package, which lowers to NEON on arm64 — the shipped softmax kernel measured ~2.5–2.6× on this M1 Pro. `simd/archsimd` (the fixed-width, arch-specific tier) does have arm64 files, but its arm64 int8 surface is one 128-bit type with no dot product. |
| `HNSW` is the real fix for the O(N) scan | Already shipped (`ann/hnsw.go`), as are `FlatI8`, `FlatBinary`, mmap and sharded variants. |
| aikit targets Go 1.27 | Confirmed: `go 1.27.0`. `GOAMD64` is deliberately **unpinned** — see `CHANGELOG.md:63`, which records that as a resolved finding, since the kernels dispatch at runtime rather than at build level. |

So: both kernels named in the brief are already vectorised, already runtime-dispatched, and
already at 72–97% of their measured issue ceilings per `docs/task-simd-audit.md` §2.

## 2. The decision gate — is the flat scan compute-bound or bandwidth-bound?

### 2.1 The answer is fixed before any measurement

One dim-`D` f32 candidate is `D*4` bytes and `D` multiply-accumulates. That is **0.25 MAC per
byte at every D**, because a dot product touches each operand once and reuses nothing. No
kernel — scalar, NEON, AVX2, AVX-512 — can change that ratio. The only thing an implementation
controls is *where on the roofline the scan actually lands*, which is what the rest of this
section measures.

### 2.2 The two ceilings, measured on this box

**Memory.** `BenchmarkReadBW` — 512 MB buffer (past every cache), data-dependent fill and a
package-level sink so the reduction cannot be folded away, eight independent accumulators so
the loop is limited by loads rather than add latency:

| threads | 1 | 2 | 4 | 6 | 8 |
|---|--:|--:|--:|--:|--:|
| **read GB/s** | 36.9 | 71.6 | 109.5 | **111.5** | 102.8 |

(benchstat, `-count=6`, all within ±7%. benchstat reports GiB/s; converted here to decimal GB/s.)

Ceiling assert, per this repo's own rule that *a bandwidth probe without a ceiling assert is not
a probe*: the M1 Pro's theoretical LPDDR5 ceiling is 200 GB/s, so the peak reading is 56% of
theoretical — plausible for a load-only loop, and not the kind of impossible figure that
catches a folded reduction. **A first version of this probe reported 19.3 GB/s single-threaded
and was wrong**: a single `s += v` chain is limited by add latency (~1/cycle), not by memory.
Unrolling to eight accumulators moved it to 36.9. The uncorrected number would have made the
scan look *faster than memory* and inverted the verdict.

**Compute.** `linalg.MeasuredFMAPeakGFLOPS()` (the repo's own register-saturating probe, no
memory traffic): **101.9 GFLOPS single-core = 50.9 GMAC/s**. Across 6 P-cores, **305.7 GMAC/s**.

**Ridge point** = 305.7 GMAC/s ÷ 111.5 GB/s = **2.74 MAC/byte**. The scan runs at **0.25** —
**11.0× onto the memory side.**

### 2.3 The scaling test — the decisive measurement

Same `scanFlat` kernel, same DRAM-resident corpus (N=200k), W concurrent workers over disjoint
slices, which is what `Flat.queryShards` does in production. If the scan were compute-bound,
throughput scales with W. If it is bandwidth-bound, it plateaus.

| workers | 1 | 2 | 4 | 6 | 8 | plateau vs 111.5 GB/s ceiling |
|---|--:|--:|--:|--:|--:|---|
| **d=256** GB/s | 47.4 | 83.2 | 98.3 | **100.7** | 97.2 | **90%** — effectively saturated at 4 workers |
| **d=768** GB/s | 21.4 | 40.5 | 68.6 | **80.1** | 73.5 | **72%** — saturated at 6 |

(benchstat, `-count=6`; ±1–6% except d256/w6 at ±14% and d768/w4 at ±13%.)

**This is the gate, and it fails for compute.** At d=256 the scan is within 2% of its plateau by
**four of eight cores** (98.3 → 100.7 GB/s from 4 to 6, inside the ±14% band at w6); past that,
adding cores does nothing, and neither would a faster kernel. In MAC terms the plateau is
25.2 GMAC/s against a machine compute peak of 305.7 — **the scan already extracts 90% of the
available bandwidth while leaving 92% of the arithmetic idle.** Wider SIMD optimises the idle 92%.

`Flat.Query` shards to `runtime.NumCPU()` (`flatQueryWorkers`, `ann/flat.go:176-187`) above a
0.5 M-element threshold, so **production runs at the plateau**, not at the single-core point.
That is why the single-core headroom below is not a latency win.

### 2.4 Two hypotheses tested and refuted

Honest reporting of what did *not* pan out, since both looked promising:

- **Allocator scatter.** `ann.New` is handed `[][]float32` — one allocation per row. Rebuilding
  the corpus from a single contiguous backing array (`BenchmarkFlatScanLayout`, same values,
  same seed, layout the only variable) changed nothing: d256/N200k 48.2 GB/s scattered vs 46.7
  contiguous; d768 21.5 vs 21.6. **Layout is not the lever.**
- **The `Dot8x4` large-K cliff.** `linalg.go:26` explicitly warns that `Dot8x4`'s 8 live
  accumulators plus 8 streamed rows outgrow the register budget at large K and that callers
  should feed it strips — and `scanFlat` hands it the whole row. Tiling K at d=768
  (`BenchmarkFlatScanTileK`) made it **worse**, not better: untiled 21.5 GB/s vs 18.3 / 19.6 /
  20.7 / 21.2 at strips of 64 / 128 / 256 / 384. The extra per-strip fold costs more than the
  locality buys. **Not the lever either.**

### 2.5 The one honest residual

At d=768 single-core streaming is 21.4 GB/s against d=256's 47.4 — the kernel, not memory,
since 6 workers still scale 3.7× from that point. So d=768 saturates the machine only at 6
workers and tops out at 72% of the ceiling rather than 90%. There is a **bounded ~1.25×**
available at d=768 *if* the per-core streaming rate were brought up to d=256's.

This is worth recording but it is **not an argument for `archsimd`**: the kernel is already
NEON FMA, the two obvious causes were tested and refuted above, and the remaining 24% is a
multi-stream prefetch/locality question at a specific dimension — not a multiply-width
question. It is future work, scoped as such, and it does not change the go/no-go.

## 3. What `archsimd` can actually express (the int8 verdict)

The brief's prediction was that the int8 kernels were the likelier win. They are the *clearer
loss*, and the reason is an API ceiling rather than a performance argument:

| primitive | needed by | in Go 1.27 `simd`/`archsimd`? |
|---|---|---|
| `VPDPBUSD` (AVX512-VNNI / AVX-VNNI, 4-way int8→int32) | `dot_i8_avx512vnni_amd64.s` (**9 occurrences**), `dot_w4a8_avx512vnni_amd64.s` | **No — absent from the entire package.** |
| `SDOT` (arm64 4-way int8→int32) | `dot_i8dp_arm64.s`, `dot_i8_tile_arm64.s`, `dot_w4a8_arm64.s` | **No — `archsimd`'s arm64 `Int8x16` has zero dot-product methods.** |
| `VPMADDWD` (int16 pairs→int32) | `dotI8AVX2` | Yes (`Int16xN.DotProductPairs`). |
| `VPMADDUBSW` | — (aikit deliberately does not use it; both operands are sign-extended) | Yes (`Uint8xN.DotProductPairsSaturated`). |

The portable `simd.Int8s` has no widening dot product at all — its `Mul` is int8×int8→int8,
which overflows immediately on a dot product. Reaching int32 accumulation through the portable
API means manually widening to `Int16s`/`Int32s`, which is precisely the work `SDOT`/`VPDPBUSD`
exist to avoid.

So a port would trade a VNNI kernel for a VPMADDWD one on amd64 and lose SDOT entirely on
arm64. **There is no version of this that is not a regression**, and this holds regardless of
what the benchmark on an amd64 box would have said — which is why the missing amd64 hardware
does not weaken this particular conclusion.

## 4. Why `simd` *did* pay for the kernels aikit already shipped

The pattern is consistent, and it is the useful generalisation:

| | shipped `simd` kernels (softmax, exp, SiLU, GELU/erf, tanh, RoPE, log1p) | the two kernels in this brief |
|---|---|---|
| bound | compute-bound elementwise passes on cache-resident data | `Dot`: memory-bound 11.0× over. int8: compute-bound but already at 72–96% of issue |
| incumbent | **none** — scalar Go; Go's `simd` ships zero transcendentals, so `log1p` had to be ported from Cephes | 25 files of tuned assembly at 91–97% of ceiling |
| expressible? | yes — `Mul`/`Add`/`MulAdd`/`Max` on `Float32s` is all these need | **no** — the required dot-product primitives are absent |
| result | **1.4×–5.6×**, shipped in v1.23.0 / v1.24.0 | would be a downgrade |

`simd` wins where there is no assembly to beat and the ops exist. Neither holds here.

## 5. What to do instead — the lever the roofline points at

If intensity is fixed at 0.25 MAC/byte and the scan is bandwidth-bound, the only lever is
**fewer bytes per candidate**. Measured end-to-end (`BenchmarkFlatPrecisionLadder`, full
`Query` including sharding and top-k, N=200k):

| | d=768 | vs f32 | d=256 | vs f32 |
|---|--:|--:|--:|--:|
| `Flat` (f32, 4 B/dim) | 8.08 ms | 1.00× | 2.32 ms | 1.00× |
| `FlatI8` (int8, 1 B/dim) | **1.82 ms** | **4.44×** | 1.16 ms | 2.00× |
| `FlatBinary` (1 bit/dim) | **0.694 ms** | **11.6×** | 0.449 ms | 5.2× |

At d=768 the int8 speedup is **4.44× against a 4× byte reduction** — the ratio tracking the
byte count almost exactly is the cleanest possible confirmation that the scan is bandwidth-bound
and not compute-bound. (The d=256 f32 column was noisy across repeats — 2.32–5.25 ms — so the
d=768 row is the one to quote; d=256's smaller per-candidate work also leaves the fixed
per-vector overhead a larger share, which is why its ratios are lower.)

**All three rungs already exist.** The recommendation is therefore not to build anything: it is
that callers on large corpora should be on `FlatI8`/`FlatBinary`/`HNSW`, which is what
`ann`'s existing documentation already says.

## 6. Scope limits

- **No amd64 hardware was available for this task.** Every measurement here is `apple-m1pro`.
  This bounds §2's specific *numbers* on amd64 — a Zen 2 box has ~30 GB/s of read bandwidth
  (`docs/task-simd-audit.md` §S-08.1) against a similar FMA peak, so its ridge point is
  *further* from 0.25 MAC/byte, not closer; the bandwidth-bound verdict travels, and would
  travel more strongly. It does **not** bound §3, which is an API-surface fact about Go 1.27
  and true on every box.
- The 0.25 MAC/byte intensity argument is arithmetic, not measurement, and holds everywhere.
- `docs/task-simd-audit.md` remains the authority on the decode/prefill kernels; this document
  covers only the two kernels the brief named and does not revisit its findings.

## 7. Verification

Both build modes green after adding the harness (2026-09-10, 21:26:53 → 21:28:40 PDT):

```
default build:              ann ok   linalg ok   encoder ok
GOEXPERIMENT=simd build:    ann ok   linalg ok   encoder ok
```

No production file was modified, so the numerics gates (cosine ≥ 1−1e-5 vs the Python
reference, the golden tests, the recall decomposition) are untouched by construction — the
only change is the addition of `ann/flat_roofline_bench_test.go`, a test-tagged file.

## 8. Realistic end-user impact

**Nobody gets a win from this, which is the finding.** Had the kernels been ported, the amd64
ken-mcp servers would have gained nothing on the flat scan — it is bandwidth-bound 11.0× over,
and already extracts 90% of the memory system by four cores — and would have *lost* 1.23–1.28×
on the int8 reranker, because `archsimd` has no VPDPBUSD and the existing hand-written kernel
does. Apple Silicon would have lost more: no SDOT in `archsimd` at all. Nothing would change
for `go install` users on the default build, which is the only genuinely good news, and it is
the good news of a no-op. The maintenance cost that would have bought this is an experimental
API, a second dispatch path per kernel, and a CI matrix that already carries two
`GOEXPERIMENT=simd` legs. The honest summary is that aikit already made this trade correctly
twice: it adopted `simd` in v1.23.0/v1.24.0 for the nine elementwise kernels where there was no
assembly to beat and the ops existed, and it should decline it here for the two kernels where
tuned assembly already runs at 91–97% of issue and the ops do not exist. **The constant factor
on this O(N) scan is not the constraint — the memory bus is, and the fix for that is the
precision ladder and HNSW, both of which are already shipped.**
