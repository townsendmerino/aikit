# Changelog

All notable changes to `aikit` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html): see
[README.md](README.md#stability-tiers) for the two-tier split — the Hard tier
follows semver (no breaking change before a v2.0), the Experimental tier is
excluded from that promise and may change in any release until it graduates.

## [Unreleased]

### Fixed

**`embed`: Q6_K dequant hoists the `d·scale` product out of its element loop (audit M-24).** The
sub-block index `is` takes only two values across the 32 iterations, so the same eight products
were recomputed sixteen times each — four float32 multiplies per output element, on a load-time
path that runs over every tensor in the file. Bit-identical, and the grouping is what makes it
so: the original reads `d * float32(scale) * float32(q)`, which Go evaluates left to right as
`(d*scale) * q`, so precomputing `d*scale` preserves the same two roundings in the same order.
`TestDequantQ6KBlock_hoistIsBitIdentical` compares BITS (not values) against the pre-hoist body
and was verified to fail on a 1-ULP perturbation.


**CPU-only `FlatI8.QueryBatch` uses the batched M-row GEMM — 3.6x (audit M-18).** The batched
W8A8 kernel was already in the file, reachable ONLY through the GPU shard split; with no device
attached `QueryBatch` fell through to a per-query loop, so the corpus weight matrix was streamed
once per query instead of once per batch. Every recorded GPU "x vs CPU" crossover was measured
against that non-batching baseline. Measured on `apple-m1pro`, benchstat `-count=6`, dim 256:
N=10k/batch=8 **-72.6%**, N=10k/batch=64 **-78.3%**, N=100k/batch=8 **-66.8%**,
N=100k/batch=64 **-70.3%**, geomean **-72.3%** (all p=0.002) — above the 2.1-2.5x the
shard-split records bounded it at. `BenchmarkFlatI8QueryBatchCPU` is new.


**`late.MaxSim` scores eight document tokens per kernel call — 2.4x (audit M-20).** Late
interaction is the one-vs-many shape `linalg.Dot8x4` exists for, but `MaxSim` was calling the
single-row `linalg.Dot` once per (query token x doc token) pair, re-streaming the query strip
every time. `Flat` and HNSW already score this way. Measured on `apple-m1pro`, benchstat
`-count=6`, nQ=32 d=128: nDoc=64 **-65.6%**, nDoc=256 **-56.8%**, nDoc=1024 **-50.2%**, geomean
**-58.0%** (all p=0.002). The `late` package had no benchmark at all, which is why this sat
unmeasured; `BenchmarkMaxSim` is new. Not bit-identical — the 8-row kernel reassociates, so a
score can move ~1 float32 ULP, the same tradeoff `Flat` and HNSW document for the same kernel.
`TestMaxSim_batchedMatchesPerPair` gates it against the per-pair reference across dims and
document lengths that straddle both the 8-row group and the 4-lane tail.


**`gpudevice`'s pasted verdict no longer counts an all-skip module as green (audit G-07).** The
per-module row already printed SKIPPED and a NOTE listed them, but the VERDICT line — the part
that ends up in a tag message and gets read months later — said "N/N applicable gpu modules
green" counting them. A module whose every test skipped for an absent fixture is evidence of
nothing about that module. The line now reports green-vs-applicable separately, names the
skipped modules, and returns INCONCLUSIVE when nothing exercised a device at all. This was not
hypothetical: on the day it was written `gpu/visionmetal` ran **0** tests and `gpu/qwenmetal`
**1**, both for missing `testdata` fixtures, and the suite printed `ok`.


**`perfgate` and `releasegate` hardened so the gate can actually fail (audit G-01..G-06).** The
audit's point was not that the gate was wrong but that it could not lose: it measured one kernel
family at one M, re-derived its floor every run, and exited 0 when it had measured nothing.
Every perf fix in this release would have shipped under a green from it. Six changes:

- **G-01 coverage.** The benchmark set was the two W8A8 M=1 families, leaving the int4 decode
  kernel, BOTH M>1 tiles, the acc64 attention kernels, the activation quantiser, the S-06
  contract kernels and `AttendTileFused` with no performance gate at all. It now also runs
  `MatmulBTW8A8Batch_prefill`, `W4A8_CanonicalVsSplitHalf`, `MatmulQKAcc64`/`MatmulAVAcc64`,
  `AttendTileFused_expKind`, `SoftmaxRowKernels`/`SiLUKernels` and `QuantizeRowInt8` — 45 rows
  where there were 6.
- **G-02 blind greens.** 0 shapes resolving the target class now exits 3 with a BLIND verdict
  instead of printing a green.
- **G-03 reproducibility.** A FAIL now has to reproduce on a second interleaved measurement.
  Previously the per-run floor made a single FAIL unreproducible and "re-run until green"
  structurally available — which is what v1.37.0 did. A regression that appears once and not
  twice is now INCONCLUSIVE (exit 5): not green, not a fail.
- **G-04 release evidence.** `releasegate` now requires the CHANGELOG section to carry a
  perfgate `VERDICT` or an explicit `EXCEPTION`. Checked against history: it passes v1.39.1 and
  **fails v1.38.0**, the release that changed both ViT towers' attention schedule and shipped
  with no perfgate and no recorded exception.
- **G-05 interleave bias.** ABAB became ABBA, so a monotone drift over the run no longer lands
  entirely on one arm — the single bias the interleave exists to remove.
- **G-06 silent no-op.** A failing benchmark binary, or one producing no parseable rows, was
  swallowed (`outB, _ :=`) and reported "PASS — 0/0 shapes". Both now exit 4 with the binary's
  own output attached. Verified: a deliberately bogus regex now exits 4 where it used to exit 0.

Two of the gate's existing rows also measured the wrong thing and were fixed: `W8A8SpanShapes`
now pins its Workspace threshold so the streamed shapes measure the KERNEL rather than whether a
fork/join happened (three of them sit within a few percent of `parThreshold` on a 16-thread box,
which is why they have read BLIND at ±11-35% in every recorded run), and `GEMV_W8A8_baseline`
moved off the allocating wrapper to `MatmulBTW8A8Into` with the same pin — its ns/op included a
`make`, and at K4096_N4096 it fanned out while the Q6K kernel it is compared against runs serial.


**The activation quantiser is row-split under the fan-out instead of running serially on the
caller (audit M-03).** Every W8A8/W4A8 entry quantised all M rows in a plain loop on the calling
goroutine and only then fanned the matmul across columns, so at prefill — where M is 512+ — six
to sixteen cores idled through the whole quantise. Five call sites now share one
`quantizeRowsInto` helper that splits by row above a 65,536-element threshold. Bit-identical:
each row's max-abs and scaled round depend only on that row.

Measured where the mechanism applies (M*K >= 65,536, i.e. M >= 43 at K=1536). On
`nvidia-rtx2070s` (Ryzen 7 3700X, where the quantiser is still scalar and the audit put this at
11-16% of prefill): q&#8214;k&#8214;v M=128 **-61.4%**, gate&#8214;up M=128 **-31.1%**, geomean **-14.1%**
over the shape set. The M=1 and M=4 control shapes take a provably identical serial path and
were re-measured in isolation to confirm they do not move (p=0.878/0.645/0.279/0.878). On
`apple-m1pro` the NEON quantiser makes the same term ~1.3% of a prefill, and the measured
movement there sits too close to the harness noise floor to attribute — see below.

*Methodology note.* A NULL A/B on `apple-m1pro` — the same binary benchmarked against itself —
reports differences up to **+6.1%** and a **+3.86% geomean drift**, including one shape at
"p=0.002" where the code is byte-identical. benchstat's ± is within-run variance; thermal and
layout drift BETWEEN runs is not in it. Treat sub-10% single-shape deltas on that box as
unattributable without an isolated re-run, which is how the controls above were checked. Every
other figure in this release is far above that floor.


**Text-encoder attention is now head-parallel — another 19.6% off a long encode (audit M-05).**
The 12-head loop and both of its per-head matmuls ran on ONE core while the linears around them
used every core. That was not an oversight so much as an unanswered question: `parallelThreshold`
is 32M MACs, the per-head QK^T at headDim=64 clears it only at L >= 708, and BERT /
CodeRankEmbed / rerank all cap at maxSeq=512 — so every per-head matmul was deliberately serial,
with the threshold's own comment noting the net effect "needs an end-to-end forward benchmark on
real weights to judge — a follow-up, not guessed at here". This is that benchmark.
`BenchmarkEncode_singleLong` on `apple-m1pro`, benchstat `-count=6`: 682.9ms -> 549.0ms,
**-19.6%** (p=0.002). Together with the contract-kernel routing earlier in this release the same
benchmark has gone 800.1ms -> 549.0ms, **-31.4%**.

Bit-identical: each head reads its own slice of Q/K/V and writes its own columns of ctx, so no
reduction crosses a head. The fan-out is deliberately narrow — it declines when a compute
backend is attached (`s.mm` would be called concurrently and a device queue makes no such
promise), when the per-head matmul would itself parallelize (the cores are already busy and a
head fan-out on top would oversubscribe), and when another forward is in flight. Per-worker head
buffers come from a pool rather than sizing the per-forward arena by `numCPU`, so the serial path
pays nothing: `scores` alone is 1 MB at L=512. Gated by the full encoder suite and `-race`.


**`vision`: an unset `Config.MaxPixels` no longer disables the decompression-bomb guard, and
Gemma 4's pooler divides by real occupancy (audit C-06).** The guard read
`cfg.MaxPixels > 0 && ...`, so a `Config` literal that simply did not name the field — the
natural thing to write, and what the zero value gives — turned the check OFF entirely. An unset
security limit now resolves to the new exported `DefaultMaxPixels` (16 MP, what `Gemma3()`
already used); a caller who wants a larger bound sets a larger number. Separately,
`Gemma4Encoder.averagePool` scaled every bucket by `1/(k*k)` regardless of how many patches
landed in it: the divisibility check guarantees the grid splits into whole buckets but not that
`positionIDs` densely covers it, so a sparse or padded patch set left a bucket under-filled and
scaled toward zero — a mean that is quietly wrong rather than an error. It now divides by the
bucket's actual count, identical to the old constant whenever every bucket is full (the only
case today's single-image path produces).


**`gpu/qwencuda`: a write-after-read race on the ViT segment bounds, and the same hazard closed
generally in `Buffer.upload`/`CopyDevice`/`UploadBatch` (audit C-01).** The encoder uploaded new
per-patch attention bounds INSIDE the layer loop whenever the block kind switched. `gpu.Upload`
copies on the legacy null stream and synchronizes AFTER the copy, which orders it before later
launches — the read-after-write race that sync was written for — but nothing ordered it after
EARLIER ones, and the queue's stream is `CU_STREAM_NON_BLOCKING`, which by definition does not
order against the null stream. So at the first full-attention block (layer 7 in Qwen2.5-VL) the
host issued full-image bounds while layers 0-6 were still queued; the DMA could land under them
and their `attention_seg` would read `n = s1-s0` up to the whole image with shared memory sized
for `MaxWinSeg` — an out-of-bounds shared write, i.e. garbage or a context-killing fault. Both
bound pairs are now resident and uploaded once before the first launch, with the layer loop only
choosing which to bind: the hazard is removed by construction rather than by adding another
sync, and it restores the "ONE Sync per forward" property the file's header claims (the old form
made 14 full-device syncs). Independently, `Buffer.upload`, `CopyDevice` and `UploadBatch` now
synchronize BEFORE the copy as well as after, closing the same write-after-read shape for every
other caller.

Verified on `nvidia-rtx2070s` (RTX 2070 SUPER, Ryzen 7 3700X): `gpu` 42 passed, `anncuda` 17,
`qwencuda` 3, `enccuda` 5, `visioncuda` 3, zero failures, with GPU≡CPU cosine **1.000000000** and
identical worst-case deltas before and after — as expected, the fix removes a race without
changing arithmetic. NOTE the tiny fixture cannot expose the original defect (each layer's GPU
work is shorter than its own enqueue, so the queue never builds); the parity tests passing is
necessary, not sufficient, and this is recorded rather than claimed as proof.


**The ViT towers no longer pay f64 `math.Exp` per attention score — the fused-attention kernel is
1.43-1.73x faster, growing with patch count (audit M-09).** P6a moved the towers onto
`AttendTileFused`, whose f64 `math.Exp` and f64 score-scale pass are part of the bit contract
goinfer's raw-bit gate depends on — but the vision towers gate at COSINE, not bits, and before
P6a they ran the f32 `expF32Core` that item 13 chose for them (measured -30.6% on the 576-patch
tower at the time). New `linalg.AttendTileFusedContractExp` keeps P6a's fused schedule and takes
the exponential from the S-06 contract kernel (vectorised NEON/AVX2, batched over the score row
rather than called per element) with an f32 scale pass; the three towers use it.
`AttendTileFused` itself is untouched and `TestAttendTileFused_bitIdenticalToGoinferRef` still
passes, so goinfer's contract is unaffected.

Kernel, `apple-m1pro`, benchstat `-count=6`, hd=80, full bidirectional attention:
np=196 **-30.0%**, np=576 **-39.2%**, np=1024 **-41.0%**, np=2048 **-42.2%** (all p=0.002) —
the ratio rises monotonically with np, as the closed form predicts.

END-TO-END THE EFFECT IS MUCH SMALLER THAN THE AUDIT'S HEADLINE, and the reason is structural
rather than a measurement problem: attention is O(np^2) while the projections and MLP around it
are O(np), so the exponential's share of a tower grows with patch count. On the in-tree tower
fixtures the whole-forward numbers are `SiglipTower/p196_h512` **-4.4%** (p=0.002) and
`/p576_h768` **no significant change** (p=0.485). The audit's "the exp costs more than the
attention MACs" is an so400m claim at np=4096, which no fixture in this repo can exercise — the
largest is np=576. The kernel sweep above is the evidence for the mechanism; the tower numbers
are the honest end-to-end figure at the sizes we can actually measure.

The running-max correction stays f64 `math.Exp` deliberately: once per (row, key block) rather
than per element, so it is free, and it rescales the whole accumulator — the one place the extra
accuracy is worth having. Gated at cosine on all three towers: SigLIP **1.00000000**,
Qwen2.5-VL **1.00000000**, Gemma 4 **1.000000000**.


**The encoder and vision towers now actually run the SIMD transcendental kernels — 15.9% off a
long text encode, 28.7-42.7% off a SigLIP tower (audit M-07, M-06, M-10).** Three findings with
one root: the `*Into` activation family (`SiLUInto`, `GELUInto`, `GELUTanhInto`,
`SoftmaxRowInto`) vectorises ONLY under `GOEXPERIMENT=simd` — a build this library cannot require
of importers, and one `docs/task-archsimd-eval.md` ruled out as a shipping path — while the
hand-written NEON/AVX2 kernels landed in v1.37.0 sit behind a separate `*ContractInto` API that
had **no callers outside `linalg/`**. Every softmax, SiLU, GELU and GELU-tanh in `encoder/` and
`vision/` ran the scalar Go loop in every shipped build, which `docs/internal/cpu-acceleration.md`
items 7-14 described as landed wins. Ten call sites are now routed. Separately, `swigluMLP`,
`swigluMLPQ8` and `layerNorm` in `encoder/`, and every norm and activation in `vision/`, ran
single-core while the matmuls around them fanned out; they now take a row split (a new
`vision.parallelRows`, the twin of encoder's).

Measured on `apple-m1pro`, quiet box, benchstat `-count=6`: `BenchmarkEncode_singleLong`
800.1ms -> 672.6ms (**-15.9%**, p=0.002); `BenchmarkSiglipTower/p196_h512` 149.7ms -> 85.7ms
(**-42.7%**), `/p576_h768` 830.5ms -> 591.8ms (**-28.7%**), geomean **-36.1%** (both p=0.002).

Numerics: the row splits are exactly bit-identical (every reduction is row-local — the
distinction from dead-ends §8.4, which rejected a *SIMD* layernorm because vectorising the fold
re-associates an f64 sum). The kernel routing is NOT bit-identical: measured max ABSOLUTE
deviation 4.8e-07 (SiLU) and 2.4e-07 (both GELUs) over 200k normal inputs, ~2 ULP at unit
magnitude. Raw ULP counts look far worse (up to 8.7e8) but that is a near-zero artefact — GELU
crosses zero, so tiny absolute deltas straddle the sign bit; absolute error is the right metric
for an activation feeding a matmul. Gated end to end: `TestSiglipEncoder_parity` cosine
**1.00000000** (max abs diff 1.669e-06), `TestQwenVisionEncoder_parity` cosine **1.00000000**
(6.706e-07), plus `TestGoldenFixture_cosine`, `TestBERT_parity`, `TestGTE_parity`,
`TestCrossEncoder_parity`, `TestSPLADE_parity` and `TestModelQ8_cosineMatchesF32`.

Also adds `linalg.SoftmaxRowScaledContractInto`, which did not exist — attention softmax is the
largest single term in M-07's bound, so the fused scaled form needed a contract entry. It applies
the scale as its own pass and calls the contract softmax unchanged, so cross-architecture
bit-identity comes free rather than having to be re-earned by a second fused kernel.

**The three vision fixture-generation scripts wrote to `scripts/testdata/`, not `testdata/`.**
`pin_siglip_vision.py`, `pin_qwen25vl_vision.py` and `gen_siglip_bench.py` resolved their output
as `dirname(__file__)/../testdata`, which is `scripts/testdata` — a directory no Go test reads
(`pin_bge.py` had it right, via `REPO_ROOT`). So following the skip message's own instruction to
"run scripts/oracle/pin_siglip_vision.py" did not fix the skip, and
`TestSiglipEncoder_parity`, `TestQwenVisionEncoder_parity`, the two Gemma 4 real-checkpoint
parity tests and `BenchmarkSiglipTower` had been silently skipping. With the paths fixed, the
regenerated goldens are **bit-identical** to the committed ones (the scripts are fully seeded),
`vision` goes from 29 passed / 5 skipped to 31 / 3, and `gpu/qwenmetal` from 1 / 3 to **4 / 0** —
which retroactively gives the v1.39.x C-02 threadgroup-budget fix real end-to-end coverage.


**Metal: an aborted dispatch is no longer silent (audit C-02).** `waitUntilCompleted` returns
CLEANLY from a GPU fault, and the fire-and-forget `Queue.Run1D`/`Run2D`/`Run1DBatch`/
`Run1DBatchTG` helpers committed, waited and returned without ever reading the command buffer's
status — so an aborted dispatch left the output buffer holding its previous contents and the
caller read that as a result. In the Qwen ViT that meant `e.att` keeping the previous layer's
values and the forward returning a plausible, wrong hidden state. New `mustCmdBufOK` latches the
terminal status at all four sites and panics with the `NSError`; it panics rather than returning
because those helpers return nothing and adding errors would break every caller, it matches
`MustBuf` (this package's existing answer to an unrecoverable device failure), and it composes
with the callers that now recover. `qwenmetal.ForwardViT` additionally declines up front when
`AttentionSeg`'s threadgroup scratch would exceed `MaxThreadgroupMemoryLength()`, which is the
documented obligation on that accessor. Note the real threshold is tighter than the audit's
estimate: counting the kernel's two STATIC threadgroup arrays (`smax`/`ssum`, `ViTBlock` floats
each) alongside the dynamic score row, a 32 KiB device caps the segment at **7680**, not 8192.

**`gpu/annmetal` recovers device OOM into an error instead of panicking (audit C-04).**
`NewI8Index`, `ScoreBatch` and `TopKBatch` allocate buffers large enough to exhaust device memory
— `TopKBatch`'s score matrix alone is 1 GB at N=1e6, M=256, the advertised shape — and the device
layer reports exhaustion as a loud `MustBuf` panic. `anncuda` has recovered that into an error at
all three of its sites since it shipped; `annmetal` did not, so an oversized corpus or batch
panicked out of `EnableGPU`/`QueryBatch` instead of honouring `ann.Backend`'s "falls through to
the tiers below on any error". Fixed at all three sites, and unlike `anncuda` the buffers are
declared before the deferred cleanup so a partial allocation is released rather than leaked — on
the one path where device memory is by definition already short. (`anncuda` should get the same
treatment; it needs a CUDA box to verify.)


**`bm25` WAND pruning no longer accepts `B > 1`, where its upper bound is unsound (audit C-03).**
`wandUsable` admitted any `B >= 0`, but the bound is evaluated at `minNorm` and assumes the
denominator `tf + K1*(1-B+B*norm)` keeps one sign across a posting list. For `B > 1` the `1-B`
term is negative, so on a corpus with skewed document lengths that denominator can cross ZERO
inside the list's range: the true impact is unbounded near the crossing and negative past it,
while the value computed at `minNorm` bounds nothing. Reproduced before the fix — 200 long
documents plus 20 two-token documents sharing a term, `K1=1.5 B=2.0` — `TopK` returned **zero**
results where the exhaustive scan returns ten. Silent and total, not a reordering. `B > 1` now
takes the exhaustive path like `K1 < 0` and `B < 0` already did; `0 <= B <= 1` (the standard
BM25 range, and every default) is unaffected. `TestWAND_bGreaterThanOneMatchesExhaustive` is the
regression, and `TestWAND_declinesWhenBoundUnsound` gained the two `B > 1` rows.

**`linalg.Workspace`'s weight-only Q8/Q4 dequant scratch is now separate from the activation-scale
buffer (audit C-05).** `QuantizeActivations` returns views into the Workspace's `f32` scratch and
documents their lifetime, but `MatmulBTQ8Into`/`MatmulBTQ4Into`'s serial paths took their
dequantized-weight-row scratch from that same buffer via `f32Buf`. When the buffer was already
large enough the slice did not GROW it — so the documented "valid until the next call that grows
the scratch" rule was satisfied while the scales were overwritten in place with dequantized
weight values. Measured on the fixture: held scales `[0.0225, 0.0190]` came back as
`[127, -57]`. Latent (no in-tree caller interleaves this today) but reachable by the `FlatI8`
paged-scan pattern or a mixed int8/W8A8 model on one Workspace. The spans now use a dedicated
`deqBuf`, and the doc comment states the real rule. `TestWorkspace_quantizeActivationsSurvivesWeightOnlyMatmul`
gates it — and warms the Workspace first, because against a cold one the growth allocates a
fresh array and the defect hides.


**`linalg.MatmulBTW8A8Batch` now reaches the S-01b W8A8 tile — 2.3-3.8x on the batched prefill
shapes (audit M-01).** `w8a8BatchSpan` carried its own column-outer loop over `dotI8`, so the
register-blocked tile that `w8a8Span` dispatches to at M>=4 was reached by the unbatched entry
and bypassed by the batched one. That is the wrong way round for the shape the tile was built
for: an `int8int8` prefill row batches q&#8214;k&#8214;v and gate&#8214;up — 30.6M of the 44.4M MACs per
Qwen2.5-Coder-1.5B layer, 69% — through the batched entry, and only down-proj (unbatched) got
the tile. `w4a8BatchOp` already routes through its span for exactly this reason; this is the
W8A8 twin. Measured on `apple-m1pro` (quiet box, benchstat `-count=6`) at goinfer's real prefill
shapes, K=1536: q&#8214;k&#8214;v -73.9% at M=4, -69.9% at M=128; gate&#8214;up -58.1% at M=4, -72.4% at
M=128; geomean **-59.4% (2.46x)**. Both M=1 controls are statistically unchanged (p=0.589,
p=0.065) — the single-token decode path does not move, which is the point. Bit-identical by
construction: `w8a8SpanRows` computes the same `float32(dotI8(...)) * aScales[i] * bScale` with
the same zero-scale short-circuit, and the tile accumulates in int32;
`TestW8A8Batch_bitIdenticalToPerOp` and `TestMatmulBTW8A8_MConsistent` gate it.
`BenchmarkMatmulBTW8A8Batch_prefill` is new and pins the shapes.

**`ann.FlatI8.Query` now shards at `Flat`'s measured threshold instead of goinfer's decode
threshold — up to 2.6x on mid-size corpora (audit M-19).** The pooled scratch `Workspace` was a
zero value, so it inherited the process-wide `parThreshold` (1<<24 MACs). That constant is
goinfer's *decode* tuning — chosen so tiny per-token projections stay serial — and it is 32x
`Flat`'s `flatParallelThreshold` (1<<19), which was measured on this exact one-query-against-n
scan shape. The int8 scan therefore ran single-core up to ~65k vectors at dim 256 while the f32
scan sharded from ~2k. One line: `sc.ws.SetThreshold(flatParallelThreshold)`. Measured on
`apple-m1pro` (quiet box, benchstat `-count=6`): every shape below the old threshold improves —
d256/N4k -16.4%, d256/N16k -51.9%, d256/N50k **-62.2%**, d768/N4k -36.5%, d768/N16k **-62.0%** —
and every shape already above it is unchanged (d768/N50k and d768/N100k both `~`). Geomean
-33.2%. Width-inert and so bit-identical: `SetThreshold` changes only the fan-out, and
`w8a8Span` produces the same result under any column split (`TestW8A8Batch_widthInert`).
`BenchmarkFlatI8QueryThreshold` is new and straddles the boundary. One unexplained residual:
d256/N100k measured +5.1% (p=0.015) on a path that is above the old threshold and should be
identical — small enough to read as drift, flagged rather than smoothed over.

## [1.39.1] — 2026-09-10

### Fixed

**`vision.Gemma4Encoder`: `vision_config.standardize=true` is now implemented, not refused.**
v1.39.0 shipped `Gemma4EncoderConfig.Standardize` as a config field but left the behavior it
gates unimplemented — `LoadGemma4Encoder` hard-refused any checkpoint that set it, confirmed
false only on the real E2B-it checkpoint at the time. The real `google/gemma-4-26b-a4b-it`
checkpoint sets it true (a genuine per-checkpoint split, not an edge case — confirmed directly
against its `config.json`), so every 26B-A4B/31B-class vision request was completely unusable
through this encoder until now. `Gemma4VisionModel.forward`'s own placement: `(hidden_states -
std_bias) * std_scale`, applied to the pooler's already root-hidden_size-scaled output,
immediately before the embedder's RMSNorm+projection — both buffers `[hidden_size]`
(`vision_tower.std_bias`/`std_scale` in the real checkpoint). Gated at cosine **1.000000000**
(max|diff| ≈1.4e-5) against the real 26B-A4B-it vision-tower weights on synthetic patches,
matched f32 precision both sides (`TestGemma4Encoder_realCheckpointParity_26B`, the twin of
v1.39.0's own E2B gate) — this checkpoint also exercises `use_clipped_linears=false` (the
unclipped ±inf-bound path), which the E2B gate does not. `TestGemma4Encoder_realCheckpointParity`
(E2B, `standardize=false`) is unchanged and still green — zero regression to the shipped path.
Found via goinfer's own real-checkpoint end-to-end VL serving gate hitting the refusal directly.

`perfgate` VERDICT: PASS — no regression vs v1.39.0 above each shape's floor — 4/10 shapes resolve
the 5.0% class (unsurprising: this release only adds a conditional per-hidden-dim elementwise
affine, never on the already-shipped `standardize=false` path and nowhere near the GEMV/matmul
kernels this gate measures).

`vulncheck` STATEMENT: no reachable vulnerabilities in 15/15 modules at nobara-pc, 2026-09-10.

## [1.39.0] — 2026-09-09

### Added

**`vision`'s third ViT family: `Gemma4Encoder`/`Gemma4Preprocess`, the "gemma4" plain family's
vision tower (E2B/E4B/26B-A4B/31B — not the separate, encoder-free `gemma4_unified` 12B family).**
Architecturally closer to this repo's own text decoders than to the other two towers: Gemma2/3-style
sandwich RMSNorm (four independent norms per layer, not SigLIP's two LayerNorms), a gated
SwiGLU-tanh MLP (not SigLIP's plain 2-layer GELU), axial 2-D RoPE on Q/K (a learned absolute
position table is ALSO applied additively at the patch-embed stage — both are real, verified
against the real forward pass, neither is dead code), q/k/v independently RMSNorm'd per head (v
unscaled), a hardcoded attention scale of 1.0 (not `head_dim**-0.5`), and every attention/MLP
projection wrapped in `Gemma4ClippableLinear`'s real input/output clamp — confirmed load-bearing
against the real `google/gemma-4-E2B-it` checkpoint (`use_clipped_linears=true` there, with genuine
finite per-tensor bounds like `[-6.375,6.3125]`, not the harmless ±inf a disabled clamp would carry).
3×3 average pooling runs once, after all 16 layers (the encoder itself runs at full 16px-patch
resolution); the projector is one unscaled RMSNorm + one bias-free Linear
(`Gemma4MultimodalEmbedder`). `Gemma4Preprocess` implements the real aspect-ratio-preserving tiling
rule (`get_aspect_ratio_preserving_size`) — resize to the largest multiple-of-48
(`pooling_kernel_size × patch_size`) dimensions within a caller-chosen soft-token budget
(`{70,140,280,560,1120}`), aspect ratio preserved by one uniform scale factor, each axis rounded
down independently.

Every load-bearing detail was checked against the real E2B-it safetensors header, not assumed from
source reading alone — this is what caught the real clip bounds above, and the position table's
actual on-disk shape (`[2,10240,768]`, one tensor spanning both axes, not two separate tables).
Gated at cosine **1.000000000** against a tiny-random `Gemma4VisionModel`+`Gemma4MultimodalEmbedder`
checkpoint (`TestGemma4Encoder_tinyGoldenParity` — exercises every component including the real
clamp) and cosine **1.000000000** (max|diff| ≈6.9e-6, pure f32 rounding noise across 16 layers)
against the real `google/gemma-4-E2B-it` vision-tower weights on synthetic patches, matched f32
precision both sides (`TestGemma4Encoder_realCheckpointParity`). The aspect-ratio tiling formula is
separately cross-checked numerically against the real HF function on five width/height/budget
combinations (`TestGemma4AspectRatioSize_matchesRealFormula`). Pixel-level resize parity (PIL
bicubic vs this repo's bilinear) is a known, explicitly deferred gap — the same one `encoder.go`'s
SigLIP preprocessing already carries — not attempted here.

CPU only; no `ResidentEncoder`/GPU backend yet (SigLIP and Qwen2.5-VL both have one). The decoder-
side integration (goinfer: an "embed-by-vector" seam for gemma4's own forward path, per-layer-
embedding PAD-token substitution at multimodal positions) and the full serving/generation wiring
are separate, goinfer-side work — see goinfer's `docs/multimodal.md` P7 entry.

`perfgate` VERDICT: PASS — no regression vs v1.38.0 above each shape's floor — 5/10 shapes resolve
the 5.0% class (unsurprising: this release adds new files calling existing `linalg`/`WeightMat`
kernels, none of which changed).

`vulncheck` STATEMENT: no reachable vulnerabilities in 15/15 modules at nobara-pc, 2026-09-09.

## [1.38.0] — 2026-09-08

### Changed

- **`vision`'s two ViT towers (SigLIP/`encoder.go`, Qwen2.5-VL/`qwen_encoder.go`) now run
  attention through `linalg.AttendTileFused`, head-parallel, instead of a serial
  materialize-QKᵀ→softmax→scores·V loop per head.** This is the same fused schedule and the
  same per-worker-serial-`Workspace` fan-out goinfer's own text-decoder prefill already uses
  (`decoder/forwardn.go`'s `attendBatchedHeads`) — ported to the tower rather than invented for
  it, including its documented gotcha: a head-parallel goroutine must route its matmuls through
  a private serial `*linalg.Workspace`, not the package-level `linalg.MatmulBT`, or it
  double-parallelizes against `MatmulBT`'s own internal column fan-out. Bidirectional attention
  (every row's key range is the whole tile) needed no change to `AttendTileFused` itself —
  `lo=0, hi=n-1` for every row was already handled by its existing block-skip logic. Qwen's
  `cu_seqlens` window/full-attention segmentation is unchanged; fusion applies per segment,
  inside the existing segment loop, with a per-worker scratch buffer resized (not reallocated)
  to the longest segment in the call.
  Both existing real-checkpoint parity gates hold at their current thresholds — `vision`'s
  fused schedule is documented as *not* bit-identical to the materialized path it replaces (the
  running-max rescale re-associates the softmax denominator), so this was a real measurement,
  not an assumption: `TestSiglipEncoder_parity` (f32 cosine 1.00000000, was already ≥0.9999;
  int8 W8A8 0.99996609, ≥0.99) and `TestQwenVisionEncoder_parity` (pre-merge and merged f32
  cosine 1.00000000 on both, ≥0.9999999 — the tightest bar in the suite) both pass unchanged,
  confirmed additionally under `-race`. First slice of `docs/multimodal.md`'s P6 ("route the
  tower through existing kernels") — CPU only; CUDA and Metal need genuinely new non-causal
  kernels and a new tower weight-quantization pipeline, out of scope here and tracked
  separately.

## [1.37.0] — 2026-09-06

### Added

**A numeric contract for the f32 transcendentals, and hand-written NEON and AVX2 kernels that
obey it.** Five new entry points — `ExpContractInto`, `SoftmaxRowContractInto`, `SiLUContractInto`,
`GELUTanhContractInto`, `GELUContractInto` — each selecting the fastest conforming implementation
for the build.

**The contract is the point, not the kernels.** Every multiply-add in these paths is a
correctly-rounded f32 fused multiply-add: on every architecture, in the assembly and in the Go
fallback alike. That matters because `exp.go`'s existing Horner chains are *not* pinned — Go fuses
them into FMADD on arm64 and does not below `GOAMD64=v3` on amd64, so the same source has been
producing different bits per architecture, and the file's own comment already recorded the
symptom.

Go has no f32 FMA (`math.FMA` is float64-only), so the Go path reaches one through f64. That is
exact rather than approximate, for two reasons, and the second is what makes it portable:

1. an f32×f32 product needs ≤48 significand bits and f64 has 53, so the product is **exact** in
   f64 — and therefore whether the compiler contracts the inner mul-add is *irrelevant*, since
   rounding an exact product is the identity;
2. rounding the f64 sum on to f32 is a double rounding, 53→24, innocuous because the intermediate
   carries ≥2p+2 = 50 bits.

Verified against `math/big` at 200-bit precision on adversarial inputs — 60 binades, plus cases
forcing total cancellation and extreme scale mismatch, which is where a double rounding would show.

**The cross-architecture golden is the result to take away.** Every other gate here proves a
kernel matches its scalar oracle *on the machine it ran on*. `TestContractCrossArchGolden` hashes
every entry point's output over a fixed deterministic input set and compares it to a **constant** —
the only way to compare across instruction sets, since no process can execute both. That constant
was computed on an M1 running NEON kernels and it passes on a Zen 2 running AVX2 kernels. **A
golden generated on one machine is valid on the other.**

Speed, n=8960 (one gate row), median of 3:

| kernel | M1 Pro | 3700X |
|---|--:|--:|
| exp vs the f64 `math.Exp` consumers call today | **11.8×** | **27.6×** |
| exp vs the shipped f32 `expF32Core` | 4.3× | 8.8× |
| SiLU vs f64 | 10.5× | 6.6× |
| softmax vs shipped `SoftmaxRowInto` | 2.1× | — |
| GELU-tanh vs shipped `GELUTanhF32` | 3.7× | **3.6×** |
| GELU-erf vs shipped `GELUF32` | 3.1× | **5.1×** |

**Softmax pins its summation order**, because a denominator is a reduction and its value depends
on the order of the adds: four f64 lane partials, element *i* into `partials[i&3]`, folded by the
fixed tree `(p0+p1)+(p2+p3)`. The scalar reference is written that way too, even though nothing
forces a Go loop to be, so that it can serve as the oracle.

**`GOAMD64` needs no pinning for these paths**, which answers a finding the SIMD audit raised as
open. Dispatch is a runtime check (`hasAVX2`, which already requires FMA3 and OS YMM state) and the
scalar fallback is bit-identical, so a v1 build and a v3 build produce identical output and differ
only in speed. That is stronger than a pin, because a pin can be forgotten by whoever builds the
binary.

**What this is worth end to end is much less than the kernel ratios, and is measured rather than
extrapolated.** On arm64, swapping a consumer's f64 transcendentals for these moved CPU prefill by
**1.03–1.09×** — 3–8% — because the transcendentals were never more than a tenth of the work once
the fan-out was parallelised. Both numbers are true at once: the kernels are ~10× and the system is
~1.05×. The amd64 end-to-end figure is not yet measured.

**Both architectures now have the full kernel set.** tanh and erf were initially left on the
scalar contract on amd64 on the grounds that they are the gemma-family activations, off the path
the share measurement covered. That was a deferral rather than a reason, and building them cost
little: the expensive parts — the contract, the oracle, the golden — already existed, so the
marginal work was the assembly plus a bit-identity gate that the golden then checked for free.
The amd64 numbers above are the result. Every entry point is now accelerated on both.

Nothing in this package routes to the new entry points yet; they exist for consumers to adopt
deliberately, since adoption changes bits by ≤4 ULP and is a goldens decision.

### Fixed

**The NEON tanh kernel returned −0 where the contract returns +0.** `tanhF32ContractNEON` took the
result's sign from x's sign BIT, but the scalar contract branches on `x < 0` — and −0 < 0 is false,
so the contract treats −0 as positive and returns +0. Exactly one input in the whole f32 domain,
and the kernel's own comment asserted the opposite ("both give a signed zero"). Both kernels now
clear the sign where |x| is zero, via an integer compare that is exact because |x| has all-zero
bits only for +0.

**No caller was affected**, and that is measured rather than argued: the only public consumer of
the tanh kernel is `GELUTanhContractInto`, whose wrapper multiplies by 0.5·x — which is −0 at that
input regardless — so both the old and the new kernel produce −0 there. The defect was confined to
the kernel-versus-contract invariant.

**Why it shipped, which is the part worth keeping.** Three separate gates cover this kernel and
none of them contained −0: every input set was built by arithmetic (`base + d*step`, a seeded
normal draw) and none of those expressions ever yields a negative zero. It surfaced only when the
amd64 twin was written and its test happened to include `math.Copysign(0, -1)`. Both zeros are now
explicit in the NEON tests and in the cross-architecture golden's input set — which is why
`contractGolden` changes in this release. **A golden is only as good as the inputs it is computed
over, and a hash cannot tell you what it never hashed.**

Also moved `BenchmarkGELUTanhKernels` and `BenchmarkGELUErfKernels` out of the arm64-only bench
file. They call nothing arch-specific, and sitting there meant amd64 had no measured number for
the tanh/erf side of the contract at all.

### Release gates

`releasegate`:

```
VERDICT: PASS — v1.37.0 — 4/4 checks passed
```

`vulncheck`, `nobara` (Ryzen 7 3700X, linux/amd64):

```
STATEMENT: no reachable vulnerabilities in 15/15 modules at 8f053c2 (2026-09-07T01:04:27Z)
```

`perfgate`, `nobara`, working tree vs v1.36.0 interleaved. **The first run FAILED and it is
recorded here rather than replaced by the passing ones**, because a gate that is re-run until it
goes green is not a gate:

```
run 1: VERDICT: FAIL — 1 regression(s) vs v1.36.0 across 10 shapes
run 2: VERDICT: PASS — no regression vs v1.36.0 above each shape's floor — 4/10 shapes resolve the 5.0% class
run 3: VERDICT: PASS — no regression vs v1.36.0 above each shape's floor — 4/10 shapes resolve the 5.0% class
```

The flag was `BenchmarkW8A8SpanShapes/K2048_N2048` at **+8.35%** against a ±4.39% floor. Judged
noise on three independent grounds, in increasing order of how much they actually prove:

1. **No source change can explain it.** Every commit in `v1.36.0..8f053c2` adds new `linalg/exp_*`
   files — 3585 insertions, zero deletions, nothing within reach of the W8A8 span path.
2. **The same binary varies by as much as the effect.** `cur` for that shape read 0.117ms, then
   0.106ms, then 0.105ms across the three runs. The 8% being attributed to a code change is
   inside the run-to-run spread of code that did not change.
3. **The delta flipped sign at the same magnitude**: +8.35%, −0.79%, −8.29%. A real regression does
   not come back as an equal-and-opposite improvement.

Note also that runs 2 and 3 do NOT by themselves refute run 1: their floor at that shape widened to
±18.96% and ±11.34%, which is blind to the 5% class. The refutation rests on the reasoning above,
particularly (2) — not on "two out of three were green."


## [1.36.0] — 2026-09-06

> **Release evidence.** `perfgate` was RUN for this tag, unlike v1.35.0's recorded exception:
> `VERDICT: PASS — no regression vs v1.35.0 above each shape's floor — 4/10 shapes resolve the
> 5.0% class`. Read that as the tool's own sensitivity line asks: it is BLIND to the 5% class on
> six shapes (`K2048_N2048` ±13.6%, `K4096_N4096` ±10.4%, `K1536_N8960` ±8.6%, `K2048_N2048`
> ±25.7%, `K3584_N4096` ±34.8%, `K768_N8192` ±12.3%), so the green there is only evidence against
> a larger regression. The four shapes that DO resolve 5% were flat: −0.10%, −0.04%, −0.03%,
> +0.70%.
>
> `STATEMENT: no reachable vulnerabilities in 15/15 modules at a0d92b4 (2026-09-06T17:16:01Z)`.

### Added

- **`embed`: MXFP4 (OCP FP4, ggml type 39) exported, in BOTH block layouts** — `MXFP4Scale`,
  `DequantMXFP4Blocks` (contiguous GGUF 17-byte blocks), `DequantMXFP4Split` (the safetensors
  form, where nibbles and scales are separate tensors), `MXFP4BlockElems` / `MXFP4BlockBytes`.
  M2 of `docs/task-goinfer-kernel-moves.md`.

  **The two layouts are not interchangeable and must not be unified.** They share the block size,
  the scale encoding and the value table, so they read as one kernel with different addressing;
  byte j packs elements j and j+16 in GGML and 2j and 2j+1 in safetensors. Measured on a real
  gpt-oss expert against the validated GGUF path: **cosine 0.081 for GGML order on safetensors
  data, 1.000000 for sequential.** The wrong choice produces finite, plausibly-scaled, completely
  wrong weights without erroring. `TestMXFP4Orders_areNotInterchangeable` guards this.

  Gated raw-bit (`math.Float32bits`, no tolerance) against frozen copies of goinfer's bodies over
  all 256 e8m0 scale bytes and every 4-bit code; mutation-checked three ways.

### Fixed

- **`embed`: a comment on `e8m0ToF32Half` claimed something untrue.** It said the exact bit
  formula, rather than `2^(x-128)`, is what keeps the x∈{0,1} subnormals bit-identical to the
  reference. Measured over all 256 inputs, the two forms agree **everywhere** — 2^-128 and 2^-127
  are exactly representable as float32 subnormals. No behaviour change; the formula is still the
  right choice for being exact by construction with no libm call. The old wording would have told
  anyone simplifying it that they had broken something they had not.

## [1.35.0] — 2026-09-06

> **RELEASE-RITUAL EXCEPTION: `perfgate` was NOT run for this tag.** RELEASING.md step 2b calls
> for an interleaved working-tree-vs-previous-tag benchmark on a real box; it was skipped by
> operator decision because the box was needed elsewhere. **This tag therefore carries no
> performance evidence.** The argument for it being safe is that the change is purely additive —
> a new file, and nothing that already existed calls it — so no existing benchmark's code path
> moved. That is an argument, not a measurement, and it is exactly the shape of reasoning
> v1.17.0 was shipped on: a kernel that was numerically identical, passed every correctness
> test, and cost a consumer ~3% at a streamed shape.
>
> To close it retroactively: `git checkout v1.35.0 && go run -C tools ./perfgate` (it compares
> the tree against the previous tag, so it must be run FROM this tag to mean anything). If it
> comes back red, the remedy is a v1.35.1, not a silent edit.

### Added

- **`linalg.AttendTileFused` — a FlashAttention-style single-tile attention schedule**
  (Experimental), with `FusedAttnScratch`/`NewFusedAttnScratch`/`Fits`, `FusedAttnKeyBlock` and
  `GatherVBlockMajor`. It blocks over KEYS and folds each block into the output accumulator with a
  running max and a running sum, so the `kt × nKeys` score matrix never exists — the materialized
  schedule makes three trips through a block that is megabytes at a production tile budget.

  **It is deliberately NOT bit-identical to a materialized QK→softmax→AV**: the running-max rescale
  re-associates both the softmax denominator and the AV fold. A caller that needs bit-identity must
  not use it. That is why it is Experimental in the strong sense, and why the gate compares it
  against a frozen copy of *itself* — "the fused schedule computes what it computed yesterday" is
  the property that actually travels — with a separate cosine ≥ 0.9982 bound against
  `MatmulQKAcc64`+softmax+`MatmulAVAcc64` recorded as a sanity check and explicitly not a bit gate.

  Scratch is an explicit argument, never an internal pool: a caller running one per worker owns the
  lifetime, and a hidden pool would either allocate per call or serialise them.

  Moved from goinfer (`decoder/fusedattn.go`, P19) **body verbatim** — including two `f64 math.Exp`
  calls inside an `f32` path, which are part of the bits rather than an oversight to tidy. The gate
  proves it: substituting `linalg.ExpF32` turns the raw-bit test red. Gated over wide binades
  (2^-60..2^60, so the running max moves and the rescale path is exercised), denormals, head dims
  64/128/256, and `nKeys` either side of the 512 block boundary (511/512/513/1024/1025) plus
  K%4/K%8/K%32 tails; mutation-checked twice.

## [1.34.0] — 2026-09-03

### Added

**`MatmulBTW4A8Batch` — several int4 matmuls sharing one activation, under one fork/join.** The
W8A8 path has had `MatmulBTW8A8Batch` for a while; this is its int4 counterpart, for the same two
fusions a transformer layer offers: q‖k‖v and gate‖up. The activation is quantized once and the
goroutine fan-out is amortized across every op's columns (the concatenated `[0, ΣN)` column space
is split across workers) instead of one quantize and one barrier per matmul. Weights stay in
place, so a caller aliasing int4 weights zero-copy gets the dispatch reduction with no concat
copy. Numerically identical to calling `MatmulBTW4A8Into` once per op.

**It was measured before it was written, and the measurement changed the design twice.**

*First*, which remedy. `docs/task-simd-audit.md` §S-02 names two — this batch form and dynamic
chunking (workers pulling blocks from an atomic counter, which needs no API at all) — and says to
find out which dominates before building either. Per-shard timestamps across one fan-out show a
**start spread of 92.6 µs against a median shard duration of 57.7 µs**, with durations uniform to
1.07×: goroutine-wake stagger, not P/E-core shard skew, which would look the opposite. Putting
more work under one barrier therefore amortizes it, and does — at six workers, GB/s by matrices
per fork/join: 1 → 58.8, 2 → 68.8, 3 → 74.0, 8 → 87.4. The control that makes those numbers
readable is the single-worker sweep, **flat to within 1%**, because with one worker there is no
stagger to amortize; without it the six-worker climb could have been cache residency. Dynamic
chunking, measured in the same harness, reaches 1.135× — about half of what three matrices per
barrier gives, so it is complementary rather than a substitute, and still worth taking separately.

*Second*, the signature. Mirroring `W8A8Op` exactly — canonical packed weights only — would have
been a **regression** for the caller this exists to serve: on arm64 an int4 tensor that has been
through `RepackInt4Row4` runs the split-half 4-row kernel at ~42 GB/s against the canonical
kernel's ~24, so a canonical-only batch would have traded a 1.75× kernel loss for a ~1.2× fan-out
gain. `W4A8Op` therefore carries optional `Row4`/`Row4Scales` and the batch keeps the fast layout.
Column edges that are not quad-aligned fall back to canonical, which is a dispatch choice and
never a numeric one — the two layouts are bit-identical for the same logical weights.

Measured against what actually ships (N separate `MatmulBTW4A8Row4Into` calls, not the canonical
kernel, which would have credited the batch with a layout win it did not earn): **q‖k‖v 1.118×,
gate‖up 1.209×** at six workers on an M1 Pro.

**What this release does NOT establish is the end-to-end decode win**, and that is stated plainly
because the number is not ours to produce. S-02's ship gate is the paired 1.5B decode cell at
≥1.15× (park below 1.05×), which lives in the consumer. The per-layer saving projects to roughly
1.5 ms of a ~25 ms token — about **1.06×**, between park and ship. This is tagged so that
projection can be replaced with a measurement against a real version rather than a pseudo-version;
if it lands short, the honest follow-up is dynamic chunking on top of it, not a looser reading of
the gate.

Bit-identity is inherited rather than re-argued: the batch span hands each op's column slice to
the same `w4a8Span` the single-matrix entry point uses, so amd64's M≥4 row tiling keeps working on
the batched shape instead of being bypassed (verified by canary, not assumed). Gates cover row4
and canonical ops, M>1 where row4 declines, an `N%4≠0` op inside a batch whose others qualify, a
ragged K, the serial fast path, a forced fan-out, and width-inertness across six worker counts.


### Release gates

`vulncheck`, run on `nobara` (linux) at `daeeff9` — the prep commit above it changes only
`CHANGELOG.md`:

```
STATEMENT: no reachable vulnerabilities in 15/15 modules at daeeff9 (2026-09-04T04:28:15Z)
```

`perfgate`, `nobara` (Ryzen 7 3700X, linux/amd64, load 0.00), working tree vs v1.33.0 interleaved:

```
VERDICT: PASS — no regression vs v1.33.0 above each shape's floor — 5/10 shapes resolve the 5.0% class
  BLIND on 5 shape(s): K2048_N2048(±11.3%) K4096_N4096(±8.3%) K1536_N8960(±11.3%)
                       K2048_N2048(±13.6%) K3584_N4096(±29.0%)
```

The v1.33.0 notes recorded that perfgate had silently measured against the wrong baseline because
the benchmark box had never fetched the new tag. `git fetch --tags` was run on that box first this
time, and the header line confirms the intended reference (`daeeff9 vs v1.33.0`) rather than
leaving it to be inferred from a green.

`releasegate`:

```
VERDICT: PASS — v1.34.0 — 4/4 checks passed
```

Note what perfgate does and does not cover here. Its instrument is the W8A8 span and GEMV
benchmarks at M=1, where `MatmulBTW4A8Batch` is not on the path at all — so this green is evidence
that the new API disturbed nothing, NOT evidence about the API's own speed. The numbers for that
are in the Added section above, measured separately on an M1 Pro, and the end-to-end decode figure
remains unmeasured by design.


## [1.33.0] — 2026-09-03

### Changed

**Three NEON kernels from the SIMD audit, all bit-identical to what they replace.** No API
changes; every gain is behind existing entry points, so consumers get them on a dependency bump.

**S-03 — the activation quantiser is no longer scalar.** `quantizeRowInt8Core` runs `FABS`/`FMAXNM`
for the abs-max and `FMUL`/`FCVTAS`/`SMIN`/`SMAX`/`SQXTN` for the quantise, with the original body
kept verbatim as `quantizeRowInt8CoreScalar` and used as the test oracle. Measured on an M1 Pro,
interleaved, median of 3: **12.99× at K=1536** (288 ns vs 3741) and **19.23× at K=8960** (1.51 µs vs
28.96). It runs before every W8A8/W4A8 matmul on the calling goroutine — 509k elements per token on
a 1.5B model — so it was 2–6% of a token spent in a scalar loop between parallel matmuls. Corner
cases are pinned against the scalar (NaN, ±Inf, −0, denormals, exact ties, saturation, all-zero).

**S-04 step 2 — the f64 attention kernels get NEON lane-per-output ports.** `MatmulAVAcc64` gains a
32-dim-block kernel (16 lane-pair accumulators, `FMLA` by element, `FCVTN` narrow) and
`MatmulQKAcc64` a 16-key kernel (lane = key, `ZIP1`/`ZIP2`, d ascending); the Go paths resume from
the index the kernels return and finish the tails, so a `K%4≠0` shape stays entirely in Go. Against
the Go accumulator-block form shipped previously:

| depth | AV | QK |
|--:|--:|--:|
| 130 | 2.32× | 1.68× |
| 2048 | 2.49× | 2.01× |
| 8192 | 2.46× | 1.82× |

The depth-8192 attention component — QK+AV, which is ~75% of a token at that depth — goes
**486.6 µs → 227.7 µs, 2.14×**. The audit's estimate for a NEON port was "~300 µs per head-call at
depth 8192"; the port beat its own bound.

Bit-identity here is structural rather than incidental: both operands are f32 widened to f64, so
every product is exact, and a lane running the same ascending-order chain is identical to the
scalar `FMADDD` one. `MatmulAVAcc64`/`MatmulQKAcc64` are checked against an independent strided
oracle at production shapes including nKeys=8192.

**S-07 — weight-only `int8` mode stops running one FMLA chain.** `q8Span` dequantises eight weight
rows into a scratch and uses the eight-column kernel, whose per-row lane partials and left-to-right
fold are the identical sequence `dotNEON4` produced. Measured 3.34× / 3.30× / 3.22× at M=1 across
K1536_N1536, K1536_N8960 and K2048_N152064, and **4.99× at M=4**. The parallel path's per-worker
`make([]float32, K)` is replaced by a pooled scratch on every architecture — that allocation was
~16 MB per token in `int8`/`int4Mix` modes.

### Fixed

**The row4 entry points now reject `K < group` by name.** `K=0` satisfies the `K%group==0` check
that guarded them, so it reached the span with `nGroups=0` and failed as an index-out-of-range on a
kernel internal instead of naming the caller's shape. Both `MatmulBTW4A8Row4Into` and
`MatmulBTW4A8Row4TileInto` now report it; the gate asserts the message rather than merely that a
panic occurs, because what changed is *which* panic.


### Release gates

`vulncheck`, run on `nobara` (linux) at `ffacb84`:

```
STATEMENT: no reachable vulnerabilities in 15/15 modules at ffacb84 (2026-09-03T20:09:30Z)
```

`perfgate`, `nobara` (Ryzen 7 3700X, linux/amd64, load 0.02), working tree vs v1.32.0 interleaved:

```
VERDICT: PASS — no regression vs v1.32.0 above each shape's floor — 5/10 shapes resolve the 5.0% class
  BLIND on 5 shape(s): K2048_N2048(±14.4%) K4096_N4096(±7.2%) K1536_N8960(±11.6%)
                       K2048_N2048(±19.6%) K3584_N4096(±12.8%)
```

**Operational note for the next release: perfgate picks its baseline from the tags the box can
see, and it does not say which tag it wanted.** The first run of this release compared against
**v1.31.0**, silently, because the benchmark box had `main` pulled but had never fetched the
`v1.32.0` tag — the header line `working tree vs v1.31.0` was the only tell. It was re-run against
the right baseline after `git fetch --tags`. A gate that quietly measures against the wrong
reference reads exactly like a green one, so `git fetch --tags` on the perf box belongs in step 2b.

`releasegate`:

```
VERDICT: PASS — v1.33.0 — 4/4 checks passed
```

The kernel measurements quoted above were taken on the M1 Pro rather than this box — they are
arm64 assembly and amd64 keeps its existing paths — and that machine was carrying roughly 1.3
cores of ordinary desktop load, not idle. Single-threaded arms on a 6 P-core part, interleaved
A/B and median-of-3 absorb that; the ratios are the trustworthy part, not the absolute ns.


## [1.32.0] — 2026-09-03

### Added

**A register-blocked W4A8 tile for arm64, and int4 prefill gets 2.88× on the kernel.**
`MatmulBTW4A8Row4TileInto` reduces four activation rows against the four weight rows of one
row4 quad in a single kernel call — 16 outputs, 16 live f32 accumulators, the register budget
exactly filled. `WeightMat.MatmulBTW4A8Into` routes every M>1 to it automatically when the
tensor is row4-resident (`RepackInt4Row4` / `WrapInt4Row4`); M=1 decode is untouched and still
takes `dotW4A8SplitHalf4Row`. Non-arm64, non-DotProd cores, non-repacked tensors and paged-MoE
spans all keep the canonical path, unchanged.

What this removes is repetition, not arithmetic. The canonical M>1 path is a GEMV per
(activation row, weight row) pair, so the 16-byte weight load, the four-op nibble unpack and
the per-group scale broadcast are each paid M times over the same weight bytes. That is why
`int8int8` prefill has been beating `int4` by 25–33% despite moving twice the bytes: at M>1 the
byte saving is served from cache anyway while the unpack ALU cost stays. The tile pays the
unpack once per weight row per group regardless of M — 0.19 SIMD µops per MAC against 0.34.

Measured on an M1 Pro at K=1536 N=8960 (the Qwen2.5-Coder-1.5B gate/up projection), three
interleaved paired passes, median: **2.88× single-core at M≥4** (24.1 → 69.5 GMAC/s) and
**2.42–2.62× on the parallel dispatch** (96 → 251 GMAC/s aggregate). M=1 and M=2 read 1.74×
because below M=4 the tile body never runs — those rows take the shipped four-weight-row kernel
one at a time. The canonical arm sits flat at 24.1 GMAC/s at every M, reproducing its own
previously recorded 24.5–25.0, so this is a same-box A/B rather than a comparison against
another machine's numbers.

**Bit-identical to the canonical path, by construction and by gate.** For each of the 16
outputs the per-group instruction sequence is the one `dotW4A8SplitHalf4Row` already runs —
zero, SDOT low, SDOT high, SCVTF, FMLA by the broadcast group scale into a persistent 4-lane
accumulator, in ascending group — and the epilogue is the same FADDP tree. Nothing is
reassociated; only sharing changes. `TestMatmulBTW4A8Row4TileInto_bitIdenticalToCanonical`
holds it against `MatmulBTW4A8Into` over M ∈ {1..9, 12, 13} at both production projections,
with companions for width-inertness and the zero-activation-row shortcut.

**A register-blocked W8A8 tile too, and this one needs no opt-in.** `dotI8Tile4x4` reduces four
activation rows against four weight rows into 16 int32 accumulators, and `w8a8Span` hands it the
largest 4×4 rectangle of every span on arm64 — so every W8A8 caller at M≥4 gets it with no layout
change, no repack and no API change. Below M=4 (decode, and the ANN scan's M=1 queries) nothing
changes at all.

`dotI8SDOT` sat on two near-coincident limits: four int32 accumulators at ~4-cycle SDOT latency
cap it at 1 SDOT/cycle, and its 2 loads per SDOT need 2 of 3 load slots to sustain even that.
Sixteen accumulators clear the first, and sharing each load across the other tile dimension takes
the ratio to 8 loads per 16 SDOTs, clearing the second. Measured serial on one core, median of
three interleaved passes: **3.5–3.9× at M≥8 across every shape tested**, from a 6 MB
cache-resident B to a 73 MB streamed one. At M=4 — the one batch size with no reuse in the tile's
inner loop — the ratio falls with K, from 3.05× at K=3072 to 1.69× at K=5120; that is the worst
cell measured anywhere, and it is still a 1.69× win.

The multi-stream regression this could have been was checked for explicitly rather than reasoned
away. A deleted eight-column kernel (`dotI8Cols8`, v1.17.0) was measured at one cache-resident
shape, shipped on it, and lost badly wherever B streamed. The tile advances four B streams, so
the grid deliberately straddles the LLC — and the streamed rows came back as strong as the
resident ones.

Bit-identity here is free rather than argued: every partial is int32, integer addition is exact
and associative, and the int32 overflow envelope is `dotI8SDOT`'s own, unchanged.

**`TestSDOTIssuePeak` measures SDOT issue width directly: 3.98 per cycle on an M1 Pro P-core**
(12.73 G SDOT/s, 203.6 GMAC/s of int8), settling a constant that had been carried as an
assumption. It asserts the ceiling and never a floor — a reading above four pipes means the loop
was folded or the clock constant is wrong, which is the failure that has produced bogus bandwidth
numbers before, while a low reading on a busy box is not a defect and must not fail CI.

**Both tiles now have amd64 halves, measured on an idle Ryzen 7 3700X.** `dotW4A8Tile4RowAVX2`
gives the W4A8 prefill path **1.65–1.90×** (15.6 → 27.6 GMAC/s at K=1536 N=8960) and
`dotI8Tile4x1AVX2` gives W8A8 **1.17–1.44×**, both at every shape and batch size tested and with
no regression anywhere. Neither is arm64's shape: AVX2 has 16 YMM registers against NEON's 32, so
sixteen live accumulators plus their operands does not fit and the amd64 tiles block only the
activation dimension.

The W4A8 tile is excluded on AVX-512 VNNI hosts, and that exclusion is about correctness rather
than speed: `dotW4A8` prefers the VNNI kernel, which folds through two f32 accumulators where the
AVX2 kernel uses one. An AVX2-based tile running at M>1 while M=1 kept the VNNI kernel would make
the result depend on M. Neither project box has VNNI, so this is ruled out by construction.

**The W8A8 amd64 tile's first shape was measured and thrown away, which is the part worth
reading.** A 4-activation × 2-weight version costs fewer instructions per MAC (0.172 vs 0.203) and
measured 1.34–1.52× on cache-resident B — then **0.70× and 0.57× on streamed B**, a 1.75×
regression. That is precisely the failure that got the eight-column `dotI8Cols8` deleted in
v1.17.0: a kernel that interleaves weight streams where the span it replaces walks one. Measured
on the prefill cell alone it would have read as a clean 1.41× and shipped. The grid straddling the
LLC is what caught it. The shipped shape is 4 activation rows × one weight row, which leaves B's
access pattern exactly as it was — the redo `w8a8Span`'s own comment had asked for.

Worth recording alongside it: arm64's tile advances *four* weight streams and shows no such
effect. "Fewer streams" is not the rule. Measuring both cache regimes is.

The amd64 ceiling is low and was predicted before the first run: VPMADDWD stays at one per 16 MACs
however the loop is blocked and issues on a single Zen 2 port, so ~67 GMAC/s binds and `dotI8AVX2`
already sat at 72–76% of it. arm64's 3.5–3.9× does not transfer — SDOT does 16 MACs on four pipes,
VPMADDWD does 16 on one.

**Validated end to end, in the consumer.** goinfer measured its CPU prefill cell against these
kernels on a quiet M1 Pro (1.5B q4_k_m, paired and interleaved with rotating arm order, n=5,
spreads 0.9–5.6%): **1.66–1.75× before/after**, and — the thing this work existed for — int4
prefill now beats `int8int8` by 1.10–1.14× at both depths, where int8int8 had been winning by
25–33%. That inversion was the symptom; paying the nibble unpack once per weight row instead of
once per activation row was the cause. The 2.88× kernel win compressing to 1.66× end to end is
expected: prefill is not all matmul, and neither the fork/join nor the serial f32 transcendentals
were touched here.

Bit-identity held across the bump in the consumer's own gates, not just aikit's:
`TestForwardN_matchesSequential` came back bit-identical across 19,447,808 logits, and
`TestMoEExpertMajor_bitIdentical` green over 56 expert-major chunks — the cross-repo confirmation
that `decode == batched prefill == speculative verify` survives a change that re-shaped how every
M>1 quantized matmul is computed.

**New M-invariance gates for the quantized kernels**, landed before the tile rather than with
it: `TestMatmulBTW4A8_MConsistent`, `TestMatmulBTW8A8_MConsistent` and
`TestWeightMatW4A8_MConsistentAcrossRow4Dispatch` pin that an output row computed inside an
M-row batch is bit-identical to the same row computed alone at M=1. `TestMatmulBT_MConsistent`
had been the only M-invariance gate in the package and it covers f32 `MatmulBT` only. The third
is the one that matters most: `WeightMat.MatmulBTW4A8Into` sends M=1 to one kernel and M>1 to
another, which is exactly the pair speculative verify exercises when a draft proposes at M=1 and
a target verifies at M=K.


### Release gates

`vulncheck`, run on `nobara` (linux) at `0f55bae` — the prep commit above it changes only
`CHANGELOG.md`:

```
STATEMENT: no reachable vulnerabilities in 15/15 modules at 0f55bae (2026-09-03T04:49:50Z)
```

`perfgate`, `nobara` (Ryzen 7 3700X, linux/amd64), working tree vs v1.31.0 interleaved. **Run
twice, both recorded, because the first run FAILED — and unlike v1.31.0's failure, this one was
REAL:**

```
run 1 (faf4b80)  VERDICT: FAIL — 3 regression(s) vs v1.31.0 across 10 shapes
                 W8A8SpanShapes/K1536_N8960       Δ=+15.53%  floor=±12.83%  REGRESSION
                 W8A8SpanShapes/K768_N8192        Δ=+8.00%   floor=±5.43%   REGRESSION
                 GEMV_W8A8_baseline/K4096_N4096   Δ=+11.18%  floor=±11.08%  REGRESSION
                 sensitivity: 4/10 shapes have a floor ≤ 5.0%

run 2 (0f55bae)  VERDICT: PASS — no regression vs v1.31.0 above each shape's floor
                 W8A8SpanShapes/K1536_N8960       Δ=-0.36%   floor=±4.75%   flat
                 W8A8SpanShapes/K768_N8192        Δ=+1.50%   floor=±2.00%   flat
                 GEMV_W8A8_baseline/K4096_N4096   Δ=+3.84%   floor=±8.75%   flat (BLIND)
                 sensitivity: 7/10 shapes have a floor ≤ 5.0% (the class this gate targets)
                 BLIND on 3 shape(s): K4096_N4096(±8.7%) K2048_N2048(±16.8%) K3584_N4096(±14.9%)
```

**The two runs are separated by a commit that fixes a real defect, which is the whole difference
from v1.31.0's double run.** There, a flagged shape reversed sign by 12 points on identical code
and its floor moved 4×, and the correct reading was shape instability. Here the flagged shapes did
not reverse on identical code — they were fixed. On the path where the S-01b tile DECLINES (M<4,
i.e. every decode call) `w8a8Span` was still making its leftover-column call with a zero-height row
range, and because that loop iterates columns on the outside, an empty row range still walked all N
columns constructing slices for zero rows. Every correctness test passed the whole time, because
the results were right, and the A/B harnesses for the tile could not see it because they measure
M≥4 where the tile engages.

The tell that this was a fix rather than a re-roll is that `K1536_N8960` did not merely go green —
it went from +15.53% at a ±12.83% floor to **−0.36% at a ±4.75% floor**, moving from BLIND into the
5% class it now resolves. A shape whose floor tightens 2.7× while its delta collapses to zero is
not drifting.

`TestSpanRows_emptyRowRangeVisitsNoColumns` gates it with no timing assertion: it hands the span a
weight slice deliberately too short for the column range, so a function that walks columns panics
and one that returns early cannot. Verified to fail on both spans with the guard removed.


## [1.31.0] — 2026-08-31

### Added

**`Tensor.Uint8s` now admits the fp8 dtypes, `F8_E4M3` and `F8_E5M2`.** This is the accessor's
existing rule applied, not an exception carved into it. Its restriction is to **byte-wide**
dtypes, and the reason is that raw bytes are only safe when there is nothing to reinterpret:
fp8 is one byte per element, so there is no endianness to get wrong, and aikit cannot offer a
typed accessor without inventing a decode convention that belongs to the caller.

The checkpoints that use these dtypes — DeepSeek V3/V3.2/V4, Qwen3-FP8 — are **block**-quantized:
the scales live in a SEPARATE tensor (`<weight>.weight_scale_inv`) covering 2-D blocks of the
weight (128×128 in both). A lone fp8 tensor therefore does not carry enough information to become
floats, which is the same argument the MXFP4 note above already makes for its `*_blocks` /
`*_scales` pair.

Header parsing already handled these dtypes; only the accessor gate refused them, so a consumer
could see `dtype=F8_E4M3` and a correct shape and then be unable to read the payload.

`BF16`/`F16`/`F32`/`I32` still refuse, and `TestUint8s_fp8Dtypes` pins both halves — a wide dtype
has a typed accessor and would hand out an endianness-dependent view.


### Release gates

`vulncheck`, run on `nobara` (linux) at `304fa83`:

```
STATEMENT: no reachable vulnerabilities in 15/15 modules at 304fa83
```

`perfgate`, `nobara` (Ryzen 7 3700X, linux/amd64), working tree vs v1.30.0 interleaved. **It was
run TWICE and both runs are recorded, because the first one FAILED:**

```
run 1  VERDICT: FAIL — 1 regression(s) vs v1.30.0 across 10 shapes
       GEMV_W8A8_baseline/K2048_N2048  Δ=+3.74%  floor=±2.85%  REGRESSION
run 2  VERDICT: PASS — no regression vs v1.30.0 above each shape's floor
       GEMV_W8A8_baseline/K2048_N2048  Δ=-8.35%  floor=±11.04%  flat (BLIND)
```

**The flagged shape reversed sign by 12 points three minutes later, on the same code and the same
box, with its derived floor moving 4x (±2.85% → ±11.04%).** A regression does not become an 8%
speedup on re-run. `K2048_N2048` is an unstable shape whose characterization pass occasionally
derives a floor far tighter than its own variance and then reads ordinary session drift as a
verdict — the same shape carried ±20.18% in the v1.30.0 run and is BLIND in most.

**The mechanism check is what makes this a defensible read rather than a convenient one.** This
release changes **zero `linalg` files**; the entire executable diff is two strings added to a
switch case and one error message, in `embed`. The benchmark is a `linalg` W8A8 GEMV. There is no
path from one to the other, and "the numbers moved" would have had to overturn that to be real.

**Both runs are published deliberately.** Quoting run 2 alone would be the exact motivated
reasoning perfgate exists to prevent, and the durable finding here is not "it passed" — it is that
**K2048_N2048 must not be trusted to resolve the 5% class on a single sample.**

## [1.30.0] — 2026-08-30

### Added

**A split-half W4A8 nibble layout for amd64, and the AVX2 kernel that reads it.** The canonical
int4 packing interleaves nibbles, so every group costs the AVX2 kernel two `VPUNPCK{L,H}BW` in
its unpack prologue. The split-half layout puts weight `i`'s low nibble and weight `i+16`'s high
nibble in byte `i`, and both shuffles disappear. `dotW4A8SplitHalfAVX2` measures **1.12x, hot
(L1-resident) and cold alike** on Zen 2.

That is the first AVX2 W4A8 win after two accumulator-depth dead ends, and the two results
together identify the bottleneck: fewer instructions per group pays, deeper accumulator chains do
not, so the shuffle port — not the FMA dependency chain — is what this kernel is short of.

New API, all Experimental tier:

- `RepackW4A8SplitHalf(packed, rows, cols, group)` — portable, allocating repack.
- `(*WeightMat).RepackInt4SplitHalf() bool` — **opt-in**, never probed for or applied
  implicitly. Gated on amd64 + AVX2 + `group == 32` + `cols % 32 == 0` + **no AVX-512 VNNI**;
  returns false otherwise and is a no-op on every other architecture, so it is safe to call
  unconditionally.

  **The VNNI exclusion is the interesting half of that gate.** The canonical W4A8 dot prefers
  the AVX-512 VNNI tier added in v1.29.0 and only falls back to AVX2; the split-half kernel
  exists at AVX2 only. So on a VNNI host, opting in would swap a VNNI kernel for an AVX2 one —
  a downgrade, on precisely the newest hardware — and the two tiers accumulate differently, so
  the result would also stop matching canonical. CI found this: the equivalence test passed on
  a Zen 2 box at every shape and failed on a VNNI runner at the largest one (rel 1.09e-4).
  Declining there makes the repack a no-op rather than a silent pessimization. A split-half
  VNNI kernel would lift the restriction and is the obvious follow-up.
- `(*WeightMat).SplitHalfBytes() int` — the layout's size, which is exactly the extra resident
  memory it cost.

**Scales are not repacked.** Split-half permutes nibbles *within* a group and never reorders
groups, so one scale array serves both layouts — unlike the arm64 row4 layout, which interleaves
across rows and therefore needs its own.

**The canonical bytes are never written through.** The repack allocates; `q4`/`q4s` stay
authoritative and are never dropped, so `M > 1` and every non-AVX2 path keep working unchanged.
This is load-bearing rather than tidy: a consumer may have mmap-aliased those bytes zero-copy
from a file (goinfer's `.giw` kind=3 does), and rewriting them in place would silently misdecode
every existing bundle — no error, wrong numbers. `TestWeightMatSplitHalf_canonicalUntouched`
pins both halves of that: bytes unchanged, and the new layout a distinct allocation.

**What it is worth end to end, since 1.12x on a kernel is not a claim about a system.** goinfer
wired the repack into its int4 load path and ran an interleaved, same-binary A/B: **+2.10%
decode tok/s** on Qwen2.5-Coder-1.5B (Ryzen 7 3700X) — real, with the two arms' sample ranges not
overlapping, but bought with a second copy of every eligible tensor's nibbles (**+80%** on int4
weight bytes). goinfer parked it default-off against its own pre-registered bar. Recorded here
because the same trade faces anyone calling `RepackInt4SplitHalf`, and the honest headline is
"1.12x on the kernel, ~2% on the token, at ~80% more weight memory".

**`gpu`: `UploadBatch`** (ships as `gpu/v0.32.0`) — N host-to-device copies behind ONE
synchronize, instead of one synchronize each. In goinfer's CUDA expert cache this cut syncs by
**10.3x** and moved decode **+9.3%** (2.98 ms/token).

**`gpu`: `CopyDevice` / `CopyDeviceBatch`** (ships as `gpu/v0.31.0`) — device-to-device copy on
both backends.

### Changed

**`gpu` reports bandwidth in both conventions** (ships as `gpu/v0.31.0`). The previous figure
counted each copied byte twice (read + write); the doubling was a reporting choice, not
something the hardware did. Both conventions are now stated so a number cannot be read as the
wrong one.

### Fixed

**`linalg`: `q4SplitHalf` was flagged unused (U1000) off amd64.** The field is declared portably
but written and read only from amd64-tagged files, so `staticcheck` reported it as dead on every
other target — invisible to CI, which analyses linux/amd64, and reddening on arm64 dev machines
instead. `SplitHalfBytes` gives it a portable reader that is worth having anyway.

- **`gpu` (Metal): the OS thread is now pinned for an autorelease pool's whole lifetime**
  (ships as `gpu/v0.30.1`). An `NSAutoreleasePool` is per-OS-thread, and Go may migrate a
  goroutine between any two calls — draining a pool on a thread other than the one that pushed
  it is undefined behaviour. It presented as an intermittent `SIGSEGV` (`fault 0x10`) inside
  `objc_msgSend`, at a crash site that MOVED between runs, which is what made it look like
  flakiness rather than a defect. `Run1D`/`Run2D`/`Run1DBatch` create and drain within one call
  and pin with a `defer`; `Queue.Begin()`/`Encoder.End()` span two calls, so the pin is held on
  the `Encoder` and released only after the drain. `BeginNP` is unchanged — it owns no pool, and
  its caller owns both the pool and the pin. `runtime.LockOSThread` nests, so a consumer that
  already pins (goinfer's resident `Forward`/`BuildResident` do) is unaffected.

  Measured on an M1 Pro against goinfer's Metal suite: **20/20 runs clean with the pin; 2 of 8
  crashed with it reverted**, at two different call sites — the site migration reproducing under
  the mutation is what identifies it as the same defect rather than a coincidence.

  **Contract this makes explicit:** an `Encoder` from `Begin()` must reach `End()` on the same
  goroutine and must not be handed to another. That was always true of the pool semantics; it is
  now enforced rather than assumed.

### Release gates

`perfgate`, `nobara` (Ryzen 7 3700X, linux/amd64), working tree vs v1.29.0 interleaved:

```
VERDICT: PASS — no regression vs v1.29.0 above each shape's floor — 5/10 shapes resolve the 5.0% class
sensitivity: 5/10 shapes have a floor ≤ 5.0% (the class this gate targets)
```

All ten shapes measured flat. Read the green as "no regression above each shape's floor": five
shapes carry floors of ±7.9–26.3% and are BLIND to the 5% class, so there the green is only
evidence against a larger regression.

**The perf gate cannot see this release's headline change, and this time not because of the
machine.** It ran on the right architecture — amd64, where the split-half kernel lives — but its
instrument is `BenchmarkW8A8SpanShapes` / `BenchmarkGEMV_W8A8_baseline`, which are **W8A8**. The
split-half work is **W4A8**, and nothing in the gate dispatches to it. A flat result is therefore
the *expected* one and says nothing about the new kernel either way.

That is also the correct outcome for a different reason: `RepackInt4SplitHalf` is opt-in and no
aikit code path calls it, so the default behaviour of every existing consumer is unchanged by
this release. The evidence for the new kernel is its own A/B (1.12x kernel; +2.10% end-to-end in
goinfer), not this gate.

`vulncheck`, run on `nobara` (linux) at `f459341`:

```
STATEMENT: no reachable vulnerabilities in 15/15 modules at f459341
```

(The same command on `darwin/arm64` reports `INCOMPLETE — 11 clean, 0 vulnerable, 4 unscanned`:
the four cuda backends have every file excluded by build tags on macOS, so govulncheck matches no
packages there. The linux run above is the one that scans all fifteen.)

### Measured and rejected

**`dotW4A8Fold2AccAVX2` — two accumulator chains, ~0.5% SLOWER.** Kept in the tree rather than
deleted, per this project's convention that a documented negative is worth something and a
silently reverted one is not. It also duplicated an earlier `dotW4A8Fold4AVX2` result recorded in
goinfer's dead-ends notes; the earlier explanation was the sharper one, and both now agree that
the serial fold is not this kernel's ceiling.

## [1.29.0] — 2026-08-26

### Added

**An AVX-512 VNNI tier for the int8 kernels.** `VPDPBUSD` does in one instruction what the AVX2
path spends a `VPMADDUBSW`+`VPMADDWD` pair on, and for W4A8 it also removes the separate convert
step in the per-group scale fold. `docs/internal/cpu-acceleration.md` item 4 had named this the most
promising amd64 lever since the ops-per-byte campaign, blocked the whole time on hardware. Both
dispatchers gate on CPUID+XGETBV like the existing AVX2 tier, so a CPU without the extension never
reaches the assembly.

- `dotI8` now peels three tiers — VNNI over the 64-multiple, AVX2 over the 16-multiple of what
  remains, scalar for the rest. Integer arithmetic throughout, no reassociation, so the result is
  **bit-identical on every input** to what shipped before regardless of which tiers a given length
  exercises.
- `dotW4A8` prefers a VNNI+VL kernel over the AVX2 one at `group == 32`.

Measured on a VNNI-capable Xeon at the FFN gate/up/down shape (K=5120, group=32, N=17408 — 55.7 MB,
past L3), four runs: **hot 1.226–1.280× over AVX2**, cold **1.109–1.270×**. The hot ratio is settled
(±2.2%); the cold one is real (above 1.0 in 4/4) but not settled (±6.8%) on a shared virtualized
host. Treat "~1.25× hot and measurably faster cold" as supported and any specific cold multiplier as
not. Full numbers, method and caveats: `docs/internal/cpu-acceleration.md` item 4.

**Row4 cold-touch remedies — exported, deliberately NOT wired.** `MatmulBTW4A8Row4PrefetchInto`
(software `PRFM` prefetch, parametrized distance) and `MatmulBTW4A8Row4DesharedInto` with its
`RepackW4A8Row4Deshared` / `RepackW4A8Row4DesharedScales` layout helpers (the four rows' bytes in
separate allocations instead of interleaved). Both are proven bit-identical to the production row4
kernel across five shapes and warm-intact (every prefetch distance exactly 1.000×, de-sharing
0.993×). They are **not** in any dispatch path: the cold penalty they were built to fix did not
reproduce under a corrected same-day re-measurement, so they are recorded rather than adopted, and
a caller must reach for them explicitly. See goinfer's `docs/task-zeno-compare.md`.

### Release gates

`perfgate`, `apple-m1pro`, working tree vs v1.28.0 interleaved:

```
VERDICT: PASS — no regression vs v1.28.0 above each shape's floor — 5/10 shapes resolve the 5.0% class
sensitivity: 5/10 shapes have a floor ≤ 5.0% (the class this gate targets)
```

All ten shapes measured flat (Δ between −2.47% and +0.46%, every branch `flat`). Read the green as
"no regression above each shape's floor": five shapes carry floors of ±14–21% and are BLIND to the
5% class, so there they are only evidence against a larger regression.

`vulncheck`, from CI's linux runner at `00055a9`:

```
STATEMENT: no reachable vulnerabilities in 15/15 modules at 00055a9
```

(The same command on `darwin/arm64` reports `INCOMPLETE — 11 clean, 0 vulnerable, 4 unscanned`:
the four cuda backends have every file excluded by build tags on macOS, so govulncheck matches no
packages there. The linux run above is the one that scans all fifteen.)

**The perf gate cannot see this release's headline change.** It runs on `apple-m1pro`, and the VNNI
tier is amd64-only — on arm64 those files are not even compiled in. The evidence for that change is
the correctness suite on VNNI silicon plus the numbers above, not this gate. `releasegate` v1.29.0:
PASS, 4/4, including `apidiff` Hard tier v1.28.0 → current (additive only).

### Notes for downstream integrators

**`dotW4A8` is no longer single-valued on amd64.** The VNNI kernel matches the scalar oracle to 1e-5
relative — the same bar `dotW4A8FoldAVX2` already meets — but it is *not* bit-identical to the AVX2
kernel, so an amd64 host with VNNI+VL and one without now produce W4A8 results that differ within
that tolerance. This does not reach aikit's own golden fixtures (W4A8 has no caller in `encoder/`,
`embed/` or `vision/`; it is linalg-internal), but a consumer comparing decode output **across
machines** should expect the difference. `dotI8` carries no such caveat — it is exact on every tier.

## [1.28.0] — 2026-08-24

### Added

`SpanCache.AdvisedBytes() int64`: cumulative bytes passed to the `WILLNEED` residency hint over
every miss across all `Touch` calls — what the cache asked the OS to fetch, independent of
whatever else the machine's disk is doing. The forcing function: goinfer's `.giw` kind-4 paged
decode measured a real ~25-30% throughput regression (`docs/task-zeno-compare.md`'s "At-scale
acceptance run"), root-caused to a member registering redundant spans (a tensor's unused
canonical copy alongside its row4 copy) under one key — `Touch`'s `WILLNEED` loop faults in
EVERY span under a key, so the redundant one doubled real disk I/O per miss though the kernel
never read it. `AdvisedBytes` makes that class of bug directly, durably assertable inside a
benchmark or test (bytes-advised-per-token should track the expected working set) — proof
against a bug like this one that an external tool (`iostat`, `/proc/diskstats`) can't give,
since those count physical reads shared with every other process on the box, not what THIS
cache specifically requested. Purely additive; `Stats()`'s existing (hits, misses, evictions)
is unchanged.

## [1.27.0] — 2026-08-24

### Added

`WeightMat` support for constructing an int4 tensor with the split-half + 4-row-interleaved
layout (v1.26.0's `dotW4A8SplitHalf4Row`) from bytes computed elsewhere, instead of only via
`RepackInt4Row4`'s in-RAM derivation — the forcing function is goinfer's `.giw` on-disk kind 4
(`docs/task-w4a8-neon-bandwidth.md`'s "Format follow-on"), which bakes the row4 layout onto disk
at prequant time so the paged-MoE path gets the faster kernel without an in-RAM repack (that
repack doubles resident bytes for every repacked tensor — measured, `+100%`). Additive only.

- **`WrapInt4Row4`** — `WrapInt4` plus already-repacked `q4Row4`/`q4Row4Scales` bytes, aliased
  without copying (a loader mmap-aliasing them back from a serialized bundle, or any caller that
  computed them itself rather than wanting `RepackInt4Row4` to derive them at call time). Gated on
  the same `hasDotProd` check `RepackInt4Row4` already applies before populating `q4Row4` — a real
  gap this closes: before this function existed, `RepackInt4Row4` was the ONLY way `q4Row4` got
  set, so the CPU-feature gate lived entirely there. `WrapInt4Row4` is a second way in, populating
  the field from bytes that may have been computed on a DIFFERENT machine than the one loading
  them — without its own gate, a non-DotProd arm64 core could dispatch to a kernel it cannot
  safely run. Silently keeps canonical-only (same as passing nil) when the gate fails, rather than
  erroring — a build/core that can't use row4 gets exactly what it would have gotten from a plain
  `WrapInt4` call.
- **`MappedSpanRow4`** — `MappedSpan`'s counterpart for the row4 layout: the page-aligned interior
  of `q4Row4` if it lies inside `[base, end)`, as a SEPARATE span from `MappedSpan`'s canonical
  one (a WeightMat carrying both layouts has two independently-mmap'd byte ranges, not one range
  twice). Lets a pager managing both layouts' residency (goinfer's `expertPager`/`layerPager`)
  register and account for both.

## [1.26.0] — 2026-08-24

### Added

A faster W4A8 CPU decode GEMV for arm64, plus the `WeightMat` plumbing to opt individual tensors
into it. Additive only — no existing signature changes.

- **`dotW4A8SplitHalf4Row`** (unexported kernel) — the winner of a harness campaign that started
  from a wrong premise and found a better one along the way. The original hypothesis (an
  issue-width probe reading `dotW4A8FoldSDOT` as issue-limited) did not reproduce under repeated
  measurement on a settled box; the real bottleneck was a serial `VFMLA` accumulator chain, the
  same failure mode the attention kernels above already fixed. Two independent accumulator
  chains alone measured 1.4x; combined with a split-half nibble repack (removes two
  `VZIP1`/`VZIP2` unpack instructions per group — invisible on its own until the accumulator
  stall was gone, then it compounded) and a 4-row interleave (one activation load shared across
  4 real output rows), the combined kernel measured 1.4-1.8x at the isolated-kernel level and
  1.31-1.35x in real decode (~75-81% real/isolated efficiency). Bit-identical to the existing
  per-row kernel by construction — proven exact (`==`), not tolerance, across hundreds of random
  comparisons.
- **`RepackW4A8Row4` / `RepackW4A8Row4Scales`** — one-time repack from the canonical group-int4
  layout into the split-half + 4-row-interleaved layout the kernel above consumes. The canonical
  packer, the `.giw` on-disk format, the scalar oracle, and amd64 are all untouched — this is
  purely an additional in-RAM layout a caller opts a tensor into.
- **`WeightMat.RepackInt4Row4`** — populates the optional layout on an int4-resident `WeightMat`
  (a no-op on non-arm64 builds and any shape the repack rejects: rows not a multiple of 4, cols
  not a multiple of the int4 group size). Explicit and caller-driven only, never probed for
  automatically inside a matmul.
- **`WeightMat.MatmulBTW4A8Into`** — dispatches to the row4 kernel when `RepackInt4Row4` has
  populated it and M=1, falls back to the existing per-row kernel otherwise. Chosen over a
  GPU-backend-style separate resident type: `WeightMat` is already the single choke point a
  caller's dispatch funnels through, and attaching state to the struct that owns its lifetime
  avoids external bookkeeping against a `WeightMat` with a paged/mmap-transient lifecycle — a
  `WeightMat` nobody repacked (e.g. one built over a read-only mmap span for on-demand paging)
  always takes the fallback branch automatically, no caller-side special case required.

### Notes for downstream integrators

Both new levers are measured, real, and reproduced — but the resident-memory cost of the repack
is real too: every repacked tensor keeps its canonical bytes AND gains a same-size second copy
(measured +100% on a real model's int4 weight set). `RepackInt4Row4` never drops the canonical
copy itself; a caller deciding whether the doubled memory is worth the speedup for a given
deployment is expected to weigh that per tensor, per model.

## [1.25.0] — 2026-08-23

### Added

Two new f64 attention GEMV kernels in `linalg`, built for CPU decode attention at M=1. Both are
additive; no existing APIs changed.

- **`MatmulQKAcc64`** — computes per-key query·key dot products as 8 concurrent f64 accumulator
  chains instead of one serialized fold at a time. Each output's own reduction order is
  unchanged — the interleave runs independent outputs side by side, never splitting or
  reassociating a single sum — so results are bit-identical to the existing
  `MatmulBTAcc64Strided` path. The 8-wide shape was chosen over 4-wide by measurement (4.4x vs
  3.0x). Measured 4.41x at the qwen2.5-1.5b decode attention shape on an M1 Pro, and
  depth-independent, consistent with what it is: an FMA-latency fix, not a memory fix.
- **`MatmulAVAcc64`** — computes scores·V with keys-outer/dims-inner accumulation, streaming
  each V row contiguously instead of reading one strided element per multiply. Per-dim sums
  still see keys in the same ascending order, so this is also bit-identical by construction.
  Measured 1.81x at depth 130 growing to 2.39x at depth 8192 on the same shape and box — the
  depth dependence is the expected signature of a memory-order fix, since the strided reads it
  replaces get more expensive as the KV history outgrows cache.

Correctness for both is held to exact equality, not tolerance: the new kernels' outputs are
compared `==` against the existing path across random and stress shapes including every
key-count residue of the interleave width, and the suite runs clean under `-race`.

Context, for the curious: these came out of a measured finding in goinfer that CPU decode
attention was running at serial f64 chain speed (~1.0 ns/MAC) while carrying a deliberate
exactness guarantee (decode bit-identical to batched prefill/verify, which speculative decoding
depends on). These kernels keep that guarantee and remove the serialization. In goinfer,
together with caller-side threading across attention heads, decode attention improved 3.86x at
depth 128 and 10.2x at depth 8192, with zero logit difference across the bit-identity gates.

## [1.24.0] — 2026-08-22

### Added

- **`ann.FlatI8.EnableGPUShardSplit`** — splits the corpus into a
  device-resident shard and a CPU-scored shard, scoring both concurrently
  per `QueryBatch` call and merging the per-query top-k results. Measured
  on real hardware (`apple-m1pro` Metal, `nvidia-rtx2070s` CUDA): a real
  but bounded, share/box/scale-dependent win — up to 1.46x over GPU-only
  on Metal, up to 1.17x on CUDA at N=10k, but a net loss on CUDA at N=100k
  where the GPU is already 15-40x faster than CPU alone and any CPU share
  becomes the bottleneck. `gpuShare` is an explicit, measured-not-computed
  tuning knob — see the doc comment and `docs/internal/cpu-acceleration.md`
  for per-box guidance. `Query`/`QueryFilter` are unaffected; this is
  `QueryBatch`-only.
- **The Go 1.27 `simd` elementwise family (v1.23.0's item 7) extended to
  four more kernels — Experimental tier, `GOEXPERIMENT=simd`.**
  `SiLUInto` (~2.9x both boxes), `GELUInto`/`ErfF32` (~5.2x `apple-m1pro`,
  ~4.25x `nvidia-rtx2070s`), `TanhInto`/`GELUTanhInto` (~5.1-5.6x /
  ~3.6-4.28x), and a fused `SoftmaxRowScaledInto` that eliminates the
  attention scale pass's separate O(L²) sweep under the experiment
  (~15-19% real win on both boxes, provably
  bit-identical to the two-pass sequence it replaces, not merely close).
  Also fixed: `encoder`'s own hot paths (`swigluMLP`, `bert.go`'s
  `gelu`/`geluTanh`, `gte.go`'s GeGLU) were calling the scalar per-element
  activation functions directly rather than these new batched kernels —
  only `vision`'s towers were exercising the vectorized path before this
  release. All four kernels and the fix are off by default: the
  non-experimental build is verified byte-for-byte unchanged (full
  `linalg`/`encoder`/`vision` suites, golden and cosine-parity tests,
  `-race`, and the `GODEBUG=simd=0` emulation leg all pass). See
  `docs/internal/cpu-acceleration.md` items 10-14 for the full numbers and
  per-kernel accuracy contracts.

### Changed

- **`gpu` package's type-suffixed `Buffer` API collapsed into a generic
  twin** on the Metal side (`NewBufferOf[T]`/`NewBufferLenOf[T]`/
  `Upload[T]`/`Download[T]`), mirroring the CUDA-side collapse already
  shipped in `gpu/v0.29.0`. Ships as `gpu/v0.30.0` alongside this release
  — see that tag's own notes; nothing in the root module's public API
  changed.

## [1.23.0] — 2026-08-20

### Added

- **Go 1.27's `simd` package landed for three CPU kernels — Experimental tier,
  gated behind `GOEXPERIMENT=simd`.** `linalg.SoftmaxRowInto` (~2.5-3.2x on both
  boxes, up to ~8.5x on the exp kernel alone with AVX2's 256-bit width), RoPE
  rotation (`encoder`/`vision`, ~1.4-1.9x), and SPLADE's `log1p` pooling
  (`encoder`, a new kernel ported from Cephes' verified single-precision
  `logf`, ~1.37-1.44x — no existing aikit kernel to vectorize, since Go 1.27's
  `simd` package ships zero transcendentals). **Every number here compares
  scalar Go arithmetic against vector CPU instructions (NEON on arm64, AVX2 on
  amd64) — nothing to do with this repo's GPU (CUDA/Metal) backends**, a
  separate code path measured in `gpu`'s own docs; stated explicitly since it
  was a point of confusion internally before this shipped. Off by default: the
  non-experimental build is byte-for-byte what shipped before this landed
  (verified — full `linalg`/`encoder`/`vision` suites, golden and
  cosine-parity tests, and SPLADE's real Python-parity comparison all pass
  unchanged under the experiment). A new CI job on both arm64 and amd64
  GitHub-hosted runners is the only place the experimental path is exercised.
- **`chunk/treesitter`: `WithParseTimeoutMicros`** — a functional option to
  override or disable the `Chunker`'s 1s default wall-clock parse timeout, for
  a deterministic, reproducible build (a borderline file's parse straddling
  the timeout made chunk output load-dependent across machines — ken issue
  #35). `New()` and the zero-value `Chunker` keep the previous default
  unchanged.

### Changed

- **Go toolchain bumped to 1.27.0** (all 16 `go.mod` files). Blocked once
  already: `golangci-lint`'s bundled `staticcheck` panicked on Go 1.27's own
  standard-library composite-literal syntax (`internal/poll`, Linux), not on
  anything in aikit's own code — resolved once `golangci-lint` v2.13.0 shipped
  a `staticcheck` bump; the pin (`tools/gpumod.GolangciLint`, the CI Action)
  moved from v2.11.4.

### Fixed

- **`MatmulBTQ8`'s doc comment** described the pre-P2 scalar int8→f32 widen;
  the widen has been SIMD (`dotI8AVX2`-based) since `2f0c65f`, months before
  this fix. A cross-repo audit re-filed the already-fixed widen as an open
  finding because the wrapper's comment didn't match `q8Span`'s own (correct)
  comment three lines below it.

## [1.22.0] — 2026-08-18

### Added

- **`bm25.MarshalTokens` / `bm25.UnmarshalTokens`** — a cache for the tokenized corpus
  (`Build`'s own input), not a serialized `Index`. Re-measured the cold-start cost behind
  the deferred N4 persistence question before building anything: it's almost exactly 50/50
  between tokenization (8.83ms) and `Build` itself (8.78ms), not tokenization-dominant as
  first assumed. This ships the smaller, reversible half — caching tokenization costs
  nothing on `Index`'s own compatibility surface, since `Build` still runs on load at its
  own already-cheap cost. Versioned, magic-tagged, bounds-checked reader; `Index` itself
  gains no format from this.

### Changed

- **Tier re-curation: `linalg` root (`Dot`/`MatmulBT`/int8-int4 quant kernels),
  `encoder.Backend`/`RegisterBackend`/`NewBackend`, `vision` (whole package), and `mmap`
  (whole package) graduate from Experimental to the Hard tier.** First real audit against
  `docs/internal/roadmap.md` §2.7's graduation trigger — checked goinfer's actual source,
  not just its `go.mod` — found organic, real production usage for all four (36 non-test
  files for `linalg`; goinfer's WebGPU backend + serving path for `encoder.Backend`;
  goinfer's serving path + GPU registration + demo agent for `vision`; 4 non-test files for
  `mmap`), each surviving well past the two-quiet-minors bar. Everything else in the
  Experimental list stays there — age alone was never the bar, and no other surface had
  organic external-production evidence in either goinfer or ken.
- **README's Versioning section rewritten out of pre-1.0 future tense.** aikit has been past
  1.0 for many minors; the section still spoke as if 1.0 were a future milestone. Also
  corrected rather than hid a related gap: the blob-format "tightens at 1.0" language
  implied that tightening was still pending arrival — 1.0 already passed without it landing,
  and the section now says so.

## [1.21.0] — 2026-08-18

### Added

- **`Tensor.Uint8s()` — raw byte access for the byte-wide safetensors dtypes** (`U8`, `I8`,
  `BOOL`). Block-quantized formats increasingly ship their payload as U8 tensors only the
  consumer can interpret: MXFP4 in safetensors stores packed 4-bit codes in a `*_blocks` U8
  tensor and their E8M0 exponents in a separate `*_scales` U8 tensor. aikit has no meaningful
  decode to apply — the layout belongs to the caller — so the honest accessor is the bytes,
  not a typed one that would have to guess. Until now these tensors were simply unreadable:
  every typed accessor rejects them and `raw` is unexported, which blocked goinfer's gpt-oss
  safetensors loader outright.

  Restricted to byte-wide dtypes on purpose. Returning raw bytes for an `F32` would hand out a
  little-endian-dependent view and invite callers to reimplement `reinterpretLE` badly; those
  dtypes already have typed accessors.

  **`dtypeSize` deliberately still reports `U8`/`I8`/`BOOL` as unknown.** Sizing them would
  extend the parser's shape×dtype byte-range check to files that load fine today, turning a
  previously-ignored header inconsistency in an *unread* tensor into a hard load failure for
  existing callers. `Uint8s` performs that same check itself, at read time, on the one tensor
  actually being read — a validation's blast radius belongs where the data is consumed, not
  where an unrelated file is opened.

## [1.19.0] — 2026-08-15

### Added

- **`ann.LoadHNSWMmap`** — zero-copy mmap loader for HNSW, mirroring `LoadFlatI8Mmap`'s int8
  aliasing on the higher-recall index. Bumps the blob format v3→v4: an 8-byte-aligned header
  (unblocks the float32 vector-block alias; int8-mode codes needed no alignment fix) plus a
  reserved flags word `Load` didn't have, the anti-churn mechanism for a future additive change
  without another version bump. `Load`/`LoadHNSWMmap` now share one parser
  (`loadHNSW(data, alias bool)`) so the copying and aliasing paths can't drift apart.
- **`ann.FlatBinaryI8`** — binary prefilter + `FlatI8`'s int8 rerank, compounding the memory win
  over `FlatBinary`'s exact float32 rerank (dim/8 + dim bytes vs. dim/8 + 4·dim) on top of the
  already-measured 13–26× throughput gain. `FlatBinary`'s own exact-rerank default and contract
  are unchanged — this is a new sibling type, not a behavior change to an existing one.
- **`late` package** — ColBERT-style late-interaction (MaxSim) reranking over pre-computed
  per-token vectors (`encoder.Model.EncodeTokens`, built for exactly this), explicitly scoped as
  a shortlist reranker rather than a corpus-scale index (a token matrix is ~L× a pooled vector's
  footprint).
- **`hybrid` package** — a thin, opt-in wrapper around "retrieve dense + lexical, then
  `fuse.RRF`", the four lines every hybrid-search example already hand-wires identically. Composes
  already-built indices; doesn't build them, embed, tokenize, chunk, or rerank.
- **`bm25.Index.TopKBatch` / `sparse.Index.QueryBatch`** — batch query APIs mirroring
  `ann.FlatI8.QueryBatch`'s work-stealing goroutine dispatch, for bulk workloads (an eval harness
  scoring many queries, say) — previously the visible asymmetry against the dense side.
- **`examples/splade`, `examples/vision`, `examples/colbert`, `examples/gpu-ann`** — four new
  runnable examples, each making a previously undemonstrated real capability visible: learned-sparse
  retrieval standalone; the vision package's image-as-document indexing and image→image similarity;
  ColBERT/MaxSim reranking against the same fused shortlist `examples/rag` uses; and the native-GPU
  `ann.FlatI8.EnableGPU` seam, verified end-to-end on both Metal (M1 Pro, near CPU parity at 1M
  vectors) and CUDA (RTX 2070 SUPER, ~69× over CPU at 1M vectors) — every run on both backends
  bit-identical to the CPU result.

### Fixed

- **`encoder`'s dense-GELU dispatch conflated `gelu_pytorch_tanh`/`gelu_new`/`gelu_fast` with the
  exact erf `gelu`.** All four activation names routed the dense (non-gated) MLP through the same
  erf implementation; the three tanh-family names are a different function
  (`linalg.GELUTanhF32`, already used correctly elsewhere for SigLIP's MLP). No currently-shipping
  checkpoint was affected — latent for any future Gemma-family-style addition declaring one of the
  three tanh names.

### Changed

- **Python is now boundaried by directory, and the residue is gone.** The four non-oracle scripts
  in `scripts/` are resolved: `m0_ceiling.py` struck; `shard_checkpoint.py` moved to goinfer (it
  generates a sharded fixture goinfer's `decoder/sharded_test.go` consumes and references goinfer's
  decoder — a leftover from the split). The reference generators now live in `scripts/oracle/` (the
  34 PyTorch/HF golden pins + `gen_iq_grids.py`, which dumps llama.cpp's ggml tables), and the one
  cost-basis exception in `scripts/fixtures/` (`prep_beir.py`, an HF-parquet fetch for a manual
  benchmark). `tools/scriptsguard` (CI) enforces both campaigns' claims by directory: every `.py`
  under `scripts/` must be in `oracle/` or `fixtures/`, and NO `.sh` may live under `scripts/`
  (deciding-shell gates are Go commands under `tools/`) — so neither residue can silently regress,
  derived by directory rather than a hand-kept list. Skip-message and doc references to the pin
  scripts moved with them. (The gpu/ CUDA build toolchain — `build_ptx.sh` + `nvrtc_compile.py` —
  is separate build infrastructure, out of this scope.)

## [1.18.0] — 2026-08-14

### Added

- **`MatmulBTAcc64Strided` — a strided-second-operand attention matmul, so a decoder can attend
  a KV-cache head-block in place instead of re-copying/re-transposing it into scratch every
  token.** The KV cache is `[nKeys, nKV·hd]` interleaved; QKᵀ wants a head's `[nKeys,hd]`
  (strided rows) and scores·V wants its transpose `[hd,nKeys]` (strided elements), and both are
  expressible as `b[j][k] = bMat[bOff + j·bRowStride + k·bElemStride]`. Only b's addressing
  changes — the sequential float64 reduction order is identical to `MatmulBTAcc64` — so it is
  **byte-for-byte identical** to running `MatmulBTAcc64` on a packed/transposed copy
  (`TestMatmulBTAcc64Strided_bitIdenticalToPacked`). Additive; the existing primitives are
  unchanged. Motivated by goinfer's decode, where that gather is a measured ~10% of per-token
  time at 1.5B/4k context, rising with model size and context (P1).

### Changed

- **The q8 weight-only matmul (`MatmulBTQ8`) is ~2× faster at M=1 on a large-vocabulary LM
  head.** `q8Span` widened each int8 weight row to f32 with a scalar loop the compiler does not
  vectorize; at M=1 that widen was ~68% of the function (measured, K∈{2048,3584}, N∈{152064,
  262144}). It now uses the existing SIMD widen (`dequantRowInt8`, AVX2/NEON) with scale 1.0.
  **Bit-identical** — the widen is a per-element convert×scale with no reassociation and ×1.0 is
  exact, so the output is byte-for-byte unchanged (asserted by `TestQ8Span_bitIdenticalToScalarWiden`
  across serial, parallel, SIMD-tail, and prefill paths). Measured working-tree vs pre-change
  with `tools/perfgate`: Δ ≈ −50% on all three shapes, above each shape's noise floor.

- **Require Go 1.26.6** (was 1.26.5). Picks up the go1.26.6 standard-library security
  fixes for `net/http`, `crypto/tls`, `net/url`, and `encoding/asn1`
  (GO-2026-5026/6090/6218/5972) — govulncheck flagged them reachable in the internal
  `benchmarks` harness; every shipped module scanned clean either way.

### Fixed

- **The eight `gpu/*` backend modules pinned a stale root version and now track the tag.**
  Each `require`d `aikit v1.17.0` while the root series had advanced to `v1.17.1` — the
  two-series scheme restated by hand in eight go.mod files, and one release forgot. The new
  `tools/gpupins` gate (CI, every push) asserts each backend names the latest `vX.Y.Z` and
  `gpu/vX.Y.Z` tag; `go run -C tools ./gpupins --fix` rewrites them from git, replacing the
  hand-edit in RELEASING.md step 3.

## [1.17.1] — 2026-08-12

### Fixed

- **The W8A8 matmul is 3–5% slower at streaming shapes in 1.17.0. Reverted.** The
  eight-column int8 kernel shipped in 1.17.0 widens the activation row once per group of
  eight output columns instead of once per column — a real arithmetic saving, measured at
  **one** shape (K=768 with a small N) and reported as +17.4%.

  What that shape hid is that the two forms walk memory differently. The previous span
  reads B strictly linearly; the eight-column form advances eight streams `K` bytes apart,
  which the hardware prefetcher handles far worse once B stops fitting in cache. Measured
  on a Ryzen 7 3700X (32 MB L3), parallel dispatch, M=1:

  | shape | B size | eight-column vs linear |
  |---|--:|--:|
  | K=768 N=8192 | 6 MB | **−31%** (the win that was shipped) |
  | K=768 N=200000 | 154 MB | +3.5% — `FlatI8`'s CPU scan over a real corpus |
  | K=3584 N=18944 | 68 MB | +5% — a 7B model's FFN |

  Both production callers are in the streaming regime. goinfer measured **~3% end to end
  on decode** and reported it against v1.17.0; this restores parity with v1.16.0 at both
  shapes, verified interleaved.

  Not replaced with a "use the wide kernel when B fits in L3" threshold, deliberately —
  that is the shape of three constants this campaign already found stale. `dotI8Cols8`
  stays in the tree with its tests and the evidence for why it is not wired, and
  `BenchmarkW8A8SpanShapes` now covers both regimes so the next attempt cannot be
  evaluated at one shape again.

## [1.17.0] — 2026-08-12

Measured on `nvidia-rtx2070s` (Ryzen 7 3700X, RTX 2070 SUPER) and an Apple M1 Pro. Every
ratio is against a ceiling measured on the machine that ran it; method, negatives and the
full decomposition are in
[`docs/internal/roofline-2026-08.md`](docs/internal/roofline-2026-08.md).

### Added

- **`gpu.Device.SMCount()`** (`gpu/v0.28.0`, linux/CUDA) — the device's multiprocessor count, so a
  launch can size its grid against *this* device instead of a constant measured on one. 0
  means "unknown", not "none". `gpu/anncuda` uses it to decide when splitting one query's
  top-k across blocks pays.

- **`gpu.Pipeline.ThreadExecutionWidth()`** (`gpu/v0.28.0`, darwin/Metal) — the pipeline's SIMD
  width, for kernels that put one SIMD group on each unit of work.

- **`bench.MinDuration` / `bench.MinWindow`** (root) — a timed loop that runs at least
  `iters` times *and* for at least 50 ms. See Fixed, below: this exists because the old
  fixed-count loop was shorter than a Go GC cycle.

- **`gpu/roofline.cu` + `TestDeviceCeilings`** (`gpu/v0.28.0`, test-only) — three device ceilings
  (streaming read, int32 multiply-add, `__dp4a`) with a self-check that the probes measure
  what they claim. Every "% of peak" in the docs above is against one of these.

- **`gpu.Device.MaxThreadgroupMemoryLength()`** (`gpu/v0.27.0`, darwin/Metal) — the device's tile-
  memory limit in bytes (~32 KiB on Apple GPUs). A dispatch whose `setThreadgroupMemoryLength`
  exceeds it aborts the command buffer, so a caller sizing threadgroup scratch from model dims can
  check against this and decline instead (goinfer audit M-11).

- **`gpu.Encoder.Err()`** (`gpu/v0.26.1`, darwin/Metal) — `WaitDone`/`End` now latch the
  committed command buffer's `status`/`error` while the cb is still alive (before the pool
  drain frees it), and `Err()` returns it. `waitUntilCompleted` returns cleanly even when a
  kernel aborts, so without this the host would trust stale results after a GPU fault;
  callers now consult `Err()` before reading a buffer's output (goinfer audit C-09).

### Changed

Performance only; every path below is gated on producing byte-identical results to the
CPU, and all of them do.

- **`ann.FlatI8.Query` selects on the device** when the backend offers it and no filter is
  set, so `k` hits cross back instead of `n` scores: **1.19× at N=200k, 1.33× at 500k,
  1.46× at 1M**. The win grows with N because what it removes is O(N). A `keep` filter
  still scores and selects on the host — the device selects before filtering, so it cannot
  be used there.

- **`gpu/anncuda` batch scoring** — the batched path ran a 16×16 tiled GEMM whose cost
  depended on `ceil(M/16)` rather than on M (6.79 ms for every M from 1 to 16 at N=200k).
  Replaced with a lane-group GEMV that stages the query tile in shared memory and strides
  over rows: **`QueryBatch` is 4.5–4.6× end to end at every batch width**.

- **`gpu/anncuda` device top-k** — was k passes over the score row, so its cost scaled with
  k while the scoring in front of it did not. Now one pass with per-thread candidates in
  registers, split across blocks when one per query would leave the device idle:
  **3.2–3.4× at k=10, 12× at batch 1**, and flat in k rather than linear.

- **`gpu/anncuda` single-query GEMV** — lane group per row retuned from 32 threads to 16:
  **19% at K=768, 34% at K=256, 15% at K=255**. 16 is not the fastest on the common
  shapes; it is the only width within 4% of the best on *every* shape measured, including
  `K%4 != 0`.

- **`gpu/annmetal`** — SIMD-group-per-row `gemv_w8a8` (**6.0–9.2×**, 27–42% of that
  device's measured bandwidth ceiling) and a parallel tree-merge `topk_rows` (**2.1×**).

### Fixed

- **`CUDA_ERROR_MISALIGNED_ADDRESS` in `gpu/anncuda`'s top-k** — the `float4` path read
  `scores + m*N` as `float4`, which is only 16-byte aligned when `m*N` is a multiple of 4.
  Any corpus whose size was not a multiple of 4 faulted at `k <= 8`, and a misaligned
  access kills the whole CUDA context rather than the one launch. Now guarded on the
  pointer.

- **Benchmark timings at small batch were wrong by up to 2×** (`bench`, and both crossover
  harnesses). The timed loop ran a fixed 10 iterations and kept the fastest; at small batch
  one iteration is ~0.1 ms, so the whole window fit inside a single Go GC cycle. Taking the
  minimum is the right defence against transient interference and cannot work when the
  interference outlasts the sample. **Published crossover numbers changed materially as a
  result** — single-query Metal at N=100k was recorded at 0.65× and is actually **0.09×** —
  so `docs/BENCH-gpu-results.md` was regenerated on both machines.

## [1.16.0] — 2026-07-31

A single additive `mmap` (Experimental) knob, requested by goinfer's MoE expert
pager. `apidiff` reports **zero incompatible changes** in every Hard-tier package
against 1.15.0; `NewSpanCache`'s existing behaviour is unchanged.

### Added

- **`mmap.NewSpanCacheWithPolicy(budget, policy)` and the `mmap.EvictPolicy` knob**
  (Experimental) — `SpanCache` can now evict by either of two policies, and the right
  one depends on the caller's DEMAND SIGNAL:
  - **`EvictMostRecent`** — scan-resistant, and **the default, so `NewSpanCache` is
    unchanged**. For a cyclic scan (an ANN paged query walking blocks 0,1,2,… every
    pass) plain LRU is pathological — the block evicted to make room is exactly the one
    wanted next round, so the cache hits 0% even at a 63/64 budget — and this pins a
    stable prefix instead (perf-campaign item 9).
  - **`EvictLeastRecent`** — classic LRU tail, frequency-aware. For a skewed-frequency
    signal (a MoE expert pager whose hottest ~10% of experts absorb ~72% of top-k picks)
    the hot set is what must stay resident; evict-most-recent throws it out the instant
    anything else is touched — measured up to **51 pp** of hit rate worse on a real
    35B-A3B trace at interactive budgets — so a frequency-skewed pager must use this.

  Pick the policy from the access pattern, not by default: the wrong one silently
  regresses hit rate (see `EvictPolicy`'s doc comment). Everything else about
  `SpanCache` — the lossless-refault contract, the resident-bytes invariant, `Stats` —
  is unchanged.

## [1.15.0] — 2026-07-31

The performance-campaign release: measured work across `ann`, `bm25`, `embed`,
`encoder`, `fuse` and `chunk`, plus the native-GPU phases. Additive only —
`apidiff` reports **zero incompatible changes** in every Hard-tier package against
1.14.0, and the ~40 new exported symbols are tier-assigned in
[README's stability tiers](README.md#stability-tiers).

> **Note on 1.13.0 and 1.14.0.** Those two tags were cut without CHANGELOG
> sections, so their changes — the native-GPU CUDA device layer and Phase-1b
> GPU-scored ANN, `FlatI8.QueryBatch`, the tiled Metal/CUDA GEMMs and the ANN GPU
> crossover harness — were never filed and are folded into this section rather
> than reconstructed after the fact. Several entries below span all three tags, so
> splitting them retroactively would mean guessing; the compare links for 1.13.0
> and 1.14.0 are listed at the bottom so the actual ranges remain inspectable.

> **Measurement provenance (2026-07 perf campaign).** Unless an entry says
> otherwise, its figures were measured on **`nvidia-rtx2070s`** (Ryzen 7 3700X, Zen 2,
> amd64 — the Phase A/B box); **`apple-m1pro`** (M1 Pro, arm64, 6P+2E, macOS) figures
> are called out where they differ. CPU-time and allocation/heap ratios transfer in
> kind across both (the absolute may move; the ratio holds — measuring-performance
> §1.34). **RSS, cold-start and page-fault numbers are OS-mediated and do NOT transfer
> between operating systems** (measuring-performance §1.35): in particular the
> `LoadWeightsQ8` peak-RSS win below is Linux-only. `ann.*.WriteTo` RSS savings are the
> different kind — they come from never allocating the blob, so they hold on any OS.

### Added

- **`ann.FlatI8.WriteTo(w)`** — streams the index to a writer instead of
  building the whole blob first. **Transient allocation drops 52.0 MB → 4.1 KB**
  on a 200k×256 index, and peak heap halves (100.0 → 50.4 MiB) because
  `MarshalBinary`'s blob is as large as the index it serializes. Implements
  `io.WriterTo`, so `os.File`, `bufio.Writer` and `io.Copy` pick it up
  automatically. Bytes are identical to `MarshalBinary`, gated byte-for-byte.

- **`embed.LoadMmap(dir)`** — loads a Model2Vec model with `model.safetensors`
  memory-mapped instead of read onto the heap. **Peak heap falls 5.8× and
  allocation 4.6×** on a 64 MB checkpoint (cold start 75.8 → 13.0 MiB peak, 82.4
  → 18.1 MB allocated), while **time-to-first-result rises 17%** (`nvidia-rtx2070s`;
  the page-fault cost is OS- and storage-dependent, so this one number does not port —
  measuring-performance §1.35) — mmap defers the page faults to the first `Encode`, and
  faulting 64 MB in costs more than reading it sequentially from a warm page cache.

  A footprint option, not a speed one: take it under a memory cap, leave it for a
  short-lived CLI. Additive, so no existing caller changes behaviour. Vectors are
  bit-identical to `LoadFromFS`, gated over a full corpus.

- **`embed.StaticModel.EncodeBatch(texts, concurrency)`** (perf campaign A1) —
  encodes a corpus concurrently, returning one vector per input in input order.
  `concurrency <= 0` means `runtime.NumCPU()`, matching
  `encoder.Model.EncodeBatch`. **8.21× at NumCPU** on an 8C/16T box (295 → 35.9
  ms over 1,557 chunks), which makes a whole index run **3.53×** faster.

  Until now `Encode` was the entire public encode surface, so every caller —
  including both shipped examples, now updated — wrote a serial loop over a
  corpus. Results are **bit-identical** to that loop: `StaticModel` is immutable
  after load and `Encode` touches no shared mutable state, asserted by exact
  equality over a full 1,557-chunk corpus at eight concurrency settings.

  Measured scaling: 91% of linear at 2 workers, 87% at 4, 78% at 8 physical
  cores, with SMT worth a further 1.31× to 16 threads.

- **`ann.FlatBinary`** (perf campaign item 38) — a two-stage retriever: a 1-bit
  sign-quantized Hamming prefilter over the whole corpus, then an exact float32
  rerank of the survivors. Same `Hit` / `Query(q, k)` shape as `Flat` and
  `FlatI8`. **13–26× faster than the exact scan** (geomean 18.6×; 115.6 ms →
  4.78 ms at dim 768, 1M vectors), with allocations down 55%.

  **This is the package's first approximate index** — `Flat` and `FlatI8` both
  score every vector. Recall@10 is 1.00 at the default `DefaultOverquery` of 16
  on a real Model2Vec corpus, but the true top-k *can* be missed; scores are
  exact. `k <= 0`, `k >= Len`, or `Overquery·k >= Len` are exact, hit for hit.
  Constructors: `NewFlatBinary(vecs)` and `NewFlatBinaryOverquery(vecs, n)`.

- **`encoder.CrossEncoder.ScoreBatch(query, docs, concurrency)`** (perf campaign
  item 28) — scores one query against many documents, returning label 0's logit
  per document in the caller's order. **7.56× over a `Score` loop** at 50
  documents (6.16 s → 814 ms) with allocations down 87k → 12.1k, from
  document-level parallelism plus longest-pair-first scheduling. Scores are
  bit-identical to `Score`.

- **`ann.HNSW.WriteTo(w)`** — streams the index to a writer instead of building
  the whole blob first, mirroring `FlatI8.WriteTo`. On a 50k×256 index
  **transient allocation drops 131.0 MB → 65.5 KB** (1999×) and cold-process peak
  RSS falls **181.3 → 125.5 MiB** — the saving is one whole copy of the index, so
  it grows with the index (~3.1 GB at 1M×768). Implements `io.WriterTo`, so
  `os.File`, `bufio.Writer` and `io.Copy` pick it up automatically. Serialized
  bytes are unchanged and SHA-256-gated against the previous format.

  It is also faster — **1.68×** f32, **8.02×** int8 — but this is a footprint
  change; the allocation figure is the one that transfers across machines.

- **`ann.HNSW.Query` allocates 2 times instead of 19** (13.4 KB → 1.2 KB per
  query), and is **−27.1%** faster on a 5k×64 index. The two search heaps grew by
  `append` from nil on every query; they are now pooled alongside the
  visited-set tracker that was already pooled. Results are unchanged. The time
  win is GC assist rather than the allocator — 12.3 KB of garbage per query at
  ~25k queries/s is ~300 MB/s, charged to the querying goroutine.

- **`ann.Load` allocates 8 times instead of 153,396** on a 50k-doc HNSW index
  (2.164 → 0.004 per doc). The graph and vectors are read into flat arenas and
  sub-sliced rather than allocated per doc and per node. No API change, no format
  change; at 1M docs this is >2.1 M allocations removed from one `Load`.

- **`ann.HNSW.MarshalBinary` allocates 2.25× less** — 131.0 MB → 58.2 MB for a
  58.2 MB blob. Its capacity estimate budgeted 8 bytes of graph header per node
  where a node with L layers needs 4+4L, so `append` doubled mid-write. Output
  bytes are unchanged.

- **`embed.SafetensorsFile.ReleaseTensors(names...)`** — drops the resident pages
  of tensors a loader has finished with, while the file stays open for the rest.
  Advisory and lossless: the mapping is read-only and file-backed, so a released
  page re-faults identical bytes; the cost of releasing too eagerly is a fault,
  never wrong data. A no-op on heap-backed files and off Linux.

### Changed

- **`encoder.LoadWeightsQ8` peaks at 3.0× less memory on Linux** (lens §4.5):
  **727.6 → 242.3 MiB peak RSS** on a 521.6 MiB F32 checkpoint, to produce the
  same 199.5 MiB model, at **+0.10%** load time (min-of-10 — inside the drift
  floor).

  **Scope: Linux-only, confirmed by measurement on both boxes.** The mechanism is
  `madvise(MADV_DONTNEED)`, which macOS does not honour for a read-only file-backed
  mapping — `apple-m1pro` measures **726.2 MiB**, i.e. the unreleased figure, no win at
  all. The pages there are clean and file-backed, so the OS still reclaims them under
  pressure and nothing OOMs; it is the peak-RSS *number* that is a Linux artifact. Do
  not quote 3.00× as a laptop figure. (This is an OS-mediated RSS number — the class
  that does not transfer between operating systems, measuring-performance §1.35 — not a
  CPU or allocation figure.)

  Quantizing reads every f32 weight, so the whole mapping used to stay resident
  until `Close()` at the end. Each tensor's pages are now released as soon as it
  has been quantized, which bounds the resident f32 set at one layer instead of
  twelve. The model bytes are unchanged.

  Peak *heap* was never the problem and does not move: it is 199.5 MiB before and
  after, exactly its steady state. This is a resident-set fix, so it matters under
  a hard memory cap (containers, cgroups) and is invisible to `B/op`.

- **`fuse.RRF` / `fuse.RSF` are 4.8–5.4× faster** (perf campaign A5): 46.4 → 8.7 µs
  at k=50, geomean 5.36× across k=10…1000, allocations 22 → 4. On a hybrid
  retrieval query that is **−20%** end to end, since RRF was 23.3% of it.

  Output is unchanged — same keys, same scores, same order — gated against a
  reference implementation of the previous algorithm on tie-heavy inputs. The
  win is mostly not the map presizing: first-appearance order used to live in a
  second map consulted from the sort comparator, i.e. O(n log n) map lookups, and
  now lives in the result slice's own positions.

  **Scope:** `fuse` is O(shortlist), not O(corpus). This is a
  small-to-medium-corpus finding — at n=1M it is under 0.1% of a query, and
  behind a reranker the whole retrieval stack is a thousandth of one.

- **`bm25.Index.TopK` now uses WAND dynamic pruning** (perf campaign item 39)
  for `k >= 0`: documents whose per-term upper bounds cannot beat the current
  k-th best score are skipped instead of scored. **3.88× on a mixed-selectivity
  query** (one common term with two rare ones, 46.0 → 11.9 µs at 200k
  documents), geomean −27.9%.

  Results are **exact and bit-identical** — same documents, same scores, same
  order — not approximate: pruning changes what is computed, never what is
  selected. It declines, falling back to the exhaustive scan, for `k < 0`, for
  negative `K1`/`B` (which would invalidate the bound), and for queries longer
  than 8 distinct terms, where the pivot loop's per-term cost overtakes the
  scan.

  Two shapes are slightly slower: a query of three equally-common terms (+7.8%,
  nothing to skip, every document is a genuine candidate) and a query of three
  rare terms (+11.2% of a 1 µs query, fixed setup cost).

- **The `regex` chunker is 1.68× faster on Go source, 1.76× on a repository
  index run** (lens scan §3.1 + §3.2). Two changes: `scanDepth`'s per-byte
  closure became a scalar compare and its `hasPrefixAt` calls are gated on a
  first-byte test, and definition/skip/attach patterns are prescreened by the
  literal prefix a match must begin with, skipping the regexp engine for lines
  that cannot match. Chunk output is unchanged; the prescreen is gated over 5.9 M
  (pattern, line) pairs across every rule and every `.go` file in the repository.

- **`bm25.Build` is 1.27–1.30× faster** (lens scan §3.7): one map from term to a
  `[]termEntry` index, where there were three (postings, document frequency, and
  the WAND bound's extrema). `m[k] = append(m[k], v)` is a mapaccess plus a
  mapassign — two hashes of the same key — and the other two maps added three
  more; that was five hashes per (document, term). The per-term extrema are now
  tracked as `Build` goes, which also removes the second pass over every posting
  list that used to compute them. Index contents are unchanged, gated against an
  independent recount of every term's postings, frequency and extrema.

- **`sparse.Index.Query` and `Scores` are 1.40–8.47× faster** — `1.70×` on the
  30-term SPLADE shape. The touched-set ordering was a full `slices.Sort`,
  O(T log T), which on a 30-term query touching ~9,200 documents cost more than
  scoring all of its postings. It is now a merge of the per-term ascending runs,
  O(T log Q). `bm25` made this change in perf campaign item 44 and this package
  was left behind; item 39's benchmarks — the first this package has had — put
  it back in view. Output is unchanged, gated against a dense scan on tied
  scores.

- **`ann.Flat.Query`/`QueryFilter` now shard the scan across cores** (perf
  campaign item 16): 1.73–2.26× depending on index size (`nvidia-rtx2070s`, 8C/16T —
  the speedup scales with core count), and −42% on the filtered path. Results are
  unchanged — still the k highest by score with ties
  broken by ascending index, identical to a serial scan, gated on
  adversarial all-ties inputs and shard-width invariance.

  **Contract note (behavioural, not a signature change):** `QueryFilter`'s `keep`
  must now be a **pure predicate that is safe for concurrent use**. It is called
  from several goroutines, and because each shard keeps its own running top-k
  threshold, the *set* of ids it is asked about and their order differ from a
  serial scan. A read-only live-set lookup — the intended use — is unaffected; a
  closure that counts calls or memoizes into shared state is not. The same
  requirement is now documented on `FlatI8.QueryFilter` and `HNSW.QueryFilter`,
  which still apply `keep` serially, so one filter works with every index.

### Added — native-GPU (Phases 1b–4)

- **Native-Metal SigLIP resident encoder (`gpu` + `gpu/visionmetal`, native-GPU
  Phase 3, Apple).** The Metal mirror of the CUDA ViT: the 8-kernel encoder set in
  MSL (`gpu/metal_vit.go` — quantized/f32 GEMM, LayerNorm, tanh-GELU, bidirectional
  MHA, per-row int8 quant, the two adds) plus `gpu/visionmetal`, a
  `vision.ResidentEncoder` registered via `vision.RegisterResident`
  (`Encoder.EnableResident()` routes Forward to the GPU). Verified on Apple M1 Pro:
  cosine **1.000000000** vs the CPU tower, worst abs Δ 6.71e-07, each kernel also
  gated individually. MSL has no `double`, so the reductions use f32 pairwise
  (tree) sums and the ViT library is compiled fast-math-off (new
  `Device.CompileLibraryPrecise`) for exact divides — enough to match the double
  CUDA result. Adds `Device.NewViT`/`ViT` and `Queue.Run1DTG` to the Metal device
  layer. Device tests are hand-run; the default build stays pure-Go.

- **`aikit/gpu` CUDA API extended to fit a tuned kernel set (`gpu/v0.3.0`).** `v0.2.0`'s
  dispatch was buffers-only on derived 1-D geometry, which the ANN proving kernels need and
  a tuned decode path cannot express. Adds, all in `cuda.go` (`//go:build linux`):
  `gpu.KernelArg` with `Arg(Buffer)` / `ArgValue[T](v)` so scalars pass **by value**
  positionally between buffer args; `gpu.LaunchConfig` with `Grid1D` / `GridOne` helpers for
  hand-picked geometry including **dynamic shared memory**; `Queue.Launch` (enqueue, no sync)
  plus `Queue.Sync` for the launch-many-then-sync-at-a-boundary model; `gpu.HostBuffer[T]`
  with `NewHostBuffer` / `ReadToHost` for **pinned** host readback; and generic buffer verbs
  `NewBufferOf` / `NewBufferLenOf` / `Upload` / `Download` covering every scalar element type.
  Together these let a consumer express a whole decode loop importing **only** `aikit/gpu` —
  no `gocudrv` type appears in any signature. `Run1D` / `Run1DBatch` / `Encoder` are unchanged
  and now built on the same machinery; the darwin surface is untouched.
- **CUDA device layer + GPU-scored ANN on NVIDIA (native-GPU Phase 1b).** The Linux
  mirror of the shipped Metal work, so the substrate is now two-platform. `gpu/cuda.go`
  (`//go:build linux`) presents the same `Device`/`Buffer`/`Queue`/`Pipeline`/`Encoder`
  vocabulary as `gpu/metal.go` over `gocudrv` (cgo-free, dlopen'd libcuda); the new
  `gpu/anncuda` nested module registers the CUDA `ann.Backend` with `gemv_w8a8` /
  `gemm_w8a8` kernels (PTX, built by `gpu/build_ptx.sh` via NVRTC, so the runtime needs
  no CUDA toolkit). Verified on an RTX 2070 SUPER: GPU top-k ≡ CPU `linalg.MatmulBTW8A8`
  top-k for both the single-query GEMV and the batched GEMM, worst score Δ `0.000e+00`
  (the int8 dot is exact integer arithmetic). Device-only and hand-run; `gocudrv` is
  confined to the `gpu` module under `//go:build linux`, so the root module's
  one-dependency-tier invariant and the darwin/default builds are untouched.
- **`ann.FlatI8.QueryBatch(queries, k)` + GPU batched int8 GEMM (native-GPU Phase 2).**
  Scores a whole batch of queries against the corpus in one dispatch when the index
  is GPU-enabled (`ann.I8BatchIndex`, implemented by the Metal backend) — the batched
  GEMM that is the GPU's sweet spot — and degrades to per-query scoring otherwise.
  Verified on M1 Pro: batch GPU top-k ≡ CPU per-query top-k, bit-identical. Also fixes
  an intermittent `NSAutoreleasePool`-drain-on-the-wrong-thread SIGSEGV in the Metal
  backend by pinning the OS thread across each dispatch.

## [1.12.0] — 2026-07-27

The native-GPU + coverage-completion release. On top of 1.11.0's embedder coverage,
this cycle adds a **cgo-free native-GPU compute substrate aikit owns** (Metal today,
the GPU analogue of `linalg`) with GPU-scored ANN as its first consumer; takes the
certified embedder set from **eight to fourteen** — including two that needed new
primitives (a byte-level BPE tokenizer and a GTE RoPE/GeGLU forward); adds **MXFP4**
GGUF dequant (gpt-oss); and folds in a full engineering-audit remediation — 13
correctness fixes (silent-wrong pooling, dropped attention biases, three data races,
determinism, div-by-zero guards) and a broad allocation/kernel performance pass, each
bit-identical or parity-gated.

Minor, not patch: new exported API (`encoder.LoadGTE`/`GTE`, `ann.Backend` + friends,
the `gpu` module, `MatryoshkaFloor` rows), no breaking changes. The behavior changes
either fix latent bugs or **widen** acceptance, so they are additive from a caller's
perspective. The default build stays pure-Go CPU with the same one dependency-tier.

### Added

- **Native-GPU device substrate (`gpu`, new Experimental darwin module) + GPU-scored
  `FlatI8`.** A cgo-free Metal device layer (Device/Buffer/Queue/Pipeline/Encoder +
  runtime MSL compiler, via `ebitengine/purego`, `CGO_ENABLED=0`) that aikit owns —
  the GPU analogue of `linalg`. `gpu/annmetal` registers an `ann.Backend` (the same
  inversion `encoder`/`vision` use, so core `ann` never imports it) that scores an
  int8 `FlatI8` corpus GEMV on the GPU; `FlatI8.EnableGPU()` opts in. Verified on
  Apple M1 Pro: GPU top-k ≡ CPU `linalg.MatmulBTW8A8` top-k, bit-identical (the int8
  dot is exact integer arithmetic). Device tests are hand-run (no GPU CI); the
  default build and the core dependency invariant are unchanged (separate module,
  `purego` never enters the root). Native-GPU Phase 1; CUDA + the tuned-kernel lift
  are Phase 1b.
- **MXFP4 (ggml type 39) GGUF dequant (`embed`).** OCP Microscaling FP4 — the
  32-element block (e8m0 scale + 16 e2m1 nibble-pairs) — unlocking gpt-oss weights.
  Hand-derived against `gguf/quants.py` and pinned (`embed/gguf.go`,
  `embed/gguf_mxfp4_test.go`).

- **GTE encoder + `encoder.LoadGTE` (`encoder`).** A pure-Go forward for the GTE
  architecture (Alibaba gte-multilingual-base): post-norm, RoPE positions, a
  packed qkv projection, and a gated-GELU (GeGLU) MLP. Certifies
  `Snowflake/snowflake-arctic-embed-m-v2.0` at cosine 1.000000 (worst hidden maxΔ
  7.8e-06), Matryoshka to 256. `encoder/gte.go`, `scripts/pin_gte.py`.
- **Byte-level BPE tokenizer (`embed`).** The GPT-2 / RoBERTa tokenizer — byte
  map, the GPT-2 pre-tokenizer (with a hand-rolled replacement for the
  `\s+(?!\S)` lookahead RE2 can't express), ranked-merge BPE, and
  RobertaProcessing specials — dispatched like the Unigram backend when
  `tokenizer.json` is `model.type: "BPE"`. Reproduces HF id-for-id; unlocks the
  RoBERTa family end-to-end. `embed/tokenize_bpe.go`.
- **`ibm-granite/granite-embedding-125m-english` certified (`encoder`).** RoBERTa
  (pad+1 offset, CLS) on the existing `LoadBERT` forward, with the new byte-level
  BPE tokenizer. `TestGranite_parity` + `TestGranite_encodeEndToEnd`.
- **`encoder.MatryoshkaFloor("Snowflake/snowflake-arctic-embed-m-v2.0")` → 256.**
- **Four more certified embedders (`encoder`), taking the set from eight to
  twelve** — all BERT-family coverage breadth on the existing architectures, no
  new forward: `BAAI/bge-large-en-v1.5` and `mixedbread-ai/mxbai-embed-large-v1`
  (1024-dim CLS-BERT), `Snowflake/snowflake-arctic-embed-m` (768-dim CLS-BERT),
  and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (BERT-shaped
  with the XLM-R Unigram vocab + mean pooling). Each is certified against its
  sentence-transformers reference at cosine 1.000000 with mean↔CLS break-it-first
  (`encoder/coverage_bert_test.go`, `scripts/pin_coverage.py`); the checkpoints
  are gitignored so the gates skip in CI.
- **`encoder.MatryoshkaFloor("mixedbread-ai/mxbai-embed-large-v1")` → 512.** mxbai
  is Matryoshka-trained; the floor is measured by `TestEmbedderCoverage_matryoshka`,
  not asserted.

### Fixed

Remediation of the 2026-07-25 engineering audit (`docs/internal/archive/AUDIT-2026-07-25.md`,
24 findings). Each fix ships with a regression test and a break-it-first check.

- **`encoder`: silent-wrong pooling and dropped biases.** `LoadWeights` now reads
  the declared pooling mode instead of always assuming CLS (#1); the batch and Q8
  attention paths no longer drop the qkv/out_proj biases a bias-carrying checkpoint
  needs (#3); `Load` takes the mmap loader rather than a full heap read (#2).
- **Data races.** `chunk.Register` guards the chunker registry against a concurrent
  map write (#17); `FlatI8.PageStats` locks the pager it shares with `Query` (#9).
  Both are `-race` clean.
- **Determinism.** `sparse` sums scores in a fixed order (was map-iteration-order
  dependent) (#16); `topk` evicts the latest tied minimum rather than an arbitrary
  one, so ties break by index (#6).
- **Robustness / crashes.** `vision` rejects zero-patch grids instead of dividing by
  zero and guards a zero pixel-std (#4, #23); the `markdown` chunker collapses
  frontmatter/fence blocks correctly and drops the quadratic they triggered (#7, #8,
  #15); the `regex` chunker attaches a block at line 0 correctly (#22).

### Performance

Allocation and kernel wins across the decode/index hot paths, each proven
bit-identical or parity-gated (float32-bits / checksum pins, or the certified cosine
suites).

- **`vision`** scratch allocated once per `Forward` (23.2 → 2.3 MB/op) and a
  `linalg.Workspace` threaded through the SigLIP projections (int8 5.87 → 2.42 MB/op)
  (#5, #12).
- **`encoder`** MoE `W2` projection SIMD-ified via a load-time transpose + the
  register-blocked kernel (~3.5× the scalar loop) and `moeMLP` arena-allocated (0
  allocs/op) (#10).
- **`embed`** allocation-free `wordPiece` probe via a pooled buffer (3.5×) and a
  fused `StaticModel.encodeIDs` pooling pass (#11, #18).
- **`ann`/`linalg`/`bm25`** — paged-`FlatI8` reuses one `Workspace` across blocks,
  weight-only Q8/Q4 matmuls gained `*Into` `Workspace` variants, `hnsw` candidate
  sorts moved to `slices.SortFunc`, and `bm25.Build` reuses one tf map across
  documents (#13, #14, #24, #20).

### Changed

- **`embed.L2Normalize` is the single canonical L2-normalize.** The three divergent
  copies (with different degenerate-vector behavior) are unified onto it; the encoder
  embedders route through it. Scale-invariant, so certified cosines are unaffected
  (#21).
- **`encoder.EncodeBatch` deduplicated**, and `ModelQ8` gained `Close()` so both
  `Model` and `ModelQ8` satisfy `io.Closer` (not added to the Hard-tier `Encoder`
  interface, which would break it) (#19).

## [1.11.0] — 2026-07-22

The embedder-coverage release. 1.10.x hardened what was already here; this adds
capability: a pure-Go SentencePiece/Unigram tokenizer, a mixture-of-experts FFN,
declared pooling and explicit BERT loader variants — and on top of those, **six
newly certified embedders**, taking the certified set from two to eight. Coverage
is now a generated table checked against the real checkpoints rather than a
claim, and the one property a serve layer cannot safely guess — whether a model's
vectors may be truncated — is exported rather than documented.

Minor, not patch: new exported API and new capabilities, no breaking changes. The
behavior changes below all **widen** acceptance (things that used to fail now
succeed), so they are additive from a caller's perspective.

### Added

- **Pure-Go SentencePiece/Unigram tokenizer (`embed`).** The Viterbi-decoded
  Unigram path plus its normalizer/pre-tokenizer, which is what unlocks the
  XLM-RoBERTa family end-to-end (`embed/tokenize_unigram.go`). No cgo, no
  sentencepiece dependency.
- **Mixture-of-experts FFN (`encoder`).** Top-2-of-8 routing on alternating
  layers, certifying `nomic-ai/nomic-embed-text-v2-moe`.
- **`encoder.MatryoshkaFloor(model) (min int, ok bool)`.** The smallest width a
  model's embeddings may be truncated to, `ok=false` when it was not trained with
  Matryoshka Representation Learning — and for unknown models, deliberately.
  Slicing a non-MRL embedding returns a unit-length, entirely plausible vector
  that simply retrieves worse; `TestEmbedderCoverage_matryoshka` measures it
  (multilingual-e5-base 1.00 → 0.80 pair recall at a quarter width, while genuine
  MRL models hold their floor). Only two of the eight certified models qualify.
  Existed as documentation and a test-only registry before; nothing downstream
  could consult it.
- **Six newly certified embedders**, each with a hidden-state gate and a
  break-it-first check: `BAAI/bge-small-en-v1.5`, `nomic-ai/nomic-embed-text-v1.5`,
  `FacebookAI/xlm-roberta-base` (forward-only; bare LM),
  `intfloat/multilingual-e5-base`, `BAAI/bge-m3`,
  `nomic-ai/nomic-embed-text-v2-moe`.
- **Generated `docs/embedder-coverage.md`.** Built from the registry in
  `encoder/coverage_test.go`, whose pooling/dimension claims are read back from
  the real checkpoints when present, with a freshness gate so the table cannot
  drift from the code.
- **Declared pooling as a per-model property** (BERT and Nomic loaders), read
  from `1_Pooling/config.json` instead of assumed — the difference between CLS
  and mean pooling is silent and total.
- **Explicit BERT loader variants:** position-id offset (XLM-R's pad+1) and
  optional `token_type` embeddings.

### Changed

Each of these accepts input that previously failed; none removes or narrows
behavior:

- **`LoadBERT` no longer hard-fails on an unparseable tokenizer.** A model whose
  weights load fine but whose tokenizer this build cannot read is now usable
  forward-only (best-effort), which is what certifying bare `xlm-roberta-base`
  requires.
- **`embed.LoadTokenizer` accepts Unigram**, not only WordPiece.
- **`(*encoder.Config).ValidateAssumptions` accepts configs it used to reject**,
  as the loader variants above made those configurations legitimate.
- **`(*embed.Tokenizer).EncodeWithSpecials` reads the tokenizer's post-processor
  template** instead of hardcoding `[CLS]`/`[SEP]`, so models with different
  special-token conventions wrap correctly.

### Removed

- `CODE_REVIEW.md` — every second-pass item shipped in 1.10.1.

## [1.10.1] — 2026-07-20

A follow-up to 1.10.0's security review: a second pass verified each applied fix
for completeness and found that several had landed at the site the first review
named but not at its structurally identical siblings — reintroducing the exact
classes they closed — plus one genuine regression the 1.10.0 dependency bump
carried in silently. **Pin this over 1.10.0.** All fixes; no API change.

### Fixed

- **treesitter parse-timeout guard, broken by the 1.10.0 `gotreesitter`
  0.20→0.40 bump (`chunk/treesitter`).** 0.20's `pool.Parse` returned an ERROR on
  timeout; 0.40 preserves tree-sitter's native partial-tree behavior — it returns
  a TRUNCATED tree with a NIL error and `tree.ParseStoppedEarly() == true`. The
  chunker only checked `if err != nil`, so on a pathological/slow file the
  runaway-parse guard silently stopped firing: instead of falling back to the
  line chunker it chunked a partial AST (degraded boundaries for the unparsed
  tail; the `parseErr` stat never incremented, so the degradation was invisible).
  Now also checks `ParseStoppedEarly()` and falls back, restoring the pre-bump
  contract. Regression-tested against the 0.40 timeout semantics.
- **Uncatchable worker-goroutine panics reintroduced in two matmul entries
  (`linalg`).** The 1.10.0 M2 fix moved shape validation before the parallel
  fan-out so a bad shape panics recoverably on the caller's goroutine — but
  `MatmulBTW8A8Batch` and `MatmulBTAcc64` (+ its Workspace twin) were left
  without it, so a short operand still faulted inside a worker (uncatchable even
  with `recover()`). Both now validate before the fan-out; a grep-checklist
  confirms all nine fan-out matmul entries are covered.
- **OOB-token crash still live in the batched embed-gather paths (`encoder`).**
  1.10.0 replaced the `id = 100` out-of-range fallback with `id = 0` (100 panics
  on any vocab ≤ 100, e.g. the repo's own `vocab_size:4` fixtures) — but only in
  the two single-sequence gathers; `forwardBatch` (the primary EncodeBatch path),
  the Q8 batch, and the token-probe path still used 100. All five now route
  through one shared `clampTokenID` helper so the fallback can't drift again.
- **Qwen2.5-VL tower didn't reach the H7/H8 parity bar (`vision`).**
  `patch_embed.proj.weight` is now shape-checked (5-D Conv3d) like every other
  tensor; `validate()` now rejects a head_dim not divisible by 4 (rotary ÷0), an
  unsupported `hidden_act` (the tower hardcodes SiLU), and a non-positive
  `temporal_patch_size` (→ zero-wide patch embedding).
- **safetensors mmap use-after-unmap on the bare `Tensor` accessors (`embed`).**
  The 1.10.0 H6 KeepAlive fix covered `TensorF32`/`RowDequantizer` but not the
  direct per-`Tensor` decoders (`Float32s`/…/`BFloat16sToF32`/`Float16sToF32`),
  which could be `munmap`'d mid-read → SIGSEGV. `Tensor` now carries an `owner`
  back-pointer and each accessor `runtime.KeepAlive`s it across the read.
- **HNSW `Load` trusted the serialized `mL` (`ann`).** A crafted blob with
  `mL = +Inf` (or `m = 1`) passed every check, then `Add`-after-load overflowed
  `randomLevel` and panicked. `Load` now clamps `m ≥ 2` and recomputes
  `mL = 1/ln(m)` (round-trip-stable), ignoring the stored value.
- **Robustness minors.** `BERT.Embed` truncates `len(ids)` to the position table
  (an unbounded id gather panicked); `Tensor.Elements()` overflow-guards its
  shape product; `MatmulBTInto` validates `a`/`b` (not just `dst`); the WordPiece
  `max_input_chars_per_word` guards a negative value (would emit all-`[UNK]`); and
  a handful of stale doc comments (`madvise_darwin` "Linux/BSD" → "Linux-only",
  a non-existent `Config.Heuristic`, a stale cAST comment) were corrected.

### Changed

- A dozen local, unexported variable renames for clarity (e.g. bert.go's `hd`
  head-index → `headIdx`, matching the head-dim convention everywhere else;
  `gguf` block-size-vs-index vocabulary; the Qwen spatial-merge interleave
  coords). Purely syntactic — no behavior change.

## [1.10.0] — 2026-07-18

A hardening + maintenance release. The centerpiece is a security-focused code
review swept across the untrusted-input boundary — aikit is the parser layer the
ecosystem routes every hostile GGUF/safetensors byte and every persisted index
through, and the clamp/validation discipline the GGUF byte cursor already had was
not uniform across its siblings. Alongside the fixes: two additive API surfaces,
several allocation/throughput wins, arm64 CI, and a toolchain/dependency refresh.
No Hard-tier API breakage — everything here is additive or behavioral, and the
changed kernels stay bit-identical (pinned by the exact-equality parity gates).

### Added

- **`linalg.MatmulBTW4A8Into(ws, …)` — zero-alloc int4 decode matmul.** The
  W4A8 (int4-weight × int8-activation) decode path had no Workspace variant, so
  it allocated the quantized-activation scratch (`make([]int8, M*K)` +
  `make([]float32, M)`) every call — GC pressure per projection per layer per
  token. The Into form quantizes once into the Workspace's reusable buffers
  (0 allocs/op in steady state, M=1 K2048×N2048), the int4 equivalent of the
  existing `MatmulBTW8A8Into`. Bit-identical to `MatmulBTW4A8`.
- **`Close()` on `Model`, `BERT`, `CrossEncoder`, `SPLADE` (`encoder`).** The
  f32 text models retained their mmapped safetensors for life — the only
  deterministic unmap was an internal call the tests reached into, so a
  long-running server swapping models kept ~547 MB mappings alive until a GC
  finalizer ran. `Close()` releases eagerly; idempotent, no-op for a heap-loaded
  model.
- **arm64 CI job.** The matrix was amd64-only, so the NEON f32/i8/SDOT/W4A8 asm,
  the `Dot2x8` pair path, `packedFill`, `detectDotProd`, and the exact-equality
  M-/width-invariance / packed-path bit-identity gates ran only their trivial
  amd64 legs — an arm64-only regression could ship green on the primary
  deployment arch. A native `ubuntu-24.04-arm` job now runs build/vet/-race +
  the `aikit_checks` contract build on real hardware.
- **staticcheck in CI (golangci-lint).** errcheck + govet + ineffassign +
  staticcheck + unused, mirroring ken's conservative config (the packages that
  moved between the repos now report identically). First run: 54 findings → 0.

### Changed

- **BERT attention runs through the pooled scratch arena (`encoder`).** The
  BERT forward (and SPLADE / CrossEncoder, which route through it) allocated
  `qH`/`kH`/`vHT` + an L² scores buffer per (layer, head) and Q/K/V/ctx/… per
  layer via the allocating `matmulBT` — ~432 small mallocs per single-sequence
  forward, and single-threaded (`matmulBT` never consults `wantParallelMatmul`).
  Routed through the same per-goroutine scratch arena the f32 Nomic path uses,
  plus `matmulBTInto` (which parallelizes a lone forward). **432 → 3 allocs/op**;
  bit-identical (the parity gates stay green).
- **`MatmulBTQ8` widens each weight row once (`linalg`).** The row-outer nest
  re-widened every int8 weight row to f32 M times (the O(K) widen dominates the
  vectorized dot at prefill). Swapped to column-outer, widening once per row and
  reusing it across the M activation rows — bit-identical, matching
  `MatmulBTQ4`/`w8a8Span`.
- **O(N) chunk line numbering (`chunk/treesitter`, `chunk/markdown`).** Both
  converted a span/chunk byte offset to a line number by counting newlines from
  byte 0 every call — O(N²/chunkSize), tens of seconds on a multi-MB generated
  file, dwarfing the 1 s parse timeout. Now a one-time newline index (treesitter)
  / index arithmetic (markdown); byte-identical line numbers.
- **`DotI8` rejects unequal lengths.** The exported int8 dot documented
  `len(a)==len(b)` but enforced nothing — a short `b` read past its allocation on
  the SIMD path (arch-dependent garbage/fault) and bounds-panicked on the scalar
  tail. Now panics on mismatch (matching `dotF32`).
- **Documentation contracts corrected.** The `Dot4x4`/`Dot8x4` godoc described a
  sums lane layout the arm64 kernel doesn't produce (portable contract is
  "horizontal-sum each 4-lane block"); the `Chunker` byte-fidelity invariant now
  carves out the `line` chunker's deliberate overlapping-window exception (used
  by every fallback); `linalg`'s "bit-identical to the serial scalar reference"
  was scoped to "a serial run of the same kernel".
- **Toolchain + dependencies.** Go directive `1.26.3 → 1.26.5` across all
  modules; `golang.org/x/text 0.37.0 → 0.40.0`, `golang.org/x/sys 0.45.0 →
  0.47.0`; `go fix` modernizers (range-over-int, `min`/`max` builtins,
  `strings.SplitSeq`). The quarantined `chunk/treesitter` submodule bumps
  `gotreesitter 0.20.2 → 0.40.0` — its byte-exact chunk boundaries are
  best-effort (ADR-010), so some may shift; byte-fidelity + AST-meaningful
  boundaries are gated and hold. Core stays cgo-free, dependency-light (x/text
  only on Linux).

### Fixed

- **HNSW `Config{M:1}` panic (`ann`).** `M=1` set `mL = 1/ln(1) = +Inf`, so
  `randomLevel` overflowed and `Add` panicked on the first insert. Clamp `M ≥ 2`.
- **regex chunker line-continuation depth bug (`chunk/regex`).** A `\` followed
  by a newline (a legal JS/TS/Rust continuation) made the byte-skip jump a line
  start that the `== pos` bookkeeping missed, freezing every later line's depth
  at 0 for the rest of the file — nested definitions read as top-level
  boundaries. Byte-fidelity was unaffected; boundary quality degraded. Fixed with
  `<=`.
- **Vision loaders panicked on mismatched checkpoints (`vision`).** Both towers
  read every tensor with no expected shape and had no config validation, so a
  wrong-shaped projection panicked inside `QuantizeRowsInt8`/`MatmulBT` and an
  absent `patch_size`/`num_heads` divided by zero. Now shape-checked (like the
  text encoder) and config-validated at load.
- **BERT id/seq bounds, OOB-token fallback (`encoder`).** `max_seq_length` is
  clamped to the position capacity; the embedding gather range-checks id/segment;
  the OOB-token fallback substitutes row 0 (the old fallback to id 100 panicked
  on the repo's own `vocab_size:4` fixtures). Q8 forward + batch now honor
  `Cfg.pooling` (was hardcoded CLS). `ValidateAssumptions` rejects
  `scale_attn_weights=false` and an odd head dim. Cross-encoder pair truncation
  uses `longest_first` (a long query no longer starves the document to zero
  tokens); load rejects a degenerate classifier / a tokenizer without
  `[CLS]`/`[SEP]`.
- **Qwen per-image grid divisibility (`vision`).** `forwardViT` validated only
  the global `n_patches % merge²`, not per-image h/w — a grid like `{1,3,4}` with
  merge 2 passed but indexed the window reorder out of bounds. Validated per grid
  entry.
- **top-K NaN handling (`topk`).** At capacity a NaN score slipped past the
  `score <= min` guard (every NaN comparison is false), evicting the true minimum
  and poisoning the ordering. NaN is now rejected in `Push`.
- **GGUF/safetensors parser robustness (`embed`).** The dim-overflow bound
  falsely rejected legitimate dense Q2_K/IQ2_S tensors (widened 2×→4× the data
  section); map-prealloc hints are bounded by the minimum entry size + a cap (a
  hostile count no longer drives a multi-GB prealloc); `Dims` rejects an
  overflowing untrusted dim; array-length / unknown-type / tensor-dim / shard-
  index errors now wrap `ErrFormat` consistently.

### Security

The untrusted-input boundary — hardening reachable from crafted (or, for the
mmap-lifetime and vision cases, merely mismatched) model files and persisted
indexes. None fired on well-formed input, which is why the parity/fuzz gates
stayed green.

- **safetensors sharded load: path traversal (`embed`).** Shard filenames came
  straight from the (untrusted) index JSON's `weight_map` and were joined to the
  index directory, so `{"w":"../../../etc/passwd"}` (or an absolute/UNC path)
  mmap'd and parsed a file outside the bundle — a zip-slip-style arbitrary read.
  Non-plain filenames are now rejected as `ErrFormat`.
- **safetensors header: shape × dtype not cross-validated (`embed`).** Only the
  byte-offset range was checked, so a header pairing a giant shape with a
  1-element byte range parsed and then panicked at inference when a caller indexed
  by shape (violating the never-panic parse contract). Now requires
  overflow-safe `∏shape · dtypeSize == byteRange`.
- **safetensors typed accessors: no alignment check (`embed`).** The `unsafe`
  reinterpret cast (`Float32s`/`Float64s`/`Int64s`/`Int32s`) assumed element
  alignment the format doesn't guarantee — a misaligned conversion is an
  unrecoverable `-race`/checkptr throw and a SIGBUS on strict-alignment ports. Now
  takes the zero-copy view only when aligned, else a byte-copy into an aligned
  slice.
- **GGUF metadata: unbounded array-of-array recursion (`embed`).** Nesting depth
  was ~input/12, so a crafted file drove millions of frames past Go's goroutine-
  stack limit — an unrecoverable abort `recover()` can't catch. Capped at 128.
- **HNSW load: missing product allocation guard (`ann`).** The f32 vector branch
  clamped `ndocs`/`dim` individually but not their product (the int8 branch
  already did), so a ~1 MB header could attempt ~250 GB of allocations. Guarded.
- **mmap lifetime: use-after-unmap (`embed`, `ann`).** Three zero-copy accessors
  (GGUF `RowDequantizer`, safetensors `TensorF32` widening, `FlatI8.query`)
  aliased an mmap region whose finalizer could `munmap` it mid-read → SIGSEGV.
  Added `runtime.KeepAlive` backstops.
- **Unrecoverable matmul panics (`linalg`).** Above the parallel threshold a
  shape-violation panic fired on a worker goroutine — uncatchable even by a
  caller's `recover()`, so a mismatched checkpoint could hard-kill the process.
  Shape validation now runs before the fan-out (recoverable, caller-side), and
  the `Wrap*`/`Quantize*` constructors reject negative dims / overflow.

## [1.9.0] — 2026-06-17

A weight-memory substrate (mmap + `madvise` + a span-residency cache) lifted into a
new leaf package, plus the payoff it unlocks: an int8 ANN index that can be queried
larger than RAM. The mechanism is the one goinfer proved demand-paging a 35B-A3B
MoE's experts; only the generic, model-free core moves here. All new surface is
**Experimental** (outside the 1.0 guarantee).

### Added

- **`mmap` — new leaf package: read-only mapping + residency control.** The
  read-only `MAP_PRIVATE` mapping primitive that `ann` and `embed` each kept a
  private byte-identical copy of (to avoid an `ann→embed` edge) now lives once, in a
  zero-dependency leaf both import:
  - `MapReadOnly` / `Unmap` — the mapping itself.
  - `Advise(span, willNeed)` — `MADV_WILLNEED` / `MADV_DONTNEED` residency hints.
    Firm cap on Linux; on darwin `WILLNEED` is an advisory prefetch and eviction is
    an OS-discretion no-op; best-effort no-op on the BSDs/Windows.
  - `SpanCache[K]` — a demand-signal-agnostic LRU of page-aligned spans bounded by a
    byte budget: `Add` registers a member's spans, `Touch` faults it in and releases
    the LRU tail to stay under budget. No model logic — the caller owns the demand
    signal. Gated by an eviction unit test and a model-free property test that a
    `MADV_DONTNEED`-released read-only mapping re-faults byte-identical.
  - `PageAlignedInterior`, `AvailableRAM`, `AutoBudget` helpers.

  Leaf invariant: stdlib-only, except `golang.org/x/sys/unix` on **darwin only**
  (the stdlib has no `madvise` wrapper there) — so it stays invisible to the core
  dependency invariant on Linux, and cgo-free everywhere. `!unix` keeps the existing
  heap-read fallback.
- **`linalg.WeightMat.MappedSpan(base, end)`.** Returns the page-aligned interior of
  a weight's int8/int4 backing bytes iff they alias the given mapping, else nil
  (f32/heap-backed → skip). The bridge from a `WeightMat` to `mmap.SpanCache`.
- **`ann.LoadFlatI8MmapPaged(path, budget)` — query an int8 index larger than RAM.**
  The int8 code block is split into blocks that page in and out of the mapping
  through `mmap.SpanCache` under `budget` (≤ 0 auto-selects ~half of available RAM);
  cold blocks re-fault from the read-only mapping. Per-block scoring is
  byte-identical to the resident whole-corpus scan (deterministic query quant +
  independent row dots), so paging changes residency only, never results — gated by
  a paged-equals-resident test that also asserts eviction count > 0. The default
  `LoadFlatI8Mmap` is unchanged (paging is opt-in). `FlatI8.PageStats()` exposes the
  cache's hit/miss/eviction counts. A paged index serializes concurrent `Query`
  calls (the pager is stateful) — the cap traded against cross-query parallelism.

### Changed

- **`ann` and `embed` now call the `mmap` leaf** instead of their local
  `mmapReadOnly`/`munmap` copies, which are deleted. Pure refactor — the existing
  `OpenSafetensorsMmap` / `OpenGGUFMmap` / `LoadFlatI8Mmap` behavior and tests are
  unchanged.

## [1.8.1] — 2026-06-15

Supersedes **1.8.0, which is retracted** — it was tagged before the release gate
passed (a missing CHANGELOG compare link) and before the GGUF parser hardening
below. 1.8.1 carries the same Qwen2.5-VL vision tower + arm64 W4A8 work as 1.8.0
plus the fix; pin this instead of 1.8.0.

### Fixed

- **GGUF metadata parser: nested-array allocation blowup (`embed`).** A hostile
  array-of-arrays where every level claims a count near the remaining input drove
  `make([]any, 0, n)` at each nesting level; since the nesting depth is itself
  ~input/12, total preallocation was O(input²) — a ~1 MB file parsed in ~700 ms,
  surfacing as a `FuzzParseGGUF` "context deadline exceeded" slow path. The eager
  array capacity is now bounded by a small constant (`append` still grows to the
  true element count), making allocation linear in the input (same ~1 MB file:
  ~700 ms → ~100 ms). Gated by `TestParseGGUF_nestedArrayBomb` (asserts bounded
  allocation). Parse output unchanged for valid files.

## [1.8.0] — 2026-06-14 [RETRACTED]

Retracted (see `retract` in go.mod): superseded by 1.8.1. The Qwen2.5-VL vision
tower and arm64 W4A8 changes below shipped unchanged in 1.8.1.

### Added

- **`vision.QwenVisionEncoder` — Qwen2.5-VL vision tower (second ViT family, dynamic
  resolution).** A pure-Go fp32 forward of the Qwen2.5-VL `.visual` submodule, added
  additively alongside the SigLIP `Encoder` (unchanged). Unlike SigLIP (fixed
  896×896 → 256 tokens, learned absolute pos, LayerNorm, gelu-tanh MLP), this is
  dynamic-resolution: `LoadQwenVisionEncoder(dir, quant)` +
  `Forward(pixelValues []float32, gridTHW [][3]int)` take pre-patchified
  `pixel_values [n_patches, patch_dim]` + per-image `(t,h,w)` grids (preprocessing
  lives upstream in goinfer) and return the merged image embeddings
  `[n_merged, out_hidden_size]` in original patch order. Implements the Qwen deltas:
  Conv3d patch embed (as a matmul), 2D rotary position embedding, RMSNorm,
  windowed + full attention (`fullatt_block_indexes` + `cu_seqlens`), a gated SiLU
  MLP, and the spatial-merge patch merger (erf-GELU). Reuses the `qmat` W8A8 wrapper
  for the projections (the patch-embed matmul stays f32). `ForwardViT` exposes the
  pre-merge hidden for stage-isolated parity. Gated by `TestQwenVisionEncoder_parity`
  (cosine ≥ 0.9999 on both the ViT pre-merge hidden and the merged features, fp32 —
  measured 1.0) vs the HF `Qwen2_5_VisionTransformerPretrainedModel` golden
  (`scripts/pin_qwen25vl_vision.py`, transformers 5.12). W8A8 quant of this tower and
  a resident-GPU path are follow-ons.

### Changed

- **arm64 W4A8 NEON now uses the in-register scale-fold kernel (`dotW4A8FoldSDOT`).**
  `dotW4A8` is wired to the fold path (mirroring the amd64 dispatch) and validated on
  an M1 Pro; the old per-group `dotW4A8GroupsSDOT` path is removed. The W4A8 matmul
  itself is ~1.33–1.43× faster on M1 (e.g. K2048×N2048 M=1: 250→175 µs), narrowing
  the W4A8/W8A8 gap from ~2.25× to ~1.5×. `MatmulBTW4A8` accuracy unchanged (relL2
  0.007–0.009); `TestW4A8_dotMatchesScalar` passes at 1e-5 vs scalar. amd64, W8A8,
  and f32 `MatmulBT` untouched.

## [1.7.3] — 2026-06-13

### Fixed

- **amd64 AVX2 `Dot8x4`/`Dot4x4` produced wrong results for odd `n4`** (= kSpan/4;
  e.g. K=13 → n4=3, K=300 → n4=75). The register-blocked kernels (`dotFMA4`/
  `dotFMA8` in `dot_amd64.s`) consume two 4-element groups per 256-bit YMM
  iteration; a trailing single group (n4 odd ⇒ n%8==4) was accumulated with an
  **XMM / VEX.128** FMA, which zero-extends and so **wiped the upper 128 bits** of
  each YMM accumulator — discarding the loop's lane-4..7 partials. The tail now
  loads the 4 a/b floats into zero-extended YMMs and FMAs in YMM form, preserving
  the upper lanes (the zero upper operand lanes contribute 0). **Latent since
  1.6.0** (the blocked-GEMM hoist): nothing routed odd-`n4` shapes through the
  blocked kernel until **1.7.2** removed `MatmulBT`'s small-shape threshold — so
  this is what makes 1.7.2's `MatmulBT` M-invariance actually correct on amd64.
  amd64+AVX2 only — arm64 NEON and the non-AVX2 scalar fallback were always
  correct (why every prior release and arm64 CI passed). It surfaced through
  `MatmulBT` and, transitively, the f32 *reference* of `MatmulBTQ4` / `MatmulBTW4A8`
  (whose quant kernels were never wrong — one root cause). The 1.7.2 threshold
  removal **stands** (fixed forward, not reverted). Gated by a new direct kernel
  regression test (`TestAVX2_blockedKernels_oddN4`, odd + even `n4` vs scalar) plus
  `TestMatmulBT_MConsistent`.

## [1.7.2] — 2026-06-12

### Changed

- **`linalg.MatmulBT` is now M-invariant** — its per-output f32 result no longer
  depends on `M`. Each output `dst[i,j]` is bit-identical whether a row is computed
  alone (M=1) or inside a batch (M>1). 1.6.0's blocked-GEMM hoist left a MAC-count
  threshold in `MatmulBT` that routed small matmuls to a **naive dot-per-output
  span** — a different f32 reduction order than the blocked kernel — so the same
  projection at M=1 vs M=K differed by the f32 reassociation (~1e-5). The threshold
  is **removed**: all `M` route through the one blocked-kernel order, so the
  per-output result is independent of `M` (and of the parallel fan-out width, which
  already shards 8-aligned). Measured bonus: the blocked kernel is **2–3.8× faster**
  than the naive span it replaces at small-M decode/attention shapes. Gated by
  `linalg.TestMatmulBT_MConsistent` (also pins `blockedFill`'s internal
  paired-vs-odd-row consistency); the invariant is documented on `MatmulBT`. The
  quantized kernels (`MatmulBTW4A8`/`Q8`/`W8A8*`) and the encoder (which routes
  through `MatmulBTInto`, always blocked) are untouched.

  **Correction (post-release):** this entry originally claimed the change "fixes
  same-model speculative-decoding parity (regressed since 1.6.0)" in a downstream
  consumer (goinfer). That was wrong. The speculative-parity failure was
  **consumer-side** — goinfer's dense attention computed QKᵀ/AV through two code
  paths that were not bit-identical — and was fixed there with f64 accumulation
  (`MatmulBTAcc64`, which aikit never touched). aikit's M-invariance is an
  independent property; it did **not** fix that bug, and removing the threshold
  *transiently* shifted goinfer's f32 attention numerics until goinfer moved that
  path onto the f64 kernel. The lesson, recorded: f32 `MatmulBT` is reassociation-
  sensitive — consumers needing cross-M / cross-path bit-exactness should use
  `MatmulBTAcc64` or the integer kernels, which is what goinfer now does.

## [1.7.1] — 2026-06-12

### Added

- **`linalg.WrapInt8` / `linalg.WrapInt4` — zero-copy constructors for
  already-quantized `WeightMat` weights (Experimental).** The inverse of the
  `Int8()`/`Int4()` accessors: they wrap caller-owned, pre-quantized slices (int8
  + per-row scales; or packed int4 nibbles + per-group scales + group size)
  **without copying or re-quantizing**, aliasing the caller's backing arrays the
  same way `WrapF32` does. This fills the gap that blocked the goinfer
  `decoder.weightMat` → `WeightMat` migration: 1.7.0 shipped only quantize-from-f32
  constructors (`QuantizeInt8`/`QuantizeInt4`), but the decoder's `.giw`
  deserialization reads weights already quantized and **zero-copy-aliases the int4
  nibbles straight off an mmap'd blob** — re-quantizing from f32 would have
  regressed both that fast load and its OS-page-cache residency. The wrap path
  preserves both. Shape-validated (panics on a mismatched length, like
  `QuantizeRowsInt8`). Additive.

## [1.7.0] — 2026-06-12

### Added

- **`vision` — a pure-Go SigLIP / ViT image encoder (Experimental).** aikit gains
  image embeddings: decode (stdlib `image/jpeg`+`png`) → preprocess (resize +
  normalize → `pixel_values`, with a pre-decode pixel-count guard against
  decompression bombs) → a pure-Go transformer forward (bidirectional MHA +
  gelu-tanh MLP, patch-embed conv as im2col+matmul) → `last_hidden_state`. The
  attention/FFN projections run f32 or int8 W8A8; parity is cosine vs the HF
  `SiglipVisionModel` golden (`scripts/pin_siglip_vision.py`) — **1.0 f32,
  ~0.9999 int8**. No cgo, no new external dependency (it's `embed`+`linalg`; the
  image codecs are stdlib). It exposes an import-free GPU-export seam
  (`GPUWeights`/`GPUMat`) and a `RegisterResident` inversion so goinfer's WebGPU
  backend attaches without the core importing it — the same seam pattern as
  `encoder.Backend`. **This makes aikit the only cgo-free image-embedding
  retrieval library** (image→image similarity and image-as-document indexing work
  day one). Additive — a new leaf package; nothing existing changes.

  The code moves in from goinfer's `vision` package (same author, MIT), verbatim
  and parity-preserving; goinfer deletes its copy and imports aikit's on the next
  pin bump. The Gemma-specific connector (the vision→LLM-token projector and the
  image-soft-token sentinels) stays in goinfer — aikit ships the encoder, not the
  multimodal glue. **Not yet present: a SigLIP text tower**, so true text↔image
  retrieval is a documented follow-up (Gemma drives the text side with its LLM,
  which aikit doesn't have); image→image and image-as-document need only the
  encoder shipped here.

- **`linalg.WeightMat` — a precision-hiding quantized-weight matrix (Experimental).**
  One type that holds an f32 / per-row-int8 / group-int4 weight behind a uniform
  `MatmulBT(a, dst, M)` (+ a `Workspace`-scoped variant honoring `SetThreshold`/
  `SetWorkers`), a `Row(i)` dequant for embedding lookup, and raw accessors
  (`Int8()`/`Int4()`/`F32()`) for GPU export, serialization, and a consumer's own
  kernel. It consolidates the weight-matrix wrapper that was open-coded **three
  times** — aikit `encoder.LayerWeightsQ8`, goinfer `decoder.weightMat`
  (f32/int8/int4-group/W8A8), goinfer `vision.qmat` (f32/W8A8). It hides **storage
  only** — precision, scales, dispatch; model *policy* (which table gets which
  precision, int4 group size, GPU-backend routing) stays with the consumer.
  Dispatch reuses the existing `linalg` kernels — **no new asm**; outputs are
  bit-identical to each consumer's prior kernel call. Additive.

### Changed

- **`encoder` int8 (Q8) path migrated onto `linalg.WeightMat` — bit-identical, zero
  output change.** `LayerWeightsQ8` now stores each of the five big projections
  (Wqkv/OutProj/fc11/fc12/fc2) as a weight-only-Q8 `linalg.WeightMat` instead of an
  open-coded `[]int8` + `[]float32` scales pair. `LoadWeightsQ8` quantizes via
  `linalg.QuantizeInt8`, which is bit-identical to the encoder's `quantizeRowsInt8`
  (same per-row symmetric max/127 round+clamp), and the forward still drives the
  encoder's own baked-scale `matmulBTQ8Into` over the codes/scales pulled via
  `WeightMat.Int8()` — the kernel is unchanged (it is numerically distinct from
  `linalg.MatmulBTQ8`: large-M dequant-once-into-scratch). `TestModelQ8_cosineMatchesF32`
  holds at cosine 0.997, full Q8 golden/parity suite green, `-race` clean. The
  removed `LayerWeightsQ8` int8/scales fields are Experimental-tier surface. First
  of the three consumer migrations; goinfer's `vision.qmat` and `decoder.weightMat`
  migrate against the released minor.

## [1.6.0] — 2026-06-12

### Changed

- **B-panel packing for the blocked GEMM at large K — prefill 46%→69% of peak, and the
  encoder's own K=3072 fc2 +15%, bit-identically** (arm64). The cache-blocked `MatmulBT`
  above still left large-K shapes short of the K=768 tiles' ~70% (M=512×4096×4096 sat at
  46%): at large K the 8 b-rows a `Dot2x8` reads simultaneously are K·4 bytes apart and
  collide in the same L1 cache sets (associativity conflicts). The fix packs each 8-row
  b-group into a contiguous low-stride buffer (rows ~kBlock apart) before the kernel, so
  the loads are conflict-free. It copies the same values in the same order, so it stays
  **bit-identical** to the unpacked path — the encoder's fc2 (K=3072) now packs and runs
  **+15%** with golden parity unchanged. Gated to K≥2048 (the K=768 transformer dims are
  already low-stride and keep the unpacked path) and to arm64 (the packed kernel is the
  NEON `Dot2x8`; amd64 keeps its AVX2 path — AVX2 packing is deferred to §2.4). Measured:
  prefill M=512×4096×4096 **46%→69%**, fc2 M=512×3072×768 **64%→75%**. Large M (≥~2048)
  recovers less (≈53%) — the a-panel re-read needs full 3-level (Goto) blocking, a
  deferred lever. `MatmulBTInto`/`MatmulBT` dispatch into it automatically.

- **`linalg.MatmulBT` is now cache + register blocked — ~6× faster at prefill shapes,
  and the blocked GEMM is now shared, not duplicated.** `MatmulBT` was a naive
  dot-per-output span that re-streamed `b` from DRAM once per `a`-row; a cross-repo gate
  (goinfer) measured it at **~7% of this M1 Pro's f32 peak** at decoder prefill shapes —
  every kit consumer of `MatmulBT`, not just one, was paying for the missing cache
  blocking. The encoder already had a proper blocked + register-tiled GEMM (32×32×768
  tiles + Dot8x4/Dot2x8); that implementation is **hoisted into `linalg`
  (`matmul_blocked.go`) as the single home** behind `MatmulBT`, the encoder's transformer
  matmuls, and other consumers. Measured: M=512×4096×4096 prefill **7%→46% of peak
  (~6.3×)**; the K=768 transformer tiles **68–75%**. Small matmuls (e.g. attention QKᵀ)
  keep the naive span via an M·K·N threshold, so they don't regress. `MatmulBT`'s results
  now differ from the old naive order by ~1e-5 (f32 reassociation, same class as the
  v1.2.0 ann change); its documented width-invariance is **preserved** — column shards are
  aligned to the kernel's 8-column group so fan-out width stays numerically inert
  (`TestParallelWidth_bitIdentical`). `MatmulBTAcc64` (f64-exact) is unchanged for callers
  needing determinism. The encoder's golden parity is bit-identical across the move.

### Added

- **`linalg.MatmulBTInto(dst, a, b, M, K, N)` — serial cache+register-blocked GEMM into a
  caller-provided dst** (Experimental surface). The entry for consumers that own their own
  parallelism (the encoder row-splits a batch across cores; goinfer's batch/vision paths
  likewise) and want each matmul serial; for process-level column parallelism use
  `MatmulBT`. Overwrites `dst` (len ≥ M*N).

- **`linalg.Dot2x8` + 2×8 register micro-kernel for the encoder GEMM** (arm64;
  ~1.5–1.6× on the encoder's f32 matmuls). The blocked GEMM's inner kernel was
  `Dot8x4`, a 1×8 micro-kernel: one shared `a`-strip reused across 8 b-rows, but
  each b-load feeding exactly one FMLA and only 8 live accumulators — both
  load-bound and below the ~16 accumulators that hide NEON FMA latency. A
  peak-fraction gate (`BenchmarkGEMMPeakFraction`) measured it at **40–49% of this
  M1 Pro's f32 ceiling**, where the ceiling itself was *measured* — a
  register-saturating FMA micro-bench clocked **95.4 GFLOPS** (≈15 f32-FMA/cycle,
  confirming the 4-pipe Firestorm figure, not the 8 a back-of-envelope assumed).
  ≤50% ⇒ headroom. `Dot2x8` folds 2 a-rows × 8 b-rows into one call with 16
  accumulators held across the K loop, so each b-load now feeds 2 FMLAs. It
  computes each dot in the **same accumulation order** as `Dot8x4`, so results are
  bit-identical and the encoder golden parity is unchanged (no tolerance change
  needed). Wired into `encoder`'s blocked matmul for M≥2 row-pairs (the odd final
  row and amd64 keep the `Dot8x4` path). Measured: peak fraction **40%→68–73%**;
  encoder FC matmuls **1.52–1.58×** (L80 fc11 8.7→5.5 ms, L512 fc11 54.8→34.6 ms);
  end-to-end **EncodeBatch 1.36×**, single-doc encode **1.27×**. M=1 decode/gemv
  paths are untouched. arm64 NEON only; the AVX2 port is gated on Zen 4+ access
  (roadmap §2.4). cgo-free, `-race` clean, Windows cross-build verified.

- **`embed.SafetensorsFile.TensorF32` / `TensorI32` — shape-checked typed tensor
  reads** (Hard-tier surface; additive). `TensorF32(name, want...)` reads a tensor as
  `[]float32`, widening BF16/F16 to f32 and optionally asserting the shape; `TensorI32`
  is the int32 sibling (GPTQ packed tensors). Lifts the read pattern that the loaders
  hand-wrote repeatedly — the in-repo `encoder.loadF32` now delegates to it (validated
  against the CodeRankEmbed / MiniLM / SPLADE / cross-encoder loaders), and the
  cross-repo consumer (goinfer, which open-codes the same shape-checked dispatch ≥6×
  in its decoder/vision loaders) can drop its helpers at its next aikit bump. Surfaced
  by the 2026-06-12 goinfer cross-repo review.

## [1.5.0] — 2026-06-11

### Added

- **`encoder.LoadCrossEncoder` / `CrossEncoder.Score` — BERT cross-encoder reranker**
  (Experimental surface). Scores a (query, document) pair *jointly* — the trunk runs
  over `[CLS] query [SEP] document [SEP]` (token types 0/1), then the `[CLS]` hidden
  state goes through the BERT pooler (dense + tanh) and a linear classification head
  to a relevance logit; rank candidates by descending `Score`. A small additive step
  on the v1.4.0 BERT trunk: `LoadCrossEncoder` reuses `LoadBERT` and adds the pooler +
  head, and `hiddenStates` gained token-type segments for the pair. Parity-pinned to
  **cross-encoder/ms-marco-MiniLM-L-6-v2** (hugot's CrossEncoders headline + Antfly's
  reranker default) at Δ 5e-6 — both the forward and the end-to-end pipeline (aikit's
  own pair tokenization matches HF); golden via `scripts/pin_crossencoder.py`. aikit
  now covers both halves of reranking (bi-encoder + cross-encoder), cgo-free.

### Documentation

- **Blob format-stability policy decided + documented** (no code change). Serialized
  index blobs (`ann.HNSW`/`ann.FlatI8` `MarshalBinary`) are **rebuild-per-minor**
  pre-1.0 — not a stable cross-version interchange format; `Load*` rejects any other
  version with `ann.ErrFormat` (loud, never a silent misread), so the policy is
  enforced by construction. README gains a "Serialized blob formats" section; a
  FORMAT-BUMP CHECKLIST at each version const specs the next bump to bundle a reserved
  header-flags word (anti-churn) + the HNSW float32-vector alignment for a future
  zero-copy `LoadHNSWMmap`.

### Fixed

- **int8 reranker (`encoder.LoadQ8`) is now latency-competitive with f32** — was ~5×
  slower on arm64 (consumer report from ken evaluating an int8 default). Two causes,
  both fixed: (1) the q8 forward allocated per-call/per-layer scratch where the f32
  path pools it — ~4.4 GiB for a 50-doc rerank — now mirrored to the shared scratch
  arena (10 MB/op, in line with f32); (2) the bigger one, the weight-only matmul
  widened int8→f32 in a *scalar* inner loop (~26× slower than the f32 SIMD matmul),
  now dequantizes the weights once per matmul and runs the vectorized `matmulBTInto`.
  Net: q8 reaches ~f32 latency at ¼ the weight storage, with weight-only numerics
  unchanged (cosine 0.997 vs f32). Full W8A8/SDOT (even faster) was rejected — it
  quantizes activations and fell below the 0.97 reranker bar. A `-benchmem` rerank
  bench guards against silent regression.

## [1.4.0] — 2026-06-11

### Added

- **`linalg.Workspace.SetThreshold` + `(*Workspace).MatmulBT` / `MatmulBTAcc64` —
  per-Workspace parallelism scoping** (Experimental surface). The process-wide
  `SetParallelThreshold` / `SetParallelWidth` globals are now *defaults*: a Workspace
  can override the parallelization threshold (`SetThreshold`) alongside the existing
  width scoping (`SetWorkers`), so independent decode streams tune their own
  parallelism without mutating — or racing on — a global. The W8A8 path
  (`MatmulBTW8A8Into` / `Batch`) honors the scoped threshold, and the f32 matmuls
  gained Workspace methods. A zero-value Workspace inherits the global defaults;
  parallelization stays numerically inert (bit-identical results). Settles the API
  shape before `linalg` graduates from Experimental.
- **`ann.ErrFormat` + `embed.ErrFormat` — typed sentinel errors for the blob
  loaders** (additive; `embed.OpenSafetensors*` is Hard-tier, gains only a wrapped
  sentinel). Every versioned-blob load path now wraps a sentinel so callers can
  `errors.Is(err, ann.ErrFormat)` instead of string-matching: `ann.ErrFormat` across
  `Load` (HNSW), `LoadFlatI8`, and `LoadFlatI8Mmap` (bad magic, unsupported version,
  truncated / inconsistent blob — the mmap path via the shared parse, so its open/
  mmap I/O errors stay un-tagged); `embed.ErrFormat` across `OpenSafetensors*` /
  `OpenGGUF*` (bad magic, unsupported version, truncated header). Per-tensor lookups
  (tensor-not-found, use-after-Close) are deliberately not wrapped. Error messages
  are otherwise unchanged.
- **`encoder.LoadSPLADE` / `SPLADE.Expand` — in-process SPLADE expansion**
  (Experimental surface). A SPLADE model is a BERT encoder (`LoadBERT`) plus a
  masked-LM head; `Expand(text)` projects each token's hidden state to vocab logits,
  applies log(1+ReLU), and max-pools to a `sparse.SparseVec` — so learned-sparse
  retrieval runs end-to-end in-process (`Expand` → `sparse.New` / `Index.Query`), no
  Python at query time. This completes the `sparse` package: the index half shipped
  in 1.2.0, the expansion head ships now. Parity-pinned to
  naver/splade-cocondenser-ensembledistil (golden via `scripts/pin_splade.py`):
  identical term sets and cosine 1.000000 across cases. Reuses the §2.2 BERT forward
  (prefix-aware now, so `LoadBERT` also reads raw `BertForMaskedLM`). Adds a small
  `encoder → sparse` edge.
- **`embed.Load` now reads the standard Model2Vec format** (Hard-tier `embed.Load`
  gains capability — additive). Previously it required the vocabulary-quantized
  layout (`embeddings` + `mapping` + `weights` tensors, e.g. `potion-code-16M`); it
  now also loads the standard layout with only an `embeddings` tensor (token ids
  index rows directly, plain mean pooling), so **`minishlab/potion-retrieval-32M`**
  — the strongest static *retrieval* model — works (parity cosine 1.000000 vs
  `StaticModel.encode`, golden via the new `scripts/pin_retrieval.py`). Docs now
  point general-retrieval users to it over the code-tuned model. `potion-code-16M`
  is unregressed.

- **`encoder.LoadBERT` / `BERT.Encode` — MiniLM-class BERT encoder** (Experimental
  surface). A second encoder architecture alongside CodeRankEmbed, implementing the
  three axes a sentence-transformers BERT model differs on: learned ABSOLUTE
  position embeddings (not RoPE), a GELU FFN (not SwiGLU), and mean pooling (not
  CLS). `LoadBERT(dir)` + `Encode(text)` is the cgo-free equivalent of
  sentence-transformers' `.encode()`. Parity-pinned to all-MiniLM-L6-v2 (golden via
  the new `scripts/pin_minilm.py` + the §2.1 toolchain): hidden states match to
  ~1e-6 and the sentence embedding to cosine 1.000000, with aikit's WordPiece
  producing the same token ids as HF. Kept in a separate `bert.go` — the
  CodeRankEmbed path is untouched (additive, no regression). Turns "two specific
  models" into "the BERT family you already use."

### Changed

- **`linalg`: removed the unused persistent worker pool; `Workspace.SetWorkers` is
  now a per-Workspace fan-out width cap** (Experimental surface — `(*Workspace).Close`
  is removed). The spin-then-park pool (built for decode hot-worker reuse) measured
  neutral and shipped unused: the serial-decode threshold keeps M=1 decode serial —
  the regime it targeted — so it only ran where goroutine spawn is already amortized.
  Deleted 154 lines of single-dispatcher concurrency. `SetWorkers` keeps its useful
  role — capping the spawn fan-out (e.g. to the P-core count on heterogeneous CPUs) —
  now as a numerically-inert width field on the per-call spawn path; `Close` is gone
  (nothing to stop). The design + measurement live in git (commit 2df6b52).

## [1.3.0] — 2026-06-10

### Added

- **`linalg.MatmulBTAcc64` — f64-accumulating A·Bᵀ matmul** (Experimental surface).
  Same shape contract as `MatmulBT` (dst[M,N] = a[M,K]·b[N,K]ᵀ, all `[]float32`),
  but each output dot accumulates in float64 in sequential order — **bit-identical
  to a scalar f64 reference** (measured max Δ 0), not just close. For where f32
  reassociation error is amplified downstream: a transformer attention feeding a
  discrete MoE top-k router, where a ~1e-6 f32 difference flips an expert at a
  near-tie and changes generated tokens; f64 drops it to ~1e-15. Keeps the
  parallelism over N, so it's ~6.5× faster than a single-threaded scalar f64 matmul
  (and ~3.7× slower than f32 `MatmulBT`, M=512/K=128/N=2048). `MatmulBT` is
  unchanged — prefer it for dense models where f32 is fine.
- **`ann.Config.Int8` — int8-quantized HNSW** (Experimental surface). The HNSW
  graph's vectors are stored as int8 (per-vector symmetric quantization) instead of
  float32 — ¼ the vector memory, and the persisted/`//go:embed`-ed blob shrinks to
  match. Build, search, and persistence all run in the integer domain (a new
  exported `linalg.DotI8` is the node-node primitive; the query is quantized once
  per search via a prepared `queryVec` threaded through the search — the float32
  path is behaviorally unchanged). Recall is essentially unaffected:
  `TestHNSW_int8RecallGate` and `TestHNSW_int8_real` measure recall@10 Δ0.0000 vs
  the f32 HNSW on real Model2Vec embeddings (the gate the roadmap required before
  building this). The persisted format is bumped to **v3** (an int8-mode byte +
  int8 codes/scales); `Load` rejects the brief-lived v2, like the v1→v2 bump.
- **`ann.FlatI8` persistence — `MarshalBinary` + `LoadFlatI8`** (Experimental
  surface). The int8 index — the one you'd most want to `//go:embed` (¼ the float32
  memory at ~equal recall, per the benchmarks) — now serializes to a versioned blob
  and loads back query-ready, like `ann.HNSW`. Same discipline: little-endian
  versioned format, a bounds-checked cursor, an overflow-safe payload-size check
  before allocation, and a `FuzzLoadFlatI8` target (plus the previously-unwired
  `FuzzLoadHNSW`) now in the CI fuzz smoke + nightly. Quantize the corpus once
  offline, embed the bytes, skip re-quantization per process.
- **`ann.LoadFlatI8Mmap` — zero-copy mmap load + `FlatI8.Close`** (Experimental
  surface). Memory-maps a FlatI8 blob and *aliases* the int8 codes straight from
  the read-only mapping (the codes are 1-byte, so no alignment constraint), copying
  only the tiny scales — so a large embedded index is query-ready instantly (no
  parse-and-copy) and its bytes live in the OS page cache, not the Go heap.
  `Close` releases the mapping (a finalizer is the backstop); querying after Close
  panics. Non-unix falls back to a heap read (same result). HNSW zero-copy is a
  follow-up — its float32 vectors need format-level alignment and its graph is
  parsed regardless.

## [1.2.1] — 2026-06-10

Docs/CI only — no code or API changes. These edits missed the v1.2.0 tag, so
pkg.go.dev rendered a stale package graph on the module front page; this tag
corrects what it renders.

### Documentation

- **Package DAG + dependency table synced with v1.2 reality** (`README.md`,
  `docs/architecture.md`): `ann → linalg, topk`; `bm25` and `sparse → topk`;
  `bench → ann` (its only production dep — `embed` is test-only); added the
  `sparse` and `bench` nodes and the `ann → linalg` edge (`ann` scores through
  `linalg`'s SIMD dot kernels since v1.2).
- **`chunk/treesitter` lockstep wording softened** — from unconditional "versioned
  in lockstep with the core" to "tagged in lockstep whenever the submodule itself
  changes," matching practice (its code is unchanged since v1.0.0, and the
  Hard-tier `chunk.Chunker` contract means an unchanged submodule keeps working
  across core minors; no 1.1.x or 1.2.x submodule tag).
- **CI: pinned the Windows job to `windows-2025`** (was `windows-latest`) ahead of
  GitHub's 2026-06-15 runner redirect, so the image can't shift unannounced.

## [1.2.0] — 2026-06-09

### Changed

- **`embed`: `SafetensorsFile.Tensor()` now errors after `Close()`** (§3.3) instead
  of returning a tensor aliasing a possibly-unmapped region — a guard for the
  common use-after-close mistake. The `Tensor` doc gains a WRONG/RIGHT example for
  the harder held-slice-outlives-Close trap (copy out, or keep the file alive).
- **`encoder` forward is now internally pooling-parameterized** (groundwork for
  BERT-family support, §2.5; no behavior or API change). The CLS extraction in
  both f32 forwards is now a `poolOne` seam (CLS default / mean alternative, the
  batched path masking padding via `realLen`), kept unexported until a
  parity-pinned mean-pooled model exists. CodeRankEmbed stays CLS — golden
  unchanged.
- **`ann.HNSW` build is ~20% faster with 7× fewer allocations** (no graph/recall
  change). Profiling the build (which the Alg-4 default below made heavier) found
  two pure-overhead hotspots: a fresh `map` allocated per search step, and
  `container/heap`'s `interface{}` API boxing every candidate — ~23M allocations /
  3.6 GB for a 10k×256-d build. Replaced with a generation-stamped visited buffer
  (reused across searches; pooled per concurrent `Query`) and a concrete typed
  heap (no boxing). The build now does 3.3M allocs / 1.3 GB and runs ~17.2 → 13.5s
  (10k); recall is byte-for-byte identical. The remaining build cost is the Alg-4
  diversity dot products — inherent to the recall win, not overhead.
- **`ann.HNSW` now defaults to the Algorithm-4 diversity heuristic for neighbor
  selection** (Experimental tier; was plain M-nearest, Algorithm 3). The `bench`
  harness exposed that the old selection capped HNSW recall@10 at ~0.68 on
  clustered real embeddings (and barely improved with `EfSearch`) — its edges
  piled into near-clone clusters and never reached the rest of the graph. The
  heuristic fans edges across directions; on the same real Model2Vec corpus
  recall@10 went **0.68 → 1.00** (and 0.57 → 1.00 on a synthetic clustered set),
  at ~2× build cost and unchanged query latency. `Config.SimpleNeighbors` opts
  back to the cheaper-to-build Algorithm 3. The persisted-index format is bumped
  to **v2** (one byte for the selection mode); `Load` rejects the brief-lived v1.
- **`ann` similarity now uses the SIMD dot kernel.** Both backends scored every
  candidate with a scalar `float64` dot loop and didn't import `linalg` at all.
  `Flat.Query` (the brute-force scan — 100% of its query cost) and `HNSW.sim`
  (the graph-walk inner loop) now use the SIMD kernels. `Flat.Query` further
  streams 8 candidates per pass through `linalg.Dot8x4` (the shared query loaded
  once, reused across 8 vectors — the blocked-matmul a-reuse trick). Measured:
  **~7× faster Flat scan** (N=50k, d=128: 5.18 ms → 0.72 ms; the per-vector
  `linalg.Dot` swap is ~4.4× and the 8-vector streaming adds ~1.6× on top, now
  near memory bandwidth) and **~1.4× faster HNSW search** (d=64; build benefits
  via the same `sim`). **Precision:** the SIMD kernels accumulate in
  `float32` (vs the old `float64` scalar sum). For unit-norm `float32` inputs the
  per-element error is bounded — recall is unchanged (verified: 0 boundary flips,
  new-vs-`float64` top-k identical); only sub-ULP near-ties may order differently.
  HNSW is approximate by contract (accepted silently); `Flat` now documents the
  `float32`-precision scoring in its invariants.
- **`encoder` attention: vectorized the scores·V context step.** An end-to-end
  CPU profile of `Model.Encode` on real weights showed the per-head `ctx =
  scores · V` accumulation — a scalar triple-loop — was the single hottest line,
  ~⅓ of `Encode`, while the QKᵀ matmul (already SIMD) was ~2.6%. The context step
  now routes through the SIMD `matmulBTInto` (folding a per-head V transpose into
  the extract), in both `selfAttention` and `selfAttentionBatched`. Output is
  bit-exact (golden cosine 1.0, batch==single, `-race` clean). The gain is the L²
  term, so it scales with sequence length: **~2.85× single `Encode`** at ~500
  tokens, neutral (no regression) at short rerank passages.

### Added

- **`QueryFilter` on `ann.Flat`/`HNSW`/`FlatI8` — query-time logical delete**
  (Experimental surface). `QueryFilter(q, k, keep func(id int) bool)` returns only
  documents for which `keep` is true, so a live-set / tombstone applies WITHOUT
  mutating the index — keeping the immutability cornerstone (lock-free reads,
  snapshot consistency; now design rule 4 in `docs/architecture.md`). Flat and
  FlatI8 are exact; HNSW still routes the search through filtered nodes so graph
  connectivity (and live recall) holds — measured recall@10 = 1.00 under 20%
  deletion. Under heavy deletion, rebuild to purge tombstones. With the
  base+delta+fuse recipe (`Example_baseDeltaFusion`), these cover the
  changing-corpus cases without in-place mutation.
- **`embed.Truncate` — Matryoshka embedding truncation** (Experimental surface).
  Returns the first `dim` components of an embedding, L2-renormalized — a
  lower-dimensional embedding for MRL-trained models, composing with `ann.FlatI8`
  for a compounded memory cut (256→128 dims at int8 is 8× smaller than 256-d
  float32). Measured on the bundled Model2Vec slice (`TestMatryoshkaRecall`):
  recall@10 holds at **0.86 down to half the dimension (256→128)** and degrades
  only below, so half-dim truncation is free here. Input is not mutated; for
  non-MRL models truncation degrades the embedding (don't use it blindly).
- **`fuse.RSF` — Relative Score Fusion** (Experimental surface). A score-based
  alternative to the rank-based `RRF`: each ranking's raw scores are min-max
  normalized to [0,1] independently, then summed (`RSFWeighted` for a per-ranking
  tilt). Unlike RRF it preserves how *much* better one hit is than the next within
  a list — better when the per-list scores are calibrated (cosine sims, BM25 in
  one corpus); RRF stays the choice for incomparable/noisy scales. Adds `Scored`
  and the `Scores` projection helper (the score-aware counterpart of `Keys`).
- **`bm25.TokenizePlain` — general-text analyzer** (Experimental surface). A
  Unicode word tokenizer (lowercase, split on any non-letter/non-digit, no
  identifier splitting) alongside the code-tuned `Tokenize` — which over-fragments
  prose (`getUserName` → get/user/name/getusername) and breaks hyphenated/
  apostrophed words. `Build`/`Query` take pre-tokenized docs, so callers pick the
  analyzer per corpus; `Tokenize` stays the code-RAG default. Widens the audience
  to natural-language corpora.
- **`bench` package — reproducible recall + latency harness** (Experimental
  tooling). `bench.Run(corpus, queries, k, cfg)` measures, for Flat / HNSW /
  FlatI8: recall@k vs the exact Flat top-k, per-query latency percentiles
  (p50/p95/p99), build time, and index memory, rendered as a Markdown `Table`. It
  turns "parity-tested" into concrete numbers and doubles as a recall regression
  gate. Its first run surfaced — and then verified the fix for — a real recall
  problem: HNSW recall@10 on clustered real embeddings was ~0.68, which the old
  random/d=64 unit test (0.99) had masked. That drove the HNSW Algorithm-4 change
  above (recall → 1.00). FlatI8 measured 0.98–1.00 recall at ¼ the memory.
- **`ann.FlatI8` — int8-quantized dense index** (Experimental tier). The int8
  sibling of `Flat`: stores each L2-normalized vector as int8 codes + a per-vector
  scale (¼ the memory) and scores a query by int8×int8 dot through `linalg`'s W8A8
  kernel (dynamic query quantization, SIMD + parallel — W8A8 at M=1). Same
  `Hit`/`Query(q, k)` shape as `Flat`, so it's a swap-in and feeds `fuse.RRF`
  identically — the lever for embedded / RAM-constrained / `//go:embed`-the-index
  retrieval. Measured recall@10 vs exact float32 `Flat`: **1.00 on real Model2Vec
  embeddings, 0.986 on adversarial random unit vectors**, at **3.94× smaller**
  storage. Follow-ups: `FlatI8` persistence, int8 HNSW, and a binary/Hamming
  pre-filter.
- **`ann.HNSW` persistence — `MarshalBinary` + `Load`** (Experimental tier). The
  graph was rebuilt per process; now a built index serializes to a versioned byte
  blob (`MarshalBinary`, also `encoding.BinaryMarshaler`) and reloads query-ready
  via `Load([]byte)` — the `//go:embed`-an-index pattern: build the graph once
  offline, embed the bytes, load at startup. The format is versioned from day one
  (magic + version, rejects unknown versions) and `Load` validates graph integrity
  — out-of-range neighbor ids, layer-inconsistent edges, truncation — so a corrupt
  or hostile blob returns an error rather than panicking or OOM-ing (fuzzed). A
  round-trip reproduces identical `Query` results; `MarshalBinary` is deterministic.
- **`sparse` package — learned-sparse (SPLADE-style) retrieval** (Experimental
  tier). The third retrieval signal alongside dense (`ann`) and lexical (`bm25`):
  an inverted index over sparse document vectors scored by sparse dot product
  (`score(q,d) = Σ_t q_t·d_t`). `Hit.Index` matches `ann.Hit`, so a sparse ranking
  feeds the existing `fuse.RRF` flow identically. This is the inference-OPTIONAL
  half — `New`/`Query` operate on pre-computed `SparseVec` values (term id →
  weight) produced by any SPLADE-family model out of band; an in-process masked-LM
  expansion head (reusing `encoder`'s NomicBert machinery) is a planned follow-up.
  Pure Go, immutable-after-`New` (concurrent-`Query`-safe), validated against a
  brute-force sparse-dot reference.
- **amd64 AVX2 fused `MatmulBTW4A8` kernel** (`dot_w4a8_amd64.s`,
  `quant_w4a8_amd64.go`) — completes the v1.1.0 follow-up: the int4×int8 decode
  kernel now has an amd64 path, not just arm64. Same shape as the arm64 kernel —
  a nibble-unpack prologue feeding the proven `dotI8AVX2` sign-extend body
  (VPMOVSXBW+VPMADDWD+VPADDD), gated by `hasAVX2`; non-AVX2 amd64 keeps the scalar
  reference. Validated on Zen 2 (Ryzen 7 3700X): bit-exact vs the scalar oracle,
  race-clean, ~1.7–1.9× of W8A8 and ~32× faster than `MatmulBTQ4` at M=1 decode.
  A VNNI (`VPDPBUSD`) variant behind the same CPUID gate remains a follow-up.

### Security / Fixed

- **Hardened the GGUF and safetensors parsers against hostile inputs.** Both
  parse untrusted files, and several untrusted size fields drove allocations or
  slice bounds directly. Fixed (found by new fuzzers, `embed/*_fuzz_test.go`):
  - GGUF `tensorCount`/`kvCount`/`nGroups` and array/string lengths could be
    enormous or overflow `int` when narrowed, causing OOM (`make(map, ~5e10)`)
    or a slice-bounds panic. Untrusted counts are now bounded by the remaining
    input and `make()` hints are clamped; over-large lengths return an error.
  - safetensors header-length check `len(data) < 8+headerLen` overflowed for a
    `headerLen` near 2⁶⁴, passing the guard and panicking on the slice. Compared
    without the add now.

  All three parse entrypoints (`OpenGGUFBytes`/`parseGGUF`, `parseSafetensors`,
  `parseShardIndex`) now return an error rather than panic/OOM on any input. The
  OOM repro is committed as a regression seed; CI runs a short fuzz smoke.
- **Hardened the GGUF dequant path** (`Tensor`/`RowDequantizer`, `FuzzGGUFDequant`).
  The tensor directory's dims and data offset are untrusted; fuzzing found two
  more crashes:
  - `∏dims` (element count) overflowed `int` for hostile dims, wrapping the
    byte-size check and OOM-ing `make([]float32, n)`. The count is now computed
    with a check-before-multiply and bounded by the data section (no supported
    type packs fewer than ~0.5 bytes/element).
  - the tensor data-range check `offset + nbytes > len(data)` overflowed `uint64`
    for an `offset` near 2⁶⁴, passing the guard and panicking on the slice — same
    fix as safetensors (compare without adding).
  Both repros are committed as regression seeds; the dequant fuzzer is in the CI
  smoke set.

### Documentation

- `linalg` now has a package `doc.go` with the kernel-dispatch map (which kernel
  fires on which CPU, and why), and `Dot8x4` documents its large-K throughput
  cliff with the "tile K to ≤~768" guidance. README's model-fetch quick start no
  longer requires `ken` — it uses the Hugging Face CLI directly.

## [1.1.1] — 2026-06-08

### Added

- **amd64 AVX2 fused kernel for `linalg.MatmulBTW4A8`** — the follow-up promised in
  1.1.0. The int4×int8 *decode* (M=1) path now has a SIMD kernel on amd64, not just
  arm64: a nibble-unpack prologue (16 packed bytes → 32 centered int8 weights) feeds
  the proven `dotI8AVX2` body (`VPMOVSXBW` + `VPMADDWD` + `VPADDD`), gated by
  `hasAVX2` (non-AVX2 amd64 keeps the scalar reference). Validated on Zen 2 (Ryzen 7
  3700X): bit-for-bit vs the scalar oracle, race-clean; at M=1 decode ~1.7–1.9× of
  `MatmulBTW8A8` and ~32× faster than `MatmulBTQ4`, on par with the arm64 SDOT
  kernel. A VNNI (`VPDPBUSD`) variant for Zen 4+ / Cascade Lake+ remains a follow-up
  behind the same CPUID gate. No signatures changed.

## [1.1.0] — 2026-06-08

### Added

- **`linalg.MatmulBTW4A8` — int4-weight × int8-activation matmul**, the integer
  analogue of `MatmulBTW8A8` and the fast int4 *decode* (M=1) path that
  `MatmulBTQ4` structurally can't be. `MatmulBTQ4` (f32 activations) is
  dequant-bound at M=1 — profiling put ~72% of decode in the per-weight f32
  dequant, which the v1.0.1 column-outer reuse only amortizes at M>1. W4A8 stays
  in the integer domain: a fused arm64 NEON+SDOT kernel streams each weight row,
  unpacking int4 nibbles to int8 in-register (the only new asm — it reuses the
  proven `dot_i8dp` SDOT body) and emitting per-group int32 dots that Go folds
  with the f32 group scales. No per-weight f32 dequant, no per-group Go↔asm
  transition.

  Result on Apple M-series (group 32): W4A8 at M=1 is **~2.0–2.3× of
  `MatmulBTW8A8`** and **~23× faster than `MatmulBTQ4`** (e.g. the 1.5B MLP shape
  K=1536,N=8960: 19.2 ms → 0.80 ms) — int4 CPU decode goes from ~28× slower than
  int8 to ~2×, i.e. usable. Output matches the dequant-f32 reference within the
  W8A8 tolerance (relL2 ≈ 0.008 ≤ 5e-2); the fused kernel is bit-exact vs the
  scalar reference on the integer accumulation.

  arm64 (DotProd) ships the fused kernel; **amd64 and non-DotProd arm64 use the
  pure-Go scalar reference** for now (correct, not yet SIMD-fast) — the amd64
  AVX2/VNNI fused kernel is a follow-up to be validated on the Linux box.
  `MatmulBTQ4` is unchanged and remains the f32-activation / prefill path. No
  existing signatures changed.

## [1.0.1] — 2026-06-06

### Fixed

- **`linalg.MatmulBTQ4` int4 matmul performance** — was ~28× slower than the
  int8 path in goinfer's 1.5B decode; the "SIMD" kernel was even slower than its
  own scalar fallback. It did `K/group` tiny 32-wide `dotF32` calls per output,
  so per-call SIMD setup + horizontal-reduction overhead swamped the work. Now it
  dequantizes each weight row ONCE into a full K-wide f32 scratch (via
  `DequantizeRowInt4`) and runs a single vectorized `dotF32` over the whole row —
  mirroring `MatmulBTQ8` — and reuses that dequant across the M activation rows
  (column-outer). Q4 is now within ~1.8× of Q8 at M=1 and *faster* than Q8 at
  M=64 (it streams each weight once; Q8 re-widens per row). Output is
  bit-identical to the `DequantizeRowInt4`-then-`MatmulBT` reference (the parity
  test's oracle) — perf only, numerics unchanged. No API/signature change.

## [1.0.0] — 2026-06-06

First stable release. No functional change since 0.5.2 — 1.0 is the commitment
that the **Hard tier** (the retrieval core: `topk`, `ann.New`/`Flat.Query`/`Hit`,
`bm25`, `fuse`, `embed` core + `OpenSafetensors*`, `encoder.Load`/`Model`/
`Encode`/`Encoder`, `chunk`) now follows semver — no breaking changes before a
v2.0. The Hard tier was verified backward-compatible across the 0.4.x and 0.5.x
minors (`apidiff`, zero incompatible changes).

The **Experimental** tier (`linalg`, `encoder.Backend`, `ann.HNSW`,
`encoder.LoadQ8`/`ModelQ8`, the mmap loader variant, the concrete chunker
structs, `chunk/treesitter`) ships but is explicitly **excluded** from the 1.0
compatibility guarantee and may change in any release — see
[README.md](README.md#stability-tiers).

## [0.5.2] — 2026-06-05

### Changed

- **W8A8 matmul re-blocked column-outer** (`w8a8Span`, `w8a8BatchSpan`): each
  weight row is now loaded once and reused across the M activation rows, instead
  of re-streamed per row. At M>1 — speculative-decode verify (M=K), prefill, the
  encoder — this streams the (bandwidth-dominant) weight matrix once rather than
  M times. **M=1 single-token decode is unchanged** (one row either way), and the
  output of every element is the same `float32(dotI8(aq[i],bQ[j]))·scales`
  expression regardless of loop order, so it's **bit-identical for any M**
  (verified: M>1 output matches stacked per-row M=1 calls; `-race` green).
  Register-tiling the M loop (an int8 multi-row kernel) is a possible follow-up.

## [0.5.1] — 2026-06-05

### Added

- **`linalg.SetParallelWidth(n)` / `ParallelWidth()`** — cap the number of worker
  shards a parallel matmul fans out to (0 = GOMAXPROCS, the default). Orthogonal
  to `SetParallelThreshold` (whether to parallelize vs how many shards). Lets a
  consumer narrow the fan-out to ~the P-core count to avoid E-core stragglers at
  the fork/join barrier on heterogeneous CPUs. Numerically inert — parallel
  matmuls partition output columns, so any width is bit-identical (verified at
  widths 1–8). aikit's default stays GOMAXPROCS; the consumer that knows its
  workload + machine sets it.

## [0.5.0] — 2026-06-05

### Added

- **`linalg.MatmulBTW8A8Into(ws, …)`** — W8A8 matmul with a caller-supplied
  `*Workspace` for the quantized-activation scratch, so a steady-state decode
  loop allocates **zero** per matmul (the allocating `MatmulBTW8A8` stays, now a
  thin wrapper). It also quantizes each activation row **once** instead of once
  per parallel worker. Output is bit-identical to `MatmulBTW8A8`.
- **`linalg.MatmulBTW8A8Batch(ws, a, M, K, ops)`** + **`W8A8Op`** — run several
  W8A8 matmuls that share one activation (fused q/k/v or gate/up) in a single
  parallel region: one quantize + one goroutine fork/join instead of per-matmul.
  Weights are read in place, so a consumer that aliases int8 weights zero-copy
  gets the dispatch reduction with **no concat copy**. Bit-identical to calling
  `MatmulBTW8A8Into` per op.
- **`linalg.Workspace`** — reusable scratch buffers for the above (one per
  goroutine / decode stream; not safe for concurrent use).
- **`linalg.SetParallelThreshold` / `ParallelThreshold`** — process-wide knob
  for the MAC count at/above which matmuls parallelize, for end-to-end tuning.
- **`Workspace.SetWorkers(n)` / `Close()`** *(opt-in, experimental)* — give a
  Workspace a persistent pool of `n` worker goroutines that spin briefly before
  parking, so a decode stream's back-to-back matmuls reuse hot workers instead
  of spawning + parking per call (and the parallel path drops from ~per-dispatch
  allocs to ~zero). Single-dispatcher only (one per stream); `Close` stops the
  workers. The zero-value Workspace has no pool — the default and the encoder's
  concurrent-forward path are unchanged. The win is workload-dependent (a warm
  microbenchmark can't show it); enable it and measure end-to-end.

### Changed

- **Matmul parallel threshold raised** to 16.78M MACs (was 32768) so M=1
  single-token decode projections run **serially** — that regime spent most of
  its CPU in goroutine park/wake for no speedup. Prompt/prefill and the encoder
  (large M) still parallelize (a ~3× win there is unchanged). No numeric change;
  purely *when* the fork/join happens.

## [0.4.2] — 2026-06-04

### Added

- **`embed.OpenGGUFBytes(raw)`** — parse a GGUF model from an in-memory byte
  slice (aliased, not copied), no filesystem touch. For `//go:embed`-ed or
  downloaded-in-memory models and read-only environments with no writable temp
  dir. `Close` is a no-op for the bytes-backed file.

## [0.4.1] — 2026-06-04

### Fixed

- **Windows build.** `embed` referenced `syscall.Mmap`/`Munmap`/`PROT_READ`/
  `MAP_PRIVATE` unconditionally, so the whole module failed to compile on
  `GOOS=windows` (and any non-unix target). The mmap implementation is now
  build-tagged: real memory-mapping on unix (`embed/mmap_unix.go`), and a
  heap-read fallback elsewhere (`embed/mmap_other.go`) with identical API and
  results. `OpenSafetensorsMmap` / `OpenGGUFMmap` behave the same; on Windows the
  bytes live in the Go heap instead of the OS page cache. No new dependencies.
- CI now builds + tests on `windows-latest` alongside Linux.

## [0.4.0] — 2026-06-04

### Changed (breaking, pre-1.0)

- **Split the LLM runtime out to [`goinfer`](https://github.com/townsendmerino/goinfer).**
  `decoder`, `tokenizer`, `constrain`, and the `demo/` generation CLI moved to the
  new `goinfer` module (which depends inward on aikit). aikit is now a focused,
  cgo-free retrieval toolkit; goinfer carries the generation stack and the cgo
  WebGPU backend.
- **`internal/linalg` → public `linalg`.** The SIMD dot/matmul + int8/int4 quant
  kernels are now an importable package (shared across the repo boundary).
- **`encoder` gained a pluggable `Backend`** (`RegisterBackend`/`NewBackend`) so
  GPU acceleration is provided by the opt-in `goinfer/gpu` module under `-tags gpu`
  — `encoder` itself carries no `webgpu` (cgo) dependency.
- **`chunk/treesitter` is now its own module** (`…/aikit/chunk/treesitter`,
  versioned `chunk/treesitter/vX.Y.Z`), quarantining the `gotreesitter` dependency
  so the core graph has no cgo.
- The root module's only dependency beyond stdlib is `golang.org/x/text`; a CI
  guard fails the build if `webgpu`/`gotreesitter` ever leak into the core graph.

## [0.3.0] — 2026-06-03

### Changed

- **Parallel weight loading** — the per-layer tensor dequant + re-quant (the bulk
  of load time, and independent per layer over the read-only mmap) now fans out
  across cores (`parallelLayers`, GOMAXPROCS workers), for both the GGUF and
  safetensors paths. The Mellum2-12B Q4_K_M GGUF load dropped from **~2 min to
  ~20 s** (`--quant int4`); race-clean. Output is unchanged (deterministic
  per-tensor work).
- **Streaming GGUF dequant → resident quant (no full-f32 round-trip).** The GGUF
  loader used to dequantize each tensor into a whole `[rows·cols]` f32 buffer and
  then re-quantize it; for a 12B model the largest tensors are hundreds of MB that
  stream to DRAM and back per tensor. Now each tensor is dequantized **row-by-row
  into a one-row scratch** and quantized straight into the resident int8/int4
  arrays (`embed.GGUFFile.RowDequantizer` drives `decoder.streamQuantized`), so the
  f32 intermediate stays in cache and the full-tensor allocation is gone. The RoPE
  q/k permutation — being a pure row reorder — is folded into the dequant order
  (rows pulled in HF order) instead of permuting a separate f32 buffer. Bit-
  identical to the old path by construction (the per-row primitives are the same
  ones `QuantizeRowsInt8`/`QuantizeGroupsInt4` use): every GGUF forward-parity test
  holds its exact prior cosine — Q8_0 0.99996, Q4_K_M 0.9975, int4-resident 0.9946,
  Mellum-12B runs — across Q8_0/Q4_0/Q4_K/Q6_K × f32/int8/int4 × plain/permuted/MoE
  tensors (`TestDequantRange_streamMatchesWhole` + the GGUF parity suite).
- **Quantized matmuls are now SIMD** — `linalg.MatmulBTQ4` and `MatmulBTQ8` widen
  each weight group/row into a reused scratch buffer and run the AVX2/NEON
  `dotF32` kernel over it (applying the scale at write-back), instead of a scalar
  multiply-accumulate loop. On a decode-step shape (M=1, K=N=2048): int4 **~6.7×**
  (8.3 → 1.2 ms), int8 **~6.9×** (3.0 → 0.43 ms). Outputs unchanged within float
  reassociation (`TestMatmulBTQ4_matchesDequant` relL2 ≤ 1e-5); decoder quant
  accuracy identical. (An int8×int8→int32 fixed-point kernel could go further.)
- **`embed.OpenGGUFMmap`** — memory-map a `.gguf` (read-only, MAP_PRIVATE)
  instead of `os.ReadFile`-ing it onto the heap, so the raw quantized bytes live
  in reclaimable page cache. `decoder` and `tokenizer` GGUF loads now use it:
  the decoder dequantizes tensor-by-tensor off the mapping then `Close`s it
  (weights are fresh copies, so nothing dangles), and `tokenizer.LoadGGUF` no
  longer pages in the multi-GB weights at all to read head-of-file metadata (its
  GGUF test dropped from ~0.5 s to ~0.03 s). Parse is bit-identical to the heap
  path (`TestGGUFMmap_matchesHeap`). Combined with streaming int8 below, a big
  quantized `.gguf` no longer needs the whole file on the heap *plus* the model
  in f32 to load. Unix only (`syscall.Mmap`), like `OpenSafetensorsMmap`;
  `OpenGGUF` (heap) remains for other platforms.
- **Streaming int8 quantization at load** — `decoder.Load(…, Quant: "int8")` now
  quantizes each matmul tensor to per-row int8 the moment it is read and frees
  its f32 before the next tensor loads, instead of materializing the whole model
  in f32 and quantizing afterward. The transient footprint drops from the whole
  model in f32 to the int8 model + one tensor's f32 — so a big quantized
  checkpoint loads in roughly a quarter of the RAM. Covers the safetensors,
  GPT-2, and GGUF paths; a quantized `.gguf` lands resident as int8 (the demo
  chats from a bare `.gguf` under `--quant int8`). Forward output is unchanged
  (it quantizes the same weights, just earlier); validated by the new
  `TestGGUF_int8_resident` (argmax + 0.9998 cosine vs the f32 oracle) and the
  unchanged `TestQuantInt8_accuracy`. Public `LoadWeights`/`LoadWeightsFromFS`
  signatures are unchanged.

### Added

- **GGUF IQ2_S + IQ3_S dequant (grid-codebook quants).** The two grid-codebook IQ
  types: each block packs grid indices + packed sign bits + 4-bit sub-scales, and
  the kernel looks up an 8-wide (IQ2_S) or 4-wide (IQ3_S) int8 pattern from a large
  codebook, applies the per-element sign, and scales (`dequantIQ2SBlock` /
  `dequantIQ3SBlock`). The grids (IQ2_S 1024×8, IQ3_S 512×4) are generated
  byte-exact from llama.cpp's `gguf` reference into `embed/iq_grids.go`
  (`scripts/gen_iq_grids.py`), not hand-transcribed. Pinned **bit-exact (Δ=0) vs
  the `gguf` reference** (`TestIQDequant_matchesReference`). Remaining unimplemented:
  IQ1_*/IQ2_XXS/IQ2_XS/IQ3_XXS (rarer extreme-low-bit grid quants).
- **GGUF IQ4_NL + IQ4_XS dequant (codebook quants).** The two tractable IQ types
  — both built on a shared 16-entry non-linear codebook (`kvaluesIQ4NL`) rather
  than the grid lookups of the IQ2*/IQ3* family. `dequantIQ4NLBlock` is a 32-block
  (a nibble per element indexing the codebook, scaled by the f16 block scale);
  `dequantIQ4XSBlock` is a 256-superblock of eight 32-sub-blocks, each with a
  6-bit scale assembled from `scales_l`/`scales_h` (recentered by −32) times the
  super f16 scale. Parity-gated **bit-exact (Δ=0) against llama.cpp's `gguf`
  Python reference** over deterministic blocks — codebook quants have no
  convenient small-model f32 oracle, so the kernel is pinned directly, every value
  (`TestIQDequant_matchesReference`; golden via `scripts/pin_iq_dequant.py`). The
  grid-codebook IQ2*/IQ3* types remain unimplemented.
- **GGUF Q2_K + Q3_K + Q5_K dequant.** Three more K-quant block types on the
  existing GGUF seam, so `Q2_K` / `Q3_K_M` / `Q5_K_M` files (and any mix using
  them) load: `embed` gained `dequantQ5KBlock` (the Q4_K scale/min packing plus a
  5th bit per element from `qh`), `dequantQ3KBlock` (the 6-bit-scale aux unpack +
  the `hmask` 3rd bit lifting each 2-bit code to [−4,3]), and `dequantQ2KBlock`
  (4-bit scale+min per sub-block, 2-bit quants — the coarsest, no high-bit mask).
  Validated against the committed f32 llama oracle on real TinyLlama mixes —
  Q5_K_M **cosine 0.9991**, Q3_K_M **0.9925**, Q2_K **0.9832** (argmax preserved
  throughout), slotting in order between Q4_K_M (0.9975) and Q8_0 (0.99996) as
  expected (`TestGGUF_Q5_K_M_parity` / `TestGGUF_Q3_K_M_parity` /
  `TestGGUF_Q2_K_parity`). The supported K-quants are now Q2_K/Q3_K/Q4_K/Q5_K/Q6_K
  (only the codebook IQ* types remain unimplemented).
- **Shared-expert MoE (Qwen-MoE / `qwen2_moe`).** A new architecture: qwen2's
  attention (q/k/v bias, no QK-norm) with the FFN replaced on every layer by a
  sparse router + top-k experts **plus an always-on shared expert** — a gated
  SwiGLU MLP at `shared_expert_intermediate_size`, scaled by
  `sigmoid(shared_gate·h)` and added to the routed sum (HF Qwen2MoeSparseMoeBlock).
  Adds `MoEConfig.SharedIntermediateDim`, the `SharedExpert`/`SharedGate` weights,
  and the `qwen2_moe` descriptor + tensor schema (`mlp.shared_expert.*` /
  `mlp.shared_expert_gate`). Validated structurally against HF on a tiny random
  Qwen1.5-MoE checkpoint — argmax + every sampled logit match, **cosine ~1.0**
  (`TestQwen2Moe_forwardParity`). Unlocks Qwen1.5-MoE-A2.7B / Qwen2-57B-A14B.
- **Gemma 3 GGUF architecture.** The most involved GGUF arch: `ggufConfig`
  dispatches `gemma3`/`gemma3_text`, and the loader maps the gemma3.* metadata onto
  the existing descriptor — sandwich norms (the new `post_attention_norm` /
  `post_ffw_norm` loads), QK-norm, GeGLU, the √hidden embed scale, the 5:1
  sliding/global pattern with dual RoPE bases, and the tied head. Two gemma-specific
  GGUF quirks handled: it's NEOX-rope (no q/k permute), and llama.cpp **bakes
  Gemma's (1+w) norm offset into the stored `*norm.weight`** — which the package's
  `RMSAddOne` forward would double — so every gemma norm has the 1 subtracted back
  out at load (`vnorm`, gated on `RMSAddOne`; no-op for the other archs). A bare
  gemma-3-270m Q8_0 GGUF runs end-to-end vs the f32 oracle — argmax matches, cosine
  **0.9998** (`TestGGUF_gemma3_parity`).
- **Qwen3 GGUF architecture.** `ggufConfig` dispatches `qwen3`: versus qwen2 it
  drops the q/k/v bias and adds **QK-norm** (per-head RMSNorm over an explicit
  `head_dim`, before RoPE). The loader already had the QK-norm load, tied-LM-head,
  and NEOX no-permute paths, so this is just the `qwen3.*` metadata mapping. A bare
  Qwen3-1.7B Q8_0 GGUF runs end-to-end vs the f32 oracle — argmax matches, cosine
  **0.9998** (`TestGGUF_qwen3_parity`).
- **Qwen2 GGUF architecture.** `ggufConfig` now dispatches `qwen2` (Qwen2/Qwen2.5)
  in addition to `llama` and `mellum`: the `qwen2.*` metadata maps onto the same
  descriptor, and the GGUF weight builder loads the q/k/v projection **biases**
  (the one thing qwen2 adds over llama). A subtlety the new path gets right: the
  q/k weight (and bias) permutation is gated on the rope type — llama.cpp permutes
  only NORM-rope archs (llama, mellum), while qwen2 is NEOX-rope and stays in HF
  order (`ggufQKPermuted`), so a wrong unconditional un-permute is avoided. A bare
  Qwen2.5-0.5B Q8_0 GGUF runs end-to-end: argmax matches the f32 oracle, cosine
  ~0.997 (`TestGGUF_qwen2_parity`, skip-when-absent). Unknown archs default to
  NEOX (no permute), the common modern case.
- **Exact Mellum2 byte-level tokenizer parity.** Mellum2's pre_tokenizer is
  `Sequence[Digits{individual_digits}, ByteLevel]` (no normalizer) — the `Digits`
  stage isolates each digit *before* the GPT-2 split, so a leading space never
  attaches to a digit (`" 1"` → `Ġ` + `1`, not the single `Ġ1`). The byte-level
  pipeline now reproduces this: a `splitDigits` knob (detected from a
  `Digits{individual_digits}` node in `tokenizer.json`, and from
  `tokenizer.ggml.pre == "mellum2"` on the GGUF path) pre-segments each gap so the
  GPT-2 regex sees digits in isolation. Validated byte-exact against an HF
  `tokenizers` oracle (`mellum2_tokenizer_golden.json`, 20 code-heavy prompts) on
  both the `tokenizer.json` and bare-GGUF paths (`TestByteLevel_mellum2GoldenParity`,
  `TestLoadGGUF_mellum2DigitParity`). Other byte-level families are unchanged
  (`splitDigits` defaults off).
- **GPTQ + AWQ (safetensors-resident int4).** The decoder loads HF int4
  checkpoints — where each linear ships as packed int4 (`qweight`/`qzeros`/
  `scales` ± `g_idx`) instead of an f32 `.weight` — detected from `config.json`'s
  `quantization_config` (`quant_method: gptq | awq`, 4-bit). `gptqReconstruct`
  un-packs the AutoGPTQ layout (`[in/8,out]`, `w = (code-(zero+1))·scale`, group
  via `g_idx` so **act-order** works); `awqReconstruct` un-packs the AutoAWQ GEMM
  layout (`[in,out/8]`, packed along the OUTPUT dim, with the `[0,4,1,5,2,6,3,7]`
  nibble de-interleave and a no-`+1` zero-point). Both transpose to `[out,in]`
  and stream through the same int8/int4 re-quant path, so a GPTQ/AWQ model can
  also run resident-int4. Embeddings/norms/LM head stay bf16/f16. Validated
  against the committed f32 oracle for the *same* model (TheBloke/TinyLlama-1.1B
  -Chat-v1.0-{GPTQ,AWQ}, 4-bit g128): argmax preserved, **cosine 0.991 (GPTQ) /
  0.996 (AWQ)** vs f32 (`TestGPTQ_parity` / `TestAWQ_parity`, skip-when-absent).
  Adds `embed.Tensor.Int32s`.
- **Mellum2 — runs end-to-end from a bare GGUF.** The decoder runs JetBrains
  Mellum2 (`model_type: "mellum"`, a 12B-A2.5B MoE code model): the `mellum`
  adapter combines axes we already had — a sparse MoE on every layer (64 experts,
  top-8, with the narrower `moe_intermediate_size` expert FFN), a 3:1 sliding/full
  attention interleave (`layer_types`), and **QK-norm** — plus the one new piece,
  **YaRN** RoPE. YaRN is HF-exact (`_compute_yarn_parameters`: the NTK-by-parts
  inv-freq blend + the `attention_factor` mscale), validated against a pinned
  reference (`TestYarn_matchesHF`, rel ≤ 1e-12), slotting into the dual-table RoPE
  via a new per-attention-type scaling path (`ropeScalingLocal`) and the nested
  `rope_parameters` config (YaRN on full layers, plain RoPE on sliding layers).
  Also usable for any long-context Qwen/Llama with `rope_scaling: {"rope_type":
  "yarn"}`.

  The **GGUF path** loads it with no sidecar: `ggufConfig` dispatches on
  `general.architecture`, building the Mellum descriptor (incl. YaRN + the
  sliding/full pattern) from `mellum.*` metadata; `buildWeightsFromGGUF` handles
  the **stacked** expert tensors (`ffn_{gate,up,down}_exps` sliced per expert),
  the QK-norm tensors (un-permuted to match the q/k RoPE permute), and the new
  **Q5_0** dequant the Q4_K_M mix uses. Verified end-to-end: a real
  Mellum2-12B Q4_K_M GGUF generates coherent Python under `--quant int4` in pure
  Go (`TestMellumGGUF_runs`, skip-when-absent). Also fixes the safetensors mellum
  path, which was missing the QK-norm tensors.
- **GGUF Q5_0 dequant** (`embed`) — the legacy 5-bit block type (some Q4_K_M
  mixes use it), with an exact unit test.
- **`constrain` package — constrained / structured decoding.** A logit mask that
  forces a model's output to satisfy a grammar: at each step every vocab token
  whose bytes would break the grammar is set to −∞, and EOS is masked until the
  output is a complete document. Ships a streaming **JSON** grammar (byte-level
  pushdown automaton, RFC 8259) — so a small model *physically cannot* emit
  malformed JSON. It plugs into the new `decoder.SamplingParams.LogitProcessor`
  hook (`constrain.Masker.Process` matches the signature) and is stdlib-only (the
  vocab→bytes map is injected as a func, e.g. `tokenizer.TokenText`). The guarantee
  is proven structurally: a hard-invariant test drives the masker with *random*
  logits over a synthetic vocab and confirms the output is always valid per
  `encoding/json` (`TestConstrainedDecode_alwaysValidJSON`). `demo/gemma --json`
  shows it end-to-end (a 1B model emits a valid JSON object). `StopWhenComplete`
  ends generation at the first complete document.
- **`decoder.SamplingParams.LogitProcessor`** — an optional per-step hook,
  `func(generated []int, logits []float32)`, called after the forward pass and
  before sampling so a caller can mask/bias logits (the seam for constrained
  decoding; can also gate EOS).
- **`tokenizer.Tokenizer.TokenText(id) []byte`** — the raw surface bytes a single
  token contributes (no whole-sequence post-processing), for mapping a vocabulary
  onto a byte-level grammar.
- **int8×int8 (W8A8) quantization** (`decoder.Load(…, Quant: "int8int8")`) — in
  addition to the weight-only int8, this quantizes the activations to int8 on the
  fly (dynamic per-row scale) and runs a true integer matmul: `linalg.dotI8`
  accumulates int8×int8→int32, with hand-written SIMD kernels — AVX2 on amd64
  (`dotI8AVX2`: VPMOVSXBW → VPMADDWD → VPADDD) and **NEON on arm64** (`dotI8NEON`:
  SMULL/SMULL2 → SADALP, base ARMv8, validated bit-exact under qemu-aarch64) — and
  a scalar fallback elsewhere. **~3.4×** faster than the f32-widen weight-only int8 on a
  decode-step shape (428 → 125 µs, K=N=2048). It is lossier (activations are also
  quantized): gemma cosine 0.9979 vs 0.9996, argmax preserved
  (`TestQuantInt8I8_accuracy`) — so it is opt-in; plain `int8` stays weight-only
  (f32 activations) for the higher accuracy.
- **ARMv8.2 DotProd (SDOT) int8 kernel.** On arm64 cores with the DotProd
  extension (Apple Silicon, Graviton2+, Neoverse, recent Cortex-A), `dotI8` now
  uses an `SDOT`-based kernel (`dotI8SDOT`) — one instruction folds 16 int8 pairs
  straight into a 4-lane int32 accumulator, replacing the base kernel's four
  (`SMULL`+`SMULL2`+`SADALP`+`SADALP`); four accumulators hide its latency.
  Selected at init by **runtime feature detection** with no new dependency:
  `detectDotProd` reads `HWCAP_ASIMDDP` from `/proc/self/auxv` on linux (true on
  Apple Silicon for darwin), falling back to the base `SMULL/SADALP` kernel where
  absent. Both kernels are bit-exact to the scalar reference, validated under
  qemu-aarch64 across `-cpu max` (DotProd → SDOT) and `-cpu cortex-a72` (no
  DotProd → base) — `TestDotI8SDOT_matchesScalar` / `TestDotI8_matchesScalar`.
- **Byte-level GGUF tokenizer** — `tokenizer.LoadGGUF` now also handles the
  byte-level family (`tokenizer.ggml.model == "gpt2"`: Llama-3 / Qwen / GPT-2),
  not just SPM/llama. It dispatches "gpt2" to the existing `modeByteLevel`
  pipeline and reads the pretokenizer knobs (digit-run cap, NFC, ignore_merges)
  from `tokenizer.ggml.pre` — the GGUF analogue of reading them from
  tokenizer.json. So a bare byte-level `.gguf` (the common modern instruct quant)
  now chats end-to-end. Parity-gated against a real Llama-3.2-1B-Instruct GGUF:
  `LoadGGUF` matches `Load` on the same model's tokenizer.json id-for-id
  (`TestLoadGGUF_byteLevelMatchesJSON`), and that json path is HF-golden-validated
  for the family.
- **int4 weight quantization** (`decoder.Load(…, Quant: "int4")`) — group-wise
  symmetric 4-bit on the projections (group size 32: a per-group f32 scale, two
  nibbles per byte; `linalg.QuantizeGroupsInt4` + a dequant-per-tile
  `MatmulBTQ4`), ~⅛ f32 on those weights. The token embedding **and** LM head
  stay int8 (they are the tied head — 4-bit there flips the argmax), mirroring
  how GGUF Q4_K_M keeps `token_embd`/`output` at Q6_K. Streams at load and works
  for safetensors, GPT-2, and GGUF (the demo chats from a bare `.gguf` under
  `--quant int4`). Validated on TinyLlama 1.1B: argmax preserved, cosine 0.994
  vs f32 (on par with Q4_K_M's own 0.9975). int4 is a big-model tool — on a 270M
  it is lossy enough to move the top token, so its strict gate runs on TinyLlama.
- **`tokenizer.LoadGGUF`** — build a `Tokenizer` from a bare `.gguf` file's
  embedded metadata (vocab + merges + special-token ids), no `tokenizer.json`
  needed. Covers the SentencePiece byte-fallback family
  (`tokenizer.ggml.model == "llama"`: Llama-2/Mistral/TinyLlama), reusing the
  `modeGemma` merge-rank core plus a `▁` dummy-prefix knob (prepend on encode,
  strip one leading space on decode). Parity-gated against HF `tokenizers` on
  TinyLlama (`testdata/tinyllama_tokenizer_golden.json`, pinned by
  `scripts/pin_tinyllama_tokenizer.py`). A bare `.gguf` now chats end-to-end —
  `demo/gemma` detects a `.gguf` path and tokenizes from it.
- `tokenizer.Load` now honors a SentencePiece `Prepend "▁"` normalizer (and the
  paired leading-space strip on decode), so non-Gemma SPM `tokenizer.json`
  files tokenize correctly; Gemma (no Prepend) is unchanged.

## [0.2.0] — 2026-06-03

Generative half of the toolkit lands. Two new public packages — `decoder` and
`tokenizer` — turn aikit from "embed + retrieve" into "embed + retrieve +
generate", in pure Go with no cgo, validated to HuggingFace parity across a
broad slice of the open-weights ecosystem.

### Added

- **`decoder` package** — autoregressive decoder-only LLM inference as a single
  generic forward pass parameterized by an `Architecture` descriptor resolved
  from the checkpoint. Validated to logit/argmax parity against HuggingFace for:
  - **Families:** Gemma 3, Qwen3, Qwen2.5, Llama-2/3, Mistral, GPT-2, and
    Mixtral (sparse-MoE).
  - **Axes:** RMSNorm/LayerNorm · RoPE (incl. llama3 frequency scaling)/learned
    positions · gated/non-gated/sparse-MoE MLP · full/sliding-window attention ·
    tied/untied heads · optional QKV/output bias · Linear/Conv1D layouts.
  - Public surface: `Load`, `LoadWeights`/`LoadWeightsFromFS`, `Model.Generate`
    (streaming), `Sampler` (temperature/top-k/top-p), `KVCache`, the `Backend`
    seam (`NewBackend`), and the `Config`/`Architecture` descriptors.
- **`tokenizer` package** — the BPE tokenizers the decoder LLMs ship, loaded
  from `tokenizer.json` with HF-exact id parity as the gate:
  - Gemma byte-fallback SentencePiece-style BPE (`▁` space normalize,
    `<0xNN>` fallback).
  - GPT-2 / Llama-3 / Qwen byte-level BPE (NFC normalize, GPT-2 split-regex
    pretokenizer, byte→printable-rune map).
  - Family auto-detected from `tokenizer.json`; special tokens resolved from
    `tokenizer_config.json`. Public surface: `Load`, `Tokenizer`,
    `SpecialTokens`, `ChatStyle`.
- **GGUF support** — self-describing quantized checkpoints (`embed/gguf.go`,
  `decoder/gguf.go`): GGUF v2/v3 container parse + block dequant for F32, F16,
  Q8_0, Q4_0, Q4_K, Q6_K. A bare `.gguf` loads with no sidecar config or
  safetensors. The llama.cpp interleaved q/k RoPE layout is un-permuted back to
  HF `rotate_half`. Validated vs the f32 oracle on TinyLlama: Q8_0 cosine
  0.99996, Q4_0 0.9944, **Q4_K_M 0.9975** (the most-downloaded laptop quant).
- **int8 weight quantization** for the decoder (`--quant int8`).
- **WebGPU backend** for the decoder — resident weights behind the `Backend`
  seam, swappable without touching the forward pass.
- **`internal/linalg`** — shared SIMD matmul/dot kernels (AVX2/FMA on amd64,
  NEON on arm64) and int8 quant helpers, factored out of `encoder` so both
  `encoder` and `decoder` share one accelerated path.
- **`encoder` acceleration** — SIMD/parallel/GPU matmul, plus `ann` HNSW
  approximate index and `fuse` RRF fusion shipped alongside.
- **`demo/gemma` and `demo/gemma-web`** — CLI and stdlib `net/http` + SSE web
  chat front-ends for the decoder.
- **`chunk/treesitter`** — Dart added to the tree-sitter language mapping.

### Changed

- `encoder`'s SIMD dot/matmul kernels moved to `internal/linalg`
  (`dot_arm64.s`, `dot_test.go`); no public-API change for `encoder` consumers.
- Bumped `github.com/odvcencio/gotreesitter` to `v0.20.0-rc3`.
- Applied Go 1.26 modernizers (`go fix ./...`).

## [0.1.1] — 2026-06-02

### Added

- `bm25.Index.IDF(term)` and `bm25.Index.DF(term)` — public read-only accessors
  mirroring the internal `idf` used by query scoring (IDF for ranking, raw DF
  for frequency filtering). Pure additive; no behavior change.

## [0.1.0] — 2026-05-30

### Added

- Initial release, extracted from [`ken`](https://github.com/townsendmerino/ken)
  per ken's ADR-034. Eight packages: `topk`, `ann`, `bm25`, `embed`, `encoder`,
  `chunk` (+ `regex`/`markdown`/`treesitter`).
- Numerical contracts: `embed` golden cosine 1.000000 vs Model2Vec; `encoder`
  golden cosine 1.000000 vs PyTorch+MPS CodeRankEmbed. See
  [README.md](README.md) for stability tiers.

[Unreleased]: https://github.com/townsendmerino/aikit/compare/v1.38.0...HEAD
[1.39.1]: https://github.com/townsendmerino/aikit/compare/v1.39.0...v1.39.1
[1.39.0]: https://github.com/townsendmerino/aikit/compare/v1.38.0...v1.39.0
[1.38.0]: https://github.com/townsendmerino/aikit/compare/v1.37.0...v1.38.0
[1.37.0]: https://github.com/townsendmerino/aikit/compare/v1.36.0...v1.37.0
[1.36.0]: https://github.com/townsendmerino/aikit/compare/v1.35.0...v1.36.0
[1.35.0]: https://github.com/townsendmerino/aikit/compare/v1.34.0...v1.35.0
[1.34.0]: https://github.com/townsendmerino/aikit/compare/v1.33.0...v1.34.0
[1.33.0]: https://github.com/townsendmerino/aikit/compare/v1.32.0...v1.33.0
[1.32.0]: https://github.com/townsendmerino/aikit/compare/v1.31.0...v1.32.0
[1.31.0]: https://github.com/townsendmerino/aikit/compare/v1.30.0...v1.31.0
[1.30.0]: https://github.com/townsendmerino/aikit/compare/v1.29.0...v1.30.0
[1.29.0]: https://github.com/townsendmerino/aikit/compare/v1.28.0...v1.29.0
[1.28.0]: https://github.com/townsendmerino/aikit/compare/v1.27.0...v1.28.0
[1.27.0]: https://github.com/townsendmerino/aikit/compare/v1.26.0...v1.27.0
[1.26.0]: https://github.com/townsendmerino/aikit/compare/v1.25.0...v1.26.0
[1.25.0]: https://github.com/townsendmerino/aikit/compare/v1.24.0...v1.25.0
[1.24.0]: https://github.com/townsendmerino/aikit/compare/v1.23.0...v1.24.0
[1.23.0]: https://github.com/townsendmerino/aikit/compare/v1.22.0...v1.23.0
[1.22.0]: https://github.com/townsendmerino/aikit/compare/v1.21.0...v1.22.0
[1.21.0]: https://github.com/townsendmerino/aikit/compare/v1.20.0...v1.21.0
[1.19.0]: https://github.com/townsendmerino/aikit/compare/v1.18.0...v1.19.0
[1.18.0]: https://github.com/townsendmerino/aikit/compare/v1.17.1...v1.18.0
[1.17.1]: https://github.com/townsendmerino/aikit/compare/v1.17.0...v1.17.1
[1.17.0]: https://github.com/townsendmerino/aikit/compare/v1.16.0...v1.17.0
[1.16.0]: https://github.com/townsendmerino/aikit/compare/v1.15.0...v1.16.0
[1.15.0]: https://github.com/townsendmerino/aikit/compare/v1.14.0...v1.15.0
[1.14.0]: https://github.com/townsendmerino/aikit/compare/v1.13.0...v1.14.0
[1.13.0]: https://github.com/townsendmerino/aikit/compare/v1.12.0...v1.13.0
[1.12.0]: https://github.com/townsendmerino/aikit/compare/v1.11.0...v1.12.0
[1.11.0]: https://github.com/townsendmerino/aikit/compare/v1.10.1...v1.11.0
[1.10.1]: https://github.com/townsendmerino/aikit/compare/v1.10.0...v1.10.1
[1.10.0]: https://github.com/townsendmerino/aikit/compare/v1.9.0...v1.10.0
[1.9.0]: https://github.com/townsendmerino/aikit/compare/v1.8.1...v1.9.0
[1.8.1]: https://github.com/townsendmerino/aikit/compare/v1.8.0...v1.8.1
[1.8.0]: https://github.com/townsendmerino/aikit/compare/v1.7.3...v1.8.0
[1.7.3]: https://github.com/townsendmerino/aikit/compare/v1.7.2...v1.7.3
[1.7.2]: https://github.com/townsendmerino/aikit/compare/v1.7.1...v1.7.2
[1.7.1]: https://github.com/townsendmerino/aikit/compare/v1.7.0...v1.7.1
[1.7.0]: https://github.com/townsendmerino/aikit/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/townsendmerino/aikit/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/townsendmerino/aikit/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/townsendmerino/aikit/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/townsendmerino/aikit/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/townsendmerino/aikit/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/townsendmerino/aikit/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/townsendmerino/aikit/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/townsendmerino/aikit/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/townsendmerino/aikit/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/townsendmerino/aikit/compare/v0.5.2...v1.0.0
[0.5.2]: https://github.com/townsendmerino/aikit/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/townsendmerino/aikit/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/townsendmerino/aikit/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/townsendmerino/aikit/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/townsendmerino/aikit/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/townsendmerino/aikit/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/townsendmerino/aikit/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/townsendmerino/aikit/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/townsendmerino/aikit/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/townsendmerino/aikit/releases/tag/v0.1.0
