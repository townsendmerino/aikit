# Benchmarking the native-GPU substrate (CUDA + Metal)

**Status: methodology — lands before the numbers, so it can't be retrofitted to
flatter a conclusion** (same discipline as `benchmarks/README.md` and goinfer's
`PERF.md`). Companion to [`task-native-gpu.md`](task-native-gpu.md): that doc is
the *plan*; this is how the plan gets *measured*.

## BLUF

Adding native GPU (aikit/gpu: `Device/Buffer/Pipeline` + quantized GEMV, with
`anncuda`/`annmetal` backends on the `ann.Backend` seam) changes what a benchmark
*means*. A single "GB/s on an M-series" number — fine for one CPU SIMD kernel — is
meaningless once three backends (CPU-SIMD, Metal, CUDA) each win in a different
region of (size × batch × precision), on machines that can't co-reside. The
headline output is no longer a number; it's a **crossover curve** per backend, and
a **dispatch threshold** derived from it.

## Scope — both repos, one substrate, two regimes

aikit/gpu is a *shared* substrate: aikit's ANN/ViT/encode build on it, and
goinfer's `cuda/`+`metal/` re-point onto it (the GPU analogue of the existing
`linalg` relationship). Benchmarking therefore splits cleanly:

| Layer | Shared? | Harness |
|---|---|---|
| Device substrate + quantized GEMV kernels (`aikit/gpu`) | **Shared** aikit↔goinfer | **One** harness, same records, same parity gate |
| End-to-end workload | **Per-repo, per-regime** | aikit = batch; goinfer = decode |

The end-to-end split is load-bearing and must never be blurred in a report:

| Repo | Workload | Regime | GPU pays | Baseline | Headline metric |
|---|---|---|---|---|---|
| **aikit** | ANN corpus GEMV, ViT, batch encode | **compute-bound (batch)** | **3–9×** | CPU SIMD, same box | throughput + recall@k vs exact-CPU |
| **goinfer** | single-stream decode | **bandwidth-bound** | parity→modest | CPU decode + Ollama-CUDA | tok/s + argmax/logit parity |

A "3× speedup" from the aikit batch regime and a "1.4× vs Ollama" from goinfer
decode are **different measurements of different things** — report them in their
own tables with their own baselines. The one number that *is* comparable across
repos is the substrate kernel (a W8A8 GEMV of a given shape), because both call
the same `aikit/gpu` code.

## The six methodology shifts

### 1. Curves, not a number — and the curve *is* the dispatch threshold
CPU-SIMD, Metal, and CUDA cross over at different problem sizes; GPU almost always
*loses* below some size (launch + transfer dominate) and wins above it. Publish a
**sweep** — (N, dim, batch, precision) on the x-axis, backend lines on the y — and
mark the crossover. That crossover is exactly the value the `ann.Backend` /
`WeightMat` dispatch should key off (the GPU analogue of `linalg`'s
`parThreshold`), so the benchmark's *product* is a tuned threshold, not just a
picture. A single point hides the one thing worth knowing.

### 2. Separate the cost components; benchmark the *real* pattern
A naïve `b.N` loop that pays upload + launch + readback every iteration measures
overhead, not the kernel, and makes GPU look terrible while hiding where it wins.
Every GPU row must break out and report:

- **one-time**: context create, PTX/pipeline compile/JIT, index/weight upload &
  residency (amortized over a session, *not* per query);
- **per-launch**: kernel launch overhead;
- **transfer**: H2D query upload + D2H result readback — **backend-dependent and
  must be labeled**: on CUDA (discrete) these are explicit copies
  (`WriteInt8s`/`ReadFloats`, per `gpu/cuda.go`); on Metal (UMA) an `MTLBuffer` is
  a zero-copy Go slice, so the transfer line is ~free. Reporting one "GPU time"
  that folds these together is the single most common way to mislead here.
- **steady compute**: the kernel itself, warm.

Then benchmark the way the code actually runs: **resident** weights/index,
**warm** (`ResetTimer` after ≥N warmup launches), **batched** (`QueryBatch`, not a
single-query loop — batching is the entire reason the GPU is here). Use the same
sync discipline the real caller uses — include D2H only if the result must land on
host.

### 3. Couple perf to parity — and hold precision fixed
A fast-but-wrong kernel is worse than no kernel. aikit already has the right gate
built in: `anncuda`/`annmetal` compute the *same* int32 dot as CPU
`linalg.MatmulBTW8A8`, so **GPU and CPU rank identically**, and the existing
harness measures recall@k against the **exact-CPU Flat top-k**. Reuse that: the
GPU backend's recall-vs-exact-CPU is simultaneously the **parity gate** (must
match CPU's ranking within the documented tolerance) and the **quality metric**.
Refuse to emit a perf number for any backend/shape/arch that isn't parity-passing
(the "decline unsupported, never mis-run" rule). And **control precision**: if a
GPU path runs f16/int8 where CPU ran f32, a raw speedup conflates hardware with
precision — fix the precision across the comparison, or report both and label it.

### 4. The machine matrix is multi-box by nature
Metal exists only on Apple Silicon; CUDA only on NVIDIA — they **cannot be
measured on one machine, ever**. So single-machine evidence isn't merely
incomplete for a cross-backend claim, it's impossible. Consequences:

- Every record carries a full **device spec**: GPU model, driver version, CUDA SM
  arch / Metal family, VRAM (and for the CPU baseline: chip, cores, `GOARCH`).
- The doc states plainly *what was measured where*; no silent cross-machine
  extrapolation.
- This is goinfer's "second-machine-confirmation debt" (DECISIONS.md) becoming
  **structural**, and the policy is settled (see *Where the benchmarks live*): no
  GPU number is CI-gated — cross-machine numbers *cannot* be, since Metal and CUDA
  never co-reside — so **all** GPU rows are a **documented periodic pass** on
  self-hosted M-series + NVIDIA runners.

  NOT YET TRUE of §5's guard, and stated here so the plan is not mistaken for the
  state: default CI has ONE benchmark job (`perf benchmarks`), it runs two
  benchmarks at `-benchtime=1x` and asserts NOTHING about timing — deliberately,
  since a shared runner cannot resolve the percentages that matter. The
  no-regression comparison is `tools/perfgate`, which runs on a known box by hand
  and whose VERDICT `releasegate` now requires in the CHANGELOG (audit G-04).

### 5. Re-run CPU as a contemporaneous baseline + a no-regression gate
Collect the CPU SIMD numbers on the *same boxes and harness build* the GPU runs
on, so the speedup is GPU-today-vs-CPU-today, not GPU-vs-a-stale-number. Separately,
the GPU build tags and dispatch shims must not slow the pure-Go default: a
`CGO_ENABLED=0`, no-GPU-tags benchmark run, diffed against the prior baseline with
`benchstat`, guards the core path the whole repo promises.

### 6. Real workloads and real data are the headline; microbenchmarks tune
Two hard-won rules from the existing harnesses:
- **Real embeddings, not synthetic** (`benchmarks/README.md`): random high-dim
  vectors concentrate distances, so recall@k is meaningless on them — the ANN GPU
  crossover must be measured on Model2Vec embeddings, same as the CPU harness.
- **End-to-end is the published number; kernel microbenchmarks are for tuning.** A
  great isolated GEMV can be swamped by the many-small-ops-with-transfers reality
  of a full ViT or encode. Keep `*_bench_test.go` microbenchmarks for kernel
  tuning; publish the batch workload (score a corpus, encode a batch, a full ViT)
  as the headline.

## Record schema (what every run emits)

One JSON record per (workload × backend × shape × precision), so `benchstat` and
ad-hoc analysis see comparable output across repos. **These records are the source
of truth**; the human-readable results doc is generated from them (next section),
so a published number can never drift from the run it came from.

```
workload        e.g. "ann.FlatI8.QueryBatch" | "vision.SigLIP" | "goinfer.decode"
backend         "cpu-simd" | "cuda" | "metal"
device          {chip/gpu, driver, sm_or_family, vram_mb, goarch, cores}
precision       "f32" | "f16" | "int8" | "int4"
shape           {n, dim, batch, k, seq, patches}   (only the relevant fields)
timing_ms       {one_time, per_launch, h2d, d2h, compute, wall}
throughput      queries_per_s | tokens_per_s | patches_per_s
quality         {recall_at_k, parity_ok, max_delta_vs_cpu}
speedup_vs_cpu  float   (same box; null if CPU-only or no CPU on this box)
meta            {aikit_commit, go, build_flags, seed, warmup_launches, iters}
```

## The results doc — generated from the records, never hand-maintained

The published table (`docs/BENCH-gpu-results.md`) is **generated** by a `bench
report` step that ingests `records.jsonl` and renders markdown — never hand-typed —
the same drift-proof discipline `benchmarks/README.md` keeps. Re-generating after a
run is the only way the doc changes.

**"cpu / metal / cuda in one table" is a merge of two machine runs, not one run.**
Metal exists only on Apple Silicon and CUDA only on NVIDIA; they never co-reside, so
one run emits `{cpu-arm64, metal}` and another emits `{cpu-amd64, cuda}`. The report
tool joins both machines' records on the comparable axis (workload × shape ×
precision).

**The `cpu` column is not one thing.** The CPU baseline paired with Metal is an
M-series arm64 chip; paired with CUDA it's a different amd64 chip. Absolute
latency/throughput compares *only within a machine*. So the report emits two views,
and **never** a raw `cpu | metal | cuda` table of absolute numbers (that silently
puts two different CPUs — and two machines — in adjacent columns):

1. **Per-machine tables — primary, apples-to-apples.** One per device, each a true
   same-box comparison showing the speedup the GPU actually delivers there:

   ```
   NVIDIA <gpu>, driver <v>, <cpu> amd64:
   | workload | shape | precision | cpu-amd64 q/s | cuda q/s | speedup | recall | parity |
   Apple <chip> arm64:
   | workload | shape | precision | cpu-arm64 q/s | metal q/s | speedup | recall | parity |
   ```

2. **Normalized cross-platform summary — the only honest all-backends view.**
   Absolute ms don't compare across machines, but *speedup over the CPU each GPU
   ships next to* does — the decision-relevant "is the GPU worth it on this
   platform" number:

   ```
   | workload | shape | precision | metal ×vs-M-CPU | cuda ×vs-amd64-CPU | crossover N×batch |
   ```

   The shared substrate GEMV is the one row even the raw kernel is cross-repo
   comparable on — normalize it to **GB/s (or %-of-peak)**, not ms.

goinfer's decode records feed the same `bench report` and land in their own
per-machine table (its own CPU-and-Ollama baselines); only the substrate-GEMV row
joins aikit's normalized summary.

## Where the benchmarks live

- **Substrate microbenchmarks** — `gpu/*_bench_test.go` (device-gated), the W8A8
  GEMV shape sweep. Shared vocabulary with goinfer's decode microbenchmarks.
- **aikit end-to-end** — extend `bench/harness.go` to accept a backend and a batch
  size, and add GPU rows (CPU-SIMD / cuda / metal) to the recall+latency table;
  add a ViT crossover under `gpu/`.
- **goinfer end-to-end** — decode throughput + parity, in goinfer, against the
  same `aikit/gpu` substrate; separate tables, its own baselines.
- **Gating** — GPU rows are **device-gated** (`Skip` when
  `CreateSystemDefaultDevice` fails), so `go test` on a GPU-less box is green. GPU
  perf is **not** default CI (needs self-hosted M-series + NVIDIA runners); run it
  as a documented periodic pass, like goinfer's kernel-corpus measurement.

## First slice (Phase 2 — the headline)

ANN is "the one workload with no GPU path at all." Start there: **`FlatI8` CPU
SIMD vs `anncuda` (and `annmetal`) `QueryBatch`, as an (N × batch) crossover on
real Model2Vec embeddings**, measured against the exact-CPU top-k for recall +
parity, with the four cost components broken out. That single sweep validates the
whole methodology and produces the first real dispatch threshold. The kickoff
prompt is [`internal/archive/perf-campaign-2026-07/bench-ann-first-slice.md`](internal/archive/perf-campaign-2026-07/bench-ann-first-slice.md).
