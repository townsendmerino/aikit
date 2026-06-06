# Task (aikit + goinfer): cap the matmul fan-out width (P-core straggler test)

> **For:** Claude Code, with both `~/tmcode/aikit` and `~/tmcode/goinfer` in the
> workspace (`go.work`). Small, low-risk perf experiment. **Measure first; only
> ship the default change if it wins.** Parity stays bit-identical.

## The hypothesis (your own data points at it)

The v0.5.0 sweep showed **P=4 → 68 tok/s, P=8 → 66** — going *wider* is *slower*.
On an M1 Pro (6 P-cores + 2 E-cores) that's the classic **straggler at the
join**: the fork/join barrier waits for the slowest worker, and an E-core handed
an equal 1/8 slice finishes ~40% later than a P-core, so all 8 wait on it. The
batched decode matmuls fan out to `GOMAXPROCS` workers; capping the fan-out
**width** to roughly the P-core count should tighten the join.

Note the earlier sweep varied **GOMAXPROCS** (whole-runtime thread cap), which
conflates two things. This experiment isolates the real knob: keep GOMAXPROCS at
its default, vary **only the matmul fan-out width**.

## Why it's parity-safe (state this, then still verify)

`parallelSpawnCols` partitions **output columns** across workers — each output
element is computed by exactly one worker doing the full K-reduction. Changing
the number of workers changes *who computes which columns*, never the summation
order of any single output. So results are **bit-identical for any width**.
`TestDecodeParity` must stay green regardless; run it to confirm.

## 1. aikit: add a fan-out width cap

In `aikit/linalg`, add a process-wide knob mirroring `SetParallelThreshold`:

```go
// SetParallelWidth caps the number of worker shards a parallel matmul fans out
// to (0 = use GOMAXPROCS, the current behavior). Lower it to avoid slow-core
// stragglers at the fork/join barrier on heterogeneous CPUs (Apple big.LITTLE,
// Intel P/E). Does not affect numerics — output columns are partitioned, so
// any width is bit-identical.
func SetParallelWidth(n int)
func ParallelWidth() int
```

Wire it into the spawn path: the worker count becomes
`min(effectiveWidth, GOMAXPROCS, work-derived cap)` where `effectiveWidth` is
the knob (GOMAXPROCS when unset). Don't change the threshold logic — width and
threshold are orthogonal (threshold decides *whether* to parallelize; width
decides *how many* shards).

## 2. Sweep — the end-to-end arbiter, both tiers

Drive it from **goinfer's `BenchmarkDecode`** (not a linalg microbench — the
microbench lesson from this campaign: warm pools / shapes mislead). Keep
GOMAXPROCS at the machine default. Sweep width W ∈ {2, 3, 4, 5, 6, 8}:

- **0.5B** (Qwen2.5-Coder-0.5B int8int8) — the current shipping tier.
- **1.5B** (Qwen2.5-Coder-1.5B int8int8) — optimum may differ; bigger matmuls
  amortize the barrier better, so they may prefer a higher width.

Record tok/s per width for each tier. Also re-confirm `allocs/op` is unchanged
(this shouldn't touch allocation).

## 3. Decision gate

- **If a fixed cap beats width=GOMAXPROCS by ≳3% on the 0.5B** (the data suggests
  W≈4–6): wire goinfer's decode default to set it — same place `main.go` /
  `decoder` already calls `SetDecodeParallelThreshold`. Add a
  `DefaultDecodeParallelWidth` and call `linalg.SetParallelWidth` with it.
  - Pick the width that's best (or tied-best) across **both** tiers; if they
    disagree, favor not-regressing the 1.5B. Comment that the value is
    M-series-tuned and the knob is hardware-specific (x86 P/E and core counts
    differ) — the consumer that knows the workload sets it; aikit's default
    stays GOMAXPROCS.
  - Pure-Go caveat to note in the comment: Go can't pin goroutines to P vs E
    cores, so this only *reduces the odds* of an E-core straggler by using fewer
    shards — it's a statistical win, not a guarantee.
- **If no width beats GOMAXPROCS by a clear margin:** record "no win — default
  stays GOMAXPROCS" in `perf-campaign.md` and stop. That's a valid, honest
  outcome.

## 4. Done

- [ ] `linalg.SetParallelWidth` / `ParallelWidth` added; orthogonal to threshold;
      `gofmt`/`vet`/`go test ./...` green in aikit (incl. `-race`).
- [ ] `TestDecodeParity` bit-identical at every swept width.
- [ ] Width sweep recorded in `goinfer/docs/perf-campaign.md` for both tiers.
- [ ] Decision applied: either a new `DefaultDecodeParallelWidth` wired in goinfer
      (if it won) or a "no win, default unchanged" note (if it didn't).
- [ ] If it won and folds into the not-yet-pushed aikit v0.5.0, include it there;
      else it's a clean v0.5.1 follow-up (note which).
```
