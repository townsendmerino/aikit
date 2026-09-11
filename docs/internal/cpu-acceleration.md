# CPU & SIMD acceleration (internal notes)

How aikit's pure-Go compute is accelerated, where it lives, how to test it, and
the open micro-kernel follow-ups. Internal/maintainer notes — the user-facing
story is the README + godoc.

> **GPU is goinfer's now.** The WebGPU backend (`encoder/gpu`) was removed from
> aikit in the v0.4.0 split — it carries the cgo `webgpu` dependency, which the
> core deliberately excludes. GPU matmul lives in `goinfer/gpu` behind the
> `encoder.Backend` seam; aikit ships only the pure-Go CPU backend. This doc is
> CPU/SIMD only.

---

## Two layers

**1. `linalg/` — the shared SIMD kernels (public package).** The single home for
the hand-written assembly, used by both `encoder` and goinfer's decoder. Dispatch
by build tag + runtime CPU detection:

| Arch  | Files | Kernel |
|-------|-------|--------|
| arm64 | `linalg/dot_arm64.{go,s}`, `dot_i8*_arm64.s`, `dot_w4a8_arm64.s`, `dotprod_arm64_*.go` | NEON; int8 `dotI8` upgrades to `SDOT` on DotProd-capable CPUs (runtime HWCAP); `dotW4A8GroupsSDOT` is the fused int4×int8 decode kernel (nibble-unpack prologue + the `dot_i8dp` SDOT body) |
| amd64 | `linalg/dot_amd64.{go,s}`, `dot_w4a8_amd64.s`, `quant_w4a8_amd64.go` | AVX2+FMA (`dotFMA`/`dotFMA4`/`dotFMA8`), int8 `dotI8AVX2` (VPMOVSXBW+VPMADDWD), and `dotW4A8GroupsAVX2` — the fused int4×int8 decode kernel (nibble-unpack prologue + the `dotI8AVX2` sign-extend body); runtime CPUID/XGETBV detect, scalar fallback |
| other | `linalg/dot_generic.go`, `dot_other.go` | portable scalar |

On top of the dot kernels, `linalg` provides:
- `Dot`, `Dot4x4`, `Dot8x4`, `Dot2x8`, `MatmulBT` (f32 column-parallel),
  `MatmulBTInto` (serial). **`MatmulBT`/`MatmulBTInto` are cache + register blocked**
  (`matmul_blocked.go`: 32×32×768 tiles over the Dot8x4/Dot2x8 kernels) above an
  M·K·N threshold; below it they keep the naive dot-per-output span (small matmuls
  like attention QKᵀ don't want the tiling prologue). This blocked GEMM is the single
  shared home — the encoder's transformer matmuls and other kit consumers route
  through it (it was hoisted out of the encoder once the un-blocked `MatmulBT`, which
  re-streamed `b` per a-row, measured ~7% of peak at prefill shapes). `Dot2x8` (arm64
  NEON) is the MR×NR register kernel inside it: 2 a-rows × 8 b-rows, 16 accumulators
  held across the K loop so each b-load feeds 2 FMLAs — vs `Dot8x4`'s 1×8, which was
  load- and latency-bound (≈40% of the *measured* 95.4 GFLOPS M1-Pro f32 ceiling;
  `BenchmarkGEMMPeakFraction` + the `fmaPeakARM64` ceiling probe gate this). It
  computes each dot in `Dot8x4`'s accumulation order (bit-identical), so the blocked
  GEMM differs from the naive span only by f32 reassociation; `MatmulBTAcc64` stays
  f64-exact. Column shards are 8-aligned, so `SetParallelWidth` stays numerically inert.
  At **K≥2048** the blocked path (arm64) first **packs** each 8-row b-group into a
  contiguous low-stride buffer (`packedFill`): at large K the simultaneously-read b-rows
  are K·4 bytes apart and collide in L1 cache sets, so packing them ~kBlock apart kills
  the conflicts (prefill 46%→69%, K=3072 fc2 +15%) — bit-identical (same values, same
  order), via a pooled buffer. K=768 dims stay unpacked (already low-stride); amd64 stays
  on the unpacked AVX2 path (AVX2 packing deferred). A **padded pack stride** breaks the
  power-of-two L1 conflict when kSpan is a power of two (item 24): **−9.8%/−10.7%/−8.8%
  on large-encoder fc2** on `apple-m1pro` — arm64-only, and it measured as pure noise on
  `nvidia-rtx2070s` where the whole function is dead code (see perf-dead-ends §6.1 for
  why that flat sweep was a signal, not a null result).
- **`packedFillQ8` / `MatmulBTQ8Fused` — the int8-weight twin of the packed path**
  (`matmul_blocked_q8.go`, arm64). It widens each int8 weight to f32 **inside** the pack
  tile — the ≤32 KB L1-resident buffer above — instead of materializing the whole
  `[N,K]` f32 weight first, so the ~0.9 GB/forward `deqW` DRAM round-trip is gone.
  Bit-identical to `DequantizeRowsInt8Into` + `MatmulBTInto` (the widen value and
  k-tiling match; `TestMatmulBTQ8Fused_bitIdentical`, mutation-checked). The encoder Q8
  path routes here on arm64 (`FusedQ8Applies`, gated on `HasFusedQ8Kernel` + `N%8==0`):
  **−28%/−10%/−4.8%/−2.0% end-to-end at L=8/64/256/512** on `apple-m1pro`, 1/M-shaped
  (the widen is fixed per forward). The dispatch mirrors BOTH f32 parallel axes —
  columns for small M, rows for large M — so each worker runs the fused kernel over a
  small row block and never opens `packedFill`'s missing-m-blocking gap (item 23). amd64
  keeps the vectorized dequant-then-GEMM path (`HasFusedQ8Kernel` false there).
- The quant matmuls: `MatmulBTQ8` (int8 weights), `MatmulBTQ4` (int4 group, f32
  activations — prefill path), **W8A8** (`MatmulBTW8A8` + the zero-alloc
  `…Into(ws *Workspace)` and the fused `MatmulBTW8A8Batch`), and **W4A8**
  (`MatmulBTW4A8`, int4 weights × int8 activations — the int4 *decode* path).
  See `quant.go`, `workspace.go`, `quant_w4a8*.go`.
- `DequantizeRowsInt8Into` — bulk int8→f32 weight widen (`float32(q)*scale` per
  element), vectorized both arches: amd64 AVX2 (`VPMOVSXBD`+`VCVTDQ2PS`+`VMULPS`,
  32/iter), arm64 NEON (`dequant_i8_arm64.s`, `SXTL/SXTL2`→`SCVTF`→`FMUL`, 16/iter).
  Bit-identical to the scalar loop. This is item 22's fix (a); `packedFillQ8` above is
  fix (b), which fuses the same widen into the pack so the full f32 matrix never lands.
- Dispatch knobs: `SetParallelThreshold` (MAC count to parallelize above) and
  `SetParallelWidth` (cap fan-out shards, for P/E straggler control) — both
  numerically inert (output columns are partitioned). (`pool.go`, the optional
  per-`Workspace` spin-then-park worker pool this used to name, was PULLED —
  see perf-dead-ends §8.1. `workspace.go` spawns per call, and since audit M-08
  hands the shards out from an atomic counter rather than one fixed slice each.)

**2. `encoder/` — the encoder's matmul orchestration.** `encoder/linalg.go` is now
thin: `matmulBTInto` dispatches small shapes to a naive in-package loop and large ones
to `linalg.MatmulBTInto` (the shared blocked GEMM), with a lone-forward row-parallel
path (`parallel.go`, each worker calling `linalg.MatmulBTInto` on its row block). The
tiling + register kernels moved to `linalg`;
`encoder/linalg_q8.go` is its int8 variant; `encoder/parallel.go` row-splits a
**single** `Encode` across cores (gated by an atomic in-flight-forward counter +
its own `parallelThreshold`, so `EncodeBatch` — already core-saturated at the
document level — stays serial per forward). This is separate from `linalg`'s
knobs above.

**Parity invariant (both layers):** parallelization and re-blocking partition
*output columns/rows* — each output is computed by one worker doing the full
K-reduction — so they're **bit-identical** to the serial path, not just within
tolerance. Tests assert exact equality.

---

## Status

- **amd64 AVX2 validated on Linux (2026-06-02, Ryzen 7 3700X, Zen 2, Go 1.26.3).**
  Every `AVX2|Dot` test PASSes with `hasAVX2=true`, no SIGILL, `-race` clean; the
  single-row and register-blocked kernels bit-match the scalar reference. Numbers
  + the one tuning finding below.
- **Perf campaign landed (v0.5.0–0.5.2)** in `linalg`: zero-alloc W8A8 decode
  (`Workspace`/`…Into`), batched W8A8 (`MatmulBTW8A8Batch`), serial-decode
  threshold + `SetParallelThreshold`/`SetParallelWidth`, the spin-park pool, and
  the column-outer W8A8 re-block (weight reused across M rows). See CHANGELOG.
  goinfer's end-to-end decode is the arbiter for those (warm microbenches mislead).
- **2026-07 CPU perf campaign (arbited across two boxes).** The CPU/SIMD-relevant
  kernel outcomes are folded into this doc: item 22a (`DequantizeRowsInt8Into` NEON
  widen, **3.43×** kernel on `apple-m1pro`), item 22b (`packedFillQ8` fused widen,
  −28%/−10%/−4.8% forward), item 24 (padded pack stride, −9.8% fc2, arm64-only). The
  full audit trail is archived under `archive/perf-campaign-2026-07/`; the two Amdahl
  decompositions are `perf-amdahl-apple-m1pro.md` (6P+2E, §5 "what transferred") and
  `perf-amdahl-linux-amd64.md`, meant to be read side by side; and every kernel idea
  that was tried and did NOT ship — item 37 outer-product, item 25, pre-packing weights
  — is in `perf-dead-ends.md` with its mechanism and number. **The one-line rule from
  that work:** on `apple-m1pro` `dotNEON2x8` is at ~95% of FMLA peak, so the f32 kernel
  is compute-bound and no load-reduction lever helps here even where the amd64 analysis
  said it would (perf-dead-ends §2).
- **`ann.FlatI8.EnableGPUShardSplit` (CPU∥GPU shard-split QueryBatch) — shipped
  2026-08-20, real win bounded and share/box/scale-dependent, NOT the ~1.5-1.7×
  the whole-corpus throughput ratio implied.** Follow-on from the July 2026 GPU
  crossover finding (M×N readback fixed by fusing top-k on-device; see
  `archive/perf-campaign-2026-07/perf-campaign-2026-07-28.md`): split the corpus
  into a device-resident shard and a CPU-scored shard, score both CONCURRENTLY
  per `QueryBatch` call, merge. The premise (measured on `apple-m1pro` at
  N=100k/batch=64: CPU 2.3k q/s, Metal 3.4k q/s, "neither saturates DRAM, so
  splitting should be roughly additive") assumed CPU and GPU are close in
  speed. Measuring the actual shard-split path — not just the whole-corpus
  numbers it was extrapolated from — on both boxes shows that assumption only
  holds sometimes:
  - **`apple-m1pro` (Metal): real, modest wins, mostly positive.** At
    N=100,000/batch=64, `metal`-only is 3548 q/s; `cpu+metal` at share=0.60
    (60k-row GPU shard) is 4485 q/s (**1.26×** over GPU-only). At
    N=10,000/batch=256, `metal`-only 23,776 q/s → `cpu+metal` share=0.35
    34,803 q/s (**1.46×**, the best case measured). But batch=1 is always
    worse than CPU-simd alone (GPU dispatch overhead isn't amortized at one
    query), and a badly-chosen share can lose to GPU-only outright (N=10,000/
    batch=64, share=0.60: 15,804 q/s vs `metal`-only's 19,726 — **0.80×**).
  - **`nvidia-rtx2070s` (CUDA): wins only at small N, actively HURTS at
    N=100,000.** At N=100,000/batch=64, `cuda`-only is 33,386 q/s
    (**30×** over CPU's 1108 q/s — this box's CPU/GPU gap is an order of
    magnitude larger than the M1's ~1.5×). Every tested share LOSES to
    GPU-only there: share=0.60 gives 6,906 q/s (**0.21×** of GPU-only),
    share=0.35 gives 4,228 q/s (**0.13×**) — the CPU shard, even at 35-40% of
    the corpus, takes far longer than CUDA's much larger complementary share,
    so the `sync.WaitGroup` merge waits on the slow side and the combined
    result is WORSE than just using CUDA alone. At N=10,000 (CPU/GPU gap only
    ~6×), the same technique genuinely helps: `cuda`-only 38,083 q/s →
    `cpu+cuda` share=0.60 44,513 q/s (**1.17×** over GPU-only, and the best
    absolute number in the whole sweep).
  - **The one-line rule:** shard-split is worth it only when CPU and GPU
    throughput are within roughly the same order of magnitude for that N —
    check the box's own `Query`/`QueryBatch` crossover numbers first. When the
    GPU is 10×+ faster than CPU at the N in question (CUDA at N≥100k on this
    box), giving the CPU ANY meaningful share of the corpus makes it the
    bottleneck and shard-split should not be used. `gpuShare` is not
    auto-tuned (see `EnableGPUShardSplit`'s doc comment) — pick it from a
    measurement like this one, per box, per N, not from the whole-corpus
    throughput ratio.
  - Correctness held everywhere measured (`parity=true`, exact-CPU-matching
    recall, on both boxes, all shares, all batches) — this is a throughput
    finding, not a correctness one. Full sweep:
    `docs/bench-records/crossover-{metal,cuda}.jsonl` (`Backend:
    "cpu+metal"`/`"cpu+cuda"` rows), reproducible via
    `AIKIT_GPU_BENCH=1 go test ./gpu/annmetal/... -run Crossover` (and the
    CUDA mirror in `gpu/anncuda`).
- **`WeightMat` int4 storage now has three representation policies, fixed at
  construction (audit M-22, aikit side; CHANGELOG `[Unreleased]`).**
  1. **Canonical-only** (`WrapInt4`) — packed nibbles + per-group scales, the
     original storage. Every accessor works; `Int4()` returns them.
  2. **"Both"** (`WrapInt4` + `RepackInt4Row4`/`RepackInt4SplitHalf`) — canonical
     PLUS the arch-specific repacked layout (row4 on arm64/NEON, split-half on
     amd64/AVX2), the pre-M-22 fast-matmul path. Costs 2× the nibbles (row4
     also 2× the scales, since split-half shares scales unrepacked). Unchanged
     by M-22 — same bytes, same dispatch order, same code path.
  3. **Repacked-only** (`RepackInt4Row4InPlace`/`RepackInt4SplitHalfInPlace`,
     or `WrapInt4Row4Only`/`WrapInt4SplitHalfOnly` for already-repacked bytes)
     — ONLY the repacked layout, no canonical bytes resident at all. The
     memory win M-22 exists for. The in-place constructors permute the
     caller's own q4/q4s bytes using one quad's (row4) or one row's
     (split-half) worth of scratch — never a second tensor-sized array —
     because both layouts are fixed permutations LOCAL to one quad/row; a
     loader that wants canonical to never exist in the heap at all can instead
     stream a checkpoint's bytes straight into repacked order via the
     exported per-quad/per-row primitives (`RepackInt4Row4Quad`+
     `RepackInt4Row4ScalesQuad`, `RepackInt4SplitHalfRow`) into a fresh
     destination buffer.
  <br>**The rule that keeps `Int4()` safe across all three: `Int4()` means
  "canonical bytes present", not "this tensor is int4".** A repacked-only
  WeightMat is int4 (`Kind()` returns `"int4"`, `IsInt4()` returns `true`,
  `Int4Layout()` names which layout) but `Int4()`'s `ok` is `false` there,
  deliberately — a GPU consult or any other canonical-bytes-only caller reads
  through `Int4()`, and `ok=true` with a nil slice would be the worst
  outcome. Ask `IsInt4()`/`Int4Layout()` for "is/which int4"; ask `Int4()`
  only when canonical bytes specifically are needed. `Row()` is
  layout-independent (identical output across all three policies, bit-exact
  to `DequantizeRowInt4`), so a repacked-only tensor still answers `.Row()`
  correctly — the property that makes a tied embedding table (per-token
  `.Row()` reads AND matmul-as-LM-head) a valid repacked-only candidate, not
  an exception requiring "both".
  <br>Aikit-side only: no loader in this repo calls the new constructors on a
  real checkpoint yet. The goinfer-side adoption (which tensors qualify, when
  to prefer streaming over in-place) is a documented follow-up, not done here
  — see the M-22 entry in `docs/audit-2026-09-10.md`.

### AVX2 kernel numbers (Ryzen 7 3700X, `-bench 'Dot'`, MB/s)

| K     | scalar `DotGo` | single-row `dotFMA` | Dot4x4   | Dot8x4        |
|-------|---------------:|--------------------:|---------:|--------------:|
| 64    | 7.4 GB/s       | 14.3 (1.9×)         | 30.8     | 35.3          |
| 768   | 8.1 GB/s       | 44.2 (5.5×)         | 51.9     | **86.5**      |
| 3072  | 8.4 GB/s       | 49.9 (5.9×)         | 51.4     | 40.5 ⚠        |

Single-row AVX2 is ~6× scalar at the linear-layer widths; register-blocking adds
a-reuse on top (`Dot8x4` peaks 86.5 GB/s at K=768). **⚠ `Dot8x4` regresses at
K=3072** — below `Dot4x4` and even single-row — the 8 live YMM accumulators plus
streamed b-rows exceed what stays hot at large K. See follow-up §1.

---

## Testing

The SIMD kernels and their differential tests live in `linalg/`:

```bash
# Kernel correctness — AVX2 asm bit-matches the scalar reference (all tail sizes).
# TestAVX2_detection logs hasAVX2; on a non-AVX2 box the asm tests SKIP.
go test ./linalg/ -run 'AVX2|Dot|W8A8|Batch|ParallelWidth' -v

# Race-clean parallelism (parallel matmul, W8A8 pool, width).
go test -race ./linalg/

# Kernel throughput.
go test ./linalg/ -run XXX -bench 'Dot|MatmulBTW8A8|DecodePool' -benchmem

# asmdecl validates every asm stack offset vs the Go signatures (CI runs this).
go vet ./linalg/
```

Encoder-level (forward-pass parity + single-forward parallelism):

```bash
go test -race ./encoder/...            # incl. parallel.go exactness + threshold benches
go test ./encoder/ -run XXX -bench 'MatmulParallel|MatmulSerial' -benchmem
```

Model-dependent encoder tests (golden cosine vs CodeRankEmbed) skip cleanly when
the checkpoint is absent (CI), and run when `testdata/encoder-model` is present.

---

## Open follow-ups (aikit, CPU-only)

1. **`Dot8x4` large-K cliff — already mitigated at the call site; now documented
   on the public kernel.** `Dot8x4` wins at mid-K (~768) but loses to `Dot4x4`
   past it (the K=3072 regression above). The encoder does NOT hit this: its
   blocked matmul tiles K at `kBlockDefault=768` (encoder/linalg.go), which is
   exactly `Dot8x4`'s peak — `fc2` (K=3072) runs as 4×768 strips, not one 3072
   strip. So there's no call-site heuristic to add; the M10 tile tuning already
   handles it. The real exposure was the *public* `linalg.Dot8x4` godoc not
   warning external callers — now fixed (it documents the cliff and the
   "tile K to ≤~768" guidance). Revisit only if a profile shows a real caller
   feeding it large-K rows.
2. **AVX-512 path** (optional, Zen 4 / recent Intel). 16-wide, more registers;
   AVX2 already covers ~all amd64 since 2015 and AVX-512 brings downclocking
   caveats, so low priority. Same shape: CPUID leaf 7 detect, `dot_amd64.s`
   entry points, `hasAVX512` gate.
3. **Per-head attention — QK^T parallelization CLOSED; scores·V vectorized
   instead.** End-to-end CPU profile of `Model.Encode` on real weights (~500-tok
   input, `BenchmarkEncode_singleLong`) overturned the microbench-driven premise:
   QK^T is already SIMD and only ~2.6% of `Encode`, so parallelizing it across
   heads (144 spawns/forward) chases nothing. The actual hotspot was the **scores·V
   context accumulation** — a scalar triple-loop (`ctx = scores · V` per head) that
   was the single hottest line at ~⅓ of `Encode`. Fixed by folding a per-head V
   transpose into the extract and routing scores·V through the SIMD `matmulBTInto`
   (A·Bᵀ), in both `selfAttention` and `selfAttentionBatched`. Bit-exact (golden
   cosine 1.0, batch==single, `-race` clean). The win is the L² term, so it scales
   with sequence length: **~2.85× single `Encode`** at ~500 tokens, neutral (no
   regression) at ~80-token rerank passages where scores·V is a small share.
   *Follow-up — DONE, and this line was stale:* `forward_q8.go` no longer has the
   scalar scores·V loop. Both its attention paths route QKᵀ and scores·V through
   `s.mm` — `forward_q8.go:180,185` (single) and `:247,253` (batched) — which
   dispatches to `matmulBTInto` or an attached backend, exactly as the f32 sibling
   does.
4. **amd64 AVX2 `MatmulBTW4A8` kernel** — ✅ **DONE** (`dot_w4a8_amd64.s`,
   `quant_w4a8_amd64.go`). The fused int4×int8 decode kernel now exists for amd64
   too: the same nibble-unpack prologue feeding the proven `dotI8AVX2`
   sign-extend body (VPMOVSXBW+VPMADDWD+VPADDD) — fully signed, no
   unsigned-offset trick, since the nibbles are centered to int8 in-register
   first. Gated by `hasAVX2`; non-AVX2 amd64 and non-DotProd arm64 keep the
   scalar `dotW4A8Scalar`. **Validated on a Zen 2 box (Ryzen 7 3700X, AVX2, no
   VNNI):** matches the scalar oracle bit-for-bit, race-clean; at M=1 decode it
   lands ~1.7–1.9× of W8A8 and ~32× faster than `MatmulBTQ4` — on par with the
   arm64 SDOT kernel (~2.0–2.3×).
   - **The VNNI variant — BUILT, MEASURED, SHIPPED 2026-08-26 (`2a7199a`).**
     `VPDPBUSD`, one instruction replacing the VPMOVSXBW+VPMADDWD pair, behind
     its own CPUID+XGETBV gate. This entry predicted it "can't be validated on
     the Zen 2 box"; that stayed true — neither local machine can execute it
     (Ryzen 3700X is Zen 2, the other box is arm64), so it was written and run
     on a VNNI-capable Xeon in a cloud session (`hasAVX512VNNIVL=true`
     confirmed via CPUID/XGETBV, not assumed). Correctness first: the
     scalar-oracle tests pass on real VNNI silicon
     (`TestAVX512VNNI_dotW4A8FoldAVX512VNNI_matchesScalar`,
     `..._dispatchesThroughEveryTier`, `TestW4A8_dotMatchesScalar`).
     **THE NUMBERS**, at `TestW4A8OpsPerByte`'s own shape (K=5120, group=32,
     N=17408 → 55.7 MB, well past that box's L3, so it extends the existing
     harness rather than measuring something else), three separate runs:

     | run | hot AVX2 | hot VNNI | hot | cold AVX2 | cold VNNI | cold | cold GB/s |
     |--:|--:|--:|--:|--:|--:|--:|--:|
     | 1 | 15.28 | 19.10 | **1.250×** | 11.96 | 13.37 | **1.117×** | 7.48→8.36 |
     | 2 | 15.61 | 19.25 | **1.233×** | 11.96 | 13.76 | **1.151×** | 7.48→8.60 |
     | 3 | 15.47 | 18.96 | **1.226×** | 10.04 | 11.13 | **1.109×** | 6.27→6.96 |
     | 4 | — | — | **1.280×** | 10.16 | ~12.90 | **1.270×** | — |

     (GMAC/s. Run 4 was added when the harness's own gate was being proved;
     its cold VNNI figure is derived from the reported ratio and baseline.)

     **The question this answers is whether the hot win survives going cold** —
     §8.9's issue-width probe had already found the AVX2 kernel with idle issue
     slots while streaming from DRAM, exactly the shape where a wider
     instruction should buy nothing. **It survives. How much of it survives is
     NOT settled by these four runs**, and the first three alone would have
     overstated the confidence:

     - **Hot is settled**: 1.226–1.280× across all four, ±2.2% about the mean.
     - **Cold is real but unsettled**: 1.109–1.270×, ±6.8%. Above 1.0 in 4/4,
       so the win itself is not in question — its size is.
     - **Survival** (cold gain ÷ hot gain) is therefore **47%, 65%, 48%, 96%**.
       An earlier revision of this entry read the first three as "compressed to
       roughly half." Run 4 breaks that; the honest range is ~47–96%, which is
       too wide to support any claim about the compression factor.

     **The obvious explanation for run 4 does not hold.** A high ratio usually
     means a depressed baseline, but run 4's cold AVX2 (10.16) is within 1% of
     run 3's (10.04) — the *VNNI* cold figure is what moved, 11.13 → ~12.90
     (+16%) at an essentially identical baseline. So this is genuine variance in
     the measured quantity, not a ratio artifact, and it cannot be dismissed.

     **Trust the hot ratio; do not quote a cold one.** Cold AVX2 swings
     10.04–11.96 GMAC/s across the runs (~19%) on a shared/virtualized cloud
     host — the same hazard that blew a 600s CI timeout on the measurement
     harnesses (the AIKIT_HARNESS gate in `dd28f90` exists for it). Pairing
     within a run fixes that for the hot ratio but demonstrably not for the
     cold one. **Settling the cold number needs quiet silicon, not more runs on
     this host.** Until then: "VNNI is ~1.25× hot and measurably faster cold" is
     supported; any specific cold multiplier is not.

     **Scope, so this is not over-read.** It answers the SIMD-width question
     only. It says nothing about the thread-scaling gap this campaign already
     flagged as the larger lever, and `dotW4A8`'s VNNI path is NOT
     bit-identical to its AVX2 one (1e-5 relative to the scalar oracle, the
     same bar AVX2 meets) — so amd64 now has two W4A8 result classes split by
     VNNI+VL. `dotI8`'s VNNI tier is exact on every path (integer, no
     reassociation) and carries no such caveat.

     **Reproducible in-tree:** `w4a8_vnni_opsperbyte_bench_test.go`, gated on
     `harnessOnly(t)` as its first statement — before the `hasAVX2` /
     `hasAVX512VNNIVL` checks, so the skip reason is the harness gate rather
     than the hardware. Verified both directions (skips bare, runs under
     `AIKIT_HARNESS=1`), so it does not reintroduce the CI timeout `dd28f90`
     added that gate to stop. Run it with:
     `AIKIT_HARNESS=1 go test ./linalg/ -run TestW4A8VNNIOpsPerByte -v`
   - **Ops-per-byte measured, 2026-08-19** (`linalg/w4a8_opsperbyte_bench_test.go`,
     requested from goinfer after Qwen3.8-27B CPU decode measured 2.55× slower than
     an Ollama/Q4_K_M engine on the same Ryzen 7 3700X — the parallelization
     threshold and the quant format's byte-count were each ruled out first, at
     ×5-1200 headroom and ~11% respectively, neither close to 2.55×). Measured on
     the same box: `dotW4A8FoldAVX2` at the FFN gate/up/down shape (K=5120)
     achieves **16.62 GMAC/s hot** (L1-resident) and **15.90 GMAC/s cold**
     (streaming 55.7 MB, forcing real DRAM reads) — cold is only **1.05× slower
     than hot**, at 9.94 GB/s against this box's ~51 GB/s DDR4-3200 peak. Neither
     regime is memory-bound; the kernel is compute-limited at both — **on ONE
     THREAD**, which is the qualification this paragraph used to omit. S-08.1
     later measured that box's read bandwidth directly: it SATURATES AT TWO
     THREADS (30.5 GB/s) and then declines slightly, and the ~51 GB/s figure is
     a DDR4-3200 spec ceiling reached at ~60%. So "compute-limited" holds per
     core and does NOT extend to a fanned-out decode, where 9.94 GB/s per worker
     crosses the real ceiling by three or four workers.
     Against `dotI8AVX2` (same K, no nibble-unpack prologue, no per-group scale
     fold) at 51.20 GMAC/s hot, `dotW4A8FoldAVX2` is **3.08× slower per MAC** —
     the measured cost of int4's unpack + fold, achieving only ~32% of that
     reference's throughput.
     The marginal-FMA issue-width probe (`priors-microgpt-c.md` §1) on the cold
     kernel gives ratio **0.91 — NOT issue-limited**: idle execution-port
     capacity exists even while streaming from DRAM, so the bottleneck is not
     raw instruction/uop count competing for ports. Leading hypothesis, from the
     assembly (not yet perf-counter-confirmed — `perf` wasn't available on the
     measurement box): `dotW4A8FoldAVX2` accumulates into **one** f32 register
     (`Y10`, via `VFMADD231PS` every one of the 160 groups) — a serial RAW
     dependency chain across the whole K-loop, unlike `dotI8AVX2`'s **four**
     independent accumulators, whose own comment states they exist specifically
     to break this exact chain ("issue independently"). A single accumulator
     caps the loop at roughly one iteration per FMA-latency cycle count
     regardless of free ports — consistent with "not issue-limited" (ports are
     idle) yet still far below `dotI8AVX2`'s throughput (latency-bound, not
     issue-bound).
     It answered no: `dotW4A8FoldAVX2` is not close to `dotI8AVX2`'s achievable
     throughput on this box, so quant-format work is not the next step.
     **The obvious fix was tried and measured negative** — a 4-independent-
     accumulator variant (mirroring `dotI8AVX2` exactly) was built same-day,
     passed correctness (1e-5 rel-err vs the scalar oracle and vs the
     production kernel), and moved throughput by ~1%, inside noise (hot
     +1.4%, cold −1.0%). The dependency-chain hypothesis was wrong, or at
     least not dominant: "not issue-limited" from the marginal-FMA probe only
     rules out FMA-port contention specifically, not contention on a
     different port — the nibble-unpack prologue (8 shuffle/logic ops/group)
     is the more likely remaining suspect, unproven. Recorded as a measured
     dead end, not a re-triable one: `perf-dead-ends.md` §8.9.
     **Follow-up, same day — the unpack suspicion confirmed and quantified,
     but it's only half the story.** A diagnostic-only kernel (weights
     pre-unpacked to one int8/weight instead of packed nibbles — never
     shippable, it doubles the weight footprint) held everything else fixed
     (single accumulator, per-group scale-fold) and isolated the unpack cost
     directly: 295ns production → 188ns unpack-free → 106ns `dotI8AVX2`
     reference. Removing unpack alone recovers **1.57×**; the remaining
     **1.77×** to the reference is the per-group scale-fold
     (`VCVTDQ2PS`+`VBROADCASTSS`+`VFMADD231PS`, 3 instructions/group) that
     `dotI8AVX2` doesn't pay either (one overall scale, not per-32-group).
     Roughly a 57/43 split of the total overhead — no third factor: unpack +
     scale-fold fully account for the gap to `dotI8AVX2`. Diagnostic kernel
     removed after recording the number (same disposition as the accumulator
     experiment). Next lever considered, NOT started: a cheaper unpack
     sequence via `VPMADDUBSW` on raw unsigned nibbles (ggml's usual AVX2 Q4
     trick), skipping the explicit center-then-widen this kernel does.
     **Worked the saturation math before touching assembly** (`dotI8AVX2`'s
     own comment defers `VPMADDUBSW` for full int8×int8 over exactly this
     risk): the concern is general u8 range (max product magnitude
     `2×255×128=65,280`, over int16's ±32,767 ceiling), but a 4-bit nibble
     caps the unsigned operand at 15, not 255 — worst case
     `2×15×128=3,840`, over 8× inside the ceiling regardless of activation
     values. **Provably safe, no saturation risk.** But the instruction-count
     win is smaller than the naive "skip 4 ops" framing suggests: raw
     (uncentered) nibbles compute `Σnib·act`, not the true `Σ(nib-8)·act`,
     so a per-group correction `8·Σact` has to be added back somewhere.
     Handled optimally — precomputing `Σact` per group ONCE per token
     (mirroring `QuantizeActivationsInto`'s existing "quantize once, reuse
     across all N rows" pattern, since the activation row is shared across
     every weight row in one M=1 matmul) rather than recomputing it per
     row — the realistic count is **18 instructions/group vs the current
     20** (~10%, not the ~50% removing 4 ops alone implied), and getting
     even that needs a real calling-convention change to `MatmulBTW4A8Into`
     (a precomputed per-group activation-sum array threaded through), not a
     drop-in kernel swap like the two prior experiments. Given §8.9's
     accumulator experiment already measured a plausible-sounding
     instruction-count argument land at ~0% real speedup, a ~10% reduction
     needing an API change was judged not worth building blind — **stopped
     here, not built**, math and estimate recorded so it isn't re-derived
     from scratch if revisited. VNNI (`VPDPBUSD`) remains the more promising
     lever if a VNNI-capable box ever becomes available — one instruction
     class removes the unpack-widen AND the scale-fold's separate convert
     step, not just the unpack. Full numbers and the revised recommendation:
     goinfer's `docs/measurements/aikit-w4a8-opsperbyte.md`, answering
     `docs/prompts/aikit-w4a8-ops-per-byte.md`.
     **Cross-ISA status (2026-08-24) — this entry is amd64-scoped; three of its
     conclusions INVERT on arm64/NEON, measured in the later campaign** (full
     record: goinfer's `docs/task-w4a8-neon-bandwidth.md`):
     (1) the accumulator dead end does not transfer — `dotW4A8FoldSDOT` was
     latency-bound where this kernel is port-bound, a 2-accumulator NEON variant
     measured a real 1.4-1.47x, and the shipped fix (`dotW4A8SplitHalf4Row`,
     v1.26.0: split-half repacked layout + 4-row interleave, one accumulator per
     real row, bit-identical fold order) landed 1.6-1.75x isolated —
     `perf-dead-ends.md` §8.9's companion note carries the same narrowing;
     (2) the marginal-FMA issue-width probe is demoted to a hint, never a
     decision input — 0-for-2: here it pointed away from the true (shuffle-port)
     bottleneck, and on arm64 its original "issue-limited, ratio 1.11" reading
     failed to reproduce on a settled box (~0.99-1.03 across 4 re-runs);
     (3) the uncentered-`Σact` idea got its arm64 trial after all — a decoupled
     correction-pass shape measured 0.972x on NEON, consistent with this entry's
     stopped-not-built call; folding the correction into a repacked layout
     remains the only sanctioned retry. The VNNI paragraph above is unaffected
     (still hardware-gated, still amd64's most promising lever).

5. **`packedFill` m-blocking (item 23) — DEFERRED WITH MEASUREMENT, not dead.**
   `packedFill` re-reads the a-panel once per 8-column group (it lost `blockedFill`'s
   m-blocking). On `apple-m1pro` serial `packedFill` still runs at 75–81% of the
   ~42 GMAC/s kernel peak (fc2 M512/M690), so the a-re-read is one bounded contributor
   to a ~20% gap it shares with the compute-overlapping b-copy and the reduce. The
   m-blocking fix trades a-reads for redundant b-re-packing — a wash-risk core-GEMM
   restructure for a sub-gap win, so it is **not built**. Note the Q8 path already
   sidesteps it: `MatmulBTQ8Fused` row-splits large M so each worker's `packedFillQ8`
   runs at small M where the gap does not open. Revisit only with a profile showing the
   a-re-read is the dominant term, or as part of the §2.12-roadmap 3-level Goto GEMM.
6. **`dequantRowInt8` (K=768) — marginal-FMA issue-width probe run for real —
   ✅ DONE, NOT issue-limited on either box.** `docs/internal/priors-microgpt-c.md`
   §1 proposed injecting independent dead FMAs into a hot loop and comparing the
   marginal ns/FMA cost against the same loop's cost measured alone; a match means
   the kernel already occupied those issue slots (issue-limited, the `dotNEON2x8`
   story above), a gap means idle slots exist. Built as
   `linalg/fma_issue_probe_test.go`'s `TestFMAIssueProbe`, run on both boxes (best
   of 3, least-squares slope over N∈{0,8,16,32,64}, reproduced twice each):
   `apple-m1pro` stacked/alone ratio **0.94–0.97**, `nvidia-rtx2070s`'s Ryzen host
   (AVX2) **0.79–0.82** — both comfortably below 1.0, so **not issue-limited on
   either architecture**, even though the exact ratio differs between them (the
   AVX2 box shows more slack, not less — the opposite of what a naive read of
   priors-microgpt-c.md §2's "amd64 has fewer registers, less headroom" caution
   might suggest; the *qualitative verdict* transferred fine even though the
   *quantitative ratio* didn't, which is the distinction §2 is actually about).
   Implication: `dequantRowInt8` has idle issue slots on both boxes — it is
   waiting on something else (memory is the obvious suspect, given the kernel is a
   load/sign-extend/convert/multiply/store chain), not issue-width bound, so a
   scheduling/unrolling change here would have nothing to win against.
   **Confirmed by a later negative:** routing `q8Span`'s scalar int8→f32 widen
   through this kernel measured flat at the decode LM-head shape (6.79-6.85 →
   6.83-6.87 ms/op, `apple-m1pro`) — exactly what "not issue-limited" predicts.
   `perf-dead-ends.md` §8.10. No load
   profile currently justifies chasing the memory side further at this kernel's
   real call frequency; revisit only if a profile flags it as a real hot path.
> **STATUS CORRECTION for items 7-14 (audit M-07, fixed 2026-09-10).** These items
> read as landed wins, and for a long time the SHIPPED build ran none of them. Two
> different kernel families have to be kept apart to see why:
>
> - The **`simd`-package** kernels these items describe (`linalg/exp_simd.go`) are
>   Experimental tier, behind `GOEXPERIMENT=simd`. They are NOT in a default build
>   and never were — aikit is a library and cannot require that flag of importers
>   (see `docs/task-archsimd-eval.md`). The ratios below are real and still apply
>   to that build.
> - The **hand-written NEON/AVX2 contract** kernels (`exp_neon_arm64.s`,
>   `exp_avx2_amd64.s`) ARE in every build, behind the `*ContractInto` API.
>
> `encoder/` and `vision/` called the `*Into` family, which resolves to the scalar
> loop off `GOEXPERIMENT=simd` — so every softmax, SiLU, GELU and GELU-tanh in a
> shipped build was scalar, while this document described them as accelerated.
> They were routed to `*ContractInto` on 2026-09-10 (encoder -15.9%, SigLIP tower
> -28.7 to -42.7%). The default build now gets the hand-written kernels; the
> numbers in items 7-14 remain `simd`-package numbers and should be quoted as such.

7. **`SoftmaxRowInto` vectorized via Go 1.27's `simd` package — ✅ DONE
   (item 13, "SIMD expF32"), Experimental tier, `linalg/exp_simd.go`.**
   **Read this framing before quoting a number from this item — it has been a
   point of confusion once already:** every ratio below compares two ways of
   running the SAME math on the SAME CPU core — today's scalar Go arithmetic
   vs. Go 1.27's `simd` package compiling to vector CPU instructions (NEON on
   arm64, AVX2 on amd64). **It has nothing to do with this repo's GPU
   (CUDA/Metal) kernels** — those are `gpu/*`, a completely separate code
   path measured in `gpu`'s own docs. If this item's numbers ever make it
   into a release note, state that explicitly, the way this entry does.

   Landed after validating an uncompiled prototype
   (`~/tmcode/go127-simd-audit`, 2026-08-20 — read the go1.27.0 `src/simd`
   source directly since the authoring container had no 1.27 toolchain) by
   actually compiling and benchmarking it on both boxes before touching
   `linalg`. Measured (production `BenchmarkSoftmaxRow_vs_scalar`, same
   function both builds, only the build tag differs):

   | box | width | scalar (today) | SIMD | speedup |
   |---|---|--:|--:|--:|
   | `apple-m1pro` (NEON) | 128-bit (native) | ~5.2-5.3 ns/elem | ~2.08-2.10 ns/elem | **~2.5x** |
   | `nvidia-rtx2070s` (AVX2) | 128-bit (default) | ~6.5-6.8 ns/elem | ~2.06-2.2 ns/elem | **~3.0-3.2x** |
   | `nvidia-rtx2070s` (AVX2) | 256-bit (opt-in, `GODEBUG=simd='+256'`) | ~4.7-4.9 ns/elem | ~0.56-0.68 ns/elem (exp only, isolated) | **~6.9-8.5x** |

   Go 1.27's `simd` package defaults conservatively to 128-bit even on AVX2
   hardware; the wider width needs an explicit runtime opt-in past a safety
   check (`GODEBUG=simd='+256'`) — correctness held there too, but a library
   can't force this on its consumers unilaterally, so it stays a documented
   knob rather than a default. Scope: only `SoftmaxRowInto` (real callers:
   `encoder`, `vision`) — `ExpF32Into` has zero production callers as of this
   writing and was deliberately NOT vectorized, since doing so correctly
   would mean replicating `ExpF32`'s NaN/overflow/underflow guards via SIMD
   masks for a function nobody calls; the validated prototype and this
   landing both use `expF32Core`'s narrower "already-bounded" contract,
   which is exactly `SoftmaxRowInto`'s real shape.

   NOT bit-identical to the non-experimental build (the vector exp differs
   by up to 1 ULP — FMA contraction; `TestExpF32CoreVec_matchesScalarULP`
   gates the bound) — gated behind `GOEXPERIMENT=simd`, off by default, so
   the default build is byte-for-byte what shipped before this landed
   (verified: full `linalg`/`encoder`/`vision` suites pass unchanged).
   Correctness verified on both boxes, both real hardware and forced
   emulation (`GODEBUG=simd=0`, runs on any box regardless of ISA) —
   `encoder`'s and `vision`'s golden/cosine tests also pass under
   `GOEXPERIMENT=simd`, confirming the ≤1 ULP shift doesn't reach them. CI
   coverage: a dedicated `simd` job (`.github/workflows/ci.yml`), both
   arm64 and amd64 runners, since this is the ONLY place the experimental
   path gets exercised — nothing else needed updating (`tools/gate`,
   `consumergate`, the release gates all operate on the module graph or the
   default build, neither of which this touches).

   Next: `A3`-`A6` siblings — `SiLUF32` done (item 10); `GELUF32`, `TanhF32`,
   `GELUTanhF32` not started — and goinfer's parity ladder
   (`queue-performance.md` items this unblocks for adoption).
8. **RoPE rotation vectorized via Go 1.27's `simd` package — ✅ DONE, both
   sites, Experimental tier.** A follow-up sweep of the rest of the
   codebase (beyond item 7's `linalg/exp.go` scope) for elementwise math
   the original audit missed found aikit's own RoPE (two independent
   implementations, both NeoX rotate_half, neither transcendental — pure
   `x*cos - y*sin` / `y*cos + x*sin`, so unlike item 7 there is no new
   minimax polynomial involved, just vectorizing arithmetic that already
   existed): `vision/qwen_encoder.go`'s `applyRotaryVision` (Qwen2-VL
   vision tower, own comment cites "~8k patches × 16 heads × 32 blocks"
   per image) and `encoder/rope.go`'s `rotateHalfInto` (text encoders,
   called per attention layer). Landed as
   `vision/rope_simd.go`/`rope_scalar.go` and
   `encoder/rope_simd.go`/`rope_scalar.go`, same build-tag dispatch shape
   as item 7.

   Measured (`BenchmarkApplyRotaryVision`/`BenchmarkRotateHalfInto`, same
   function both builds):

   | box | kernel | scalar | SIMD | speedup |
   |---|---|--:|--:|--:|
   | `apple-m1pro` | applyRotaryVision (headDim=128) | 0.534 ns/elem | 0.284 ns/elem | ~1.9x |
   | `apple-m1pro` | rotateHalfInto (half=32) | 0.401 ns/elem | 0.267 ns/elem | ~1.5x |
   | `nvidia-rtx2070s` | applyRotaryVision | 0.603 ns/elem | 0.336 ns/elem | ~1.8x |
   | `nvidia-rtx2070s` | rotateHalfInto | 0.532 ns/elem | 0.334 ns/elem | ~1.6x |

   Smaller than item 7's ~2.5-3.2x — these are tiny, cheap-per-element
   calls (no polynomial, just a handful of multiply/add/subtract), so
   fixed per-call overhead eats a bigger share of the win. Still real on
   both boxes.

   **Accuracy note, worth stating precisely because it differs from item
   7**: this is pure `Mul`/`Sub`/`Add` — no `MulAdd`, so no FMA
   contraction is *requested*. Measured anyway rather than assumed
   bit-identical (`TestApplyRotaryVision_matchesScalar`,
   `TestRotateHalfInto_matchesScalar`): **bit-identical (0 diff) on
   `nvidia-rtx2070s`**, but **~2.4e-7 max abs diff (~2 ULP-class) on
   `apple-m1pro`** — Go's arm64 scalar compiler auto-fuses `a*b - c*d`
   shapes into a real hardware FMA where amd64's does not (same asymmetry
   item 7's doc noted for `p*r+c`), so the scalar REFERENCE itself already
   differs by architecture; the SIMD build's divergence from IT differs
   correspondingly. `encoder`/`vision` golden and cosine-parity tests pass
   unchanged under `GOEXPERIMENT=simd` on both boxes.

   **Two related candidates from the same sweep, ruled out — not a
   priority call, a hard API gap:** `embed/model.go`'s `encodeIDs`
   (Model2Vec weighted-mean pooling, "77.8% of an index run") and
   `embed/pool.go`'s `L2Normalize` both carry an explicit, tested
   precision contract — sum-of-squares/weighted-sum accumulates in
   float64, each float32 element widened before the multiply-accumulate,
   narrowed to float32 only at the very end ("accumulating in float32
   silently drifts cosine below the 1−1e-5 parity bar... do not
   'optimize' it away," `embed/model.go`). **Go 1.27's `simd` package has
   no float32↔float64 conversion at all** — not even a widen — so this
   contract cannot be expressed with it, full stop. Not attempted; would
   need either a different technique entirely or waiting for a future Go
   release.

9. **SPLADE's `log1p` pooling — ✅ DONE, a new kernel, Experimental
   tier.** The fourth candidate from item 8's sweep, and genuinely
   different scope from items 7/8: `encoder/splade.go`'s pooling applies
   `float32(math.Log1p(float64(x)))` once per vocab entry post-max
   (perf-campaign item 2, already V=30522 calls, down from millions).
   Unlike items 7/8, aikit had no existing `Log1pF32` to vectorize, and
   Go 1.27's `simd` package ships zero transcendentals — this meant
   designing a new float32 kernel, not vectorizing an existing one.

   **Sourced from Cephes' verified single-precision `logf`** (Moshier,
   `single/logf.c`), not invented: small x (< 2^-12) uses
   `log1p(x) ≈ x - 0.5x²` directly (the dropped x³/3 term is empirically
   below float32 precision there); larger x computes `u = 1+x` (safe —
   x is never tiny in this branch, so no `1+x` cancellation), decomposes
   u's IEEE-754 bits into a frexp-equivalent mantissa/exponent, reduces
   into Cephes' exact domain (threshold `sqrt(2)/2`), and evaluates its
   verified 9-coefficient polynomial. Cross-checked before writing any
   code: the reduction bounds independently match Go's own
   `src/math/log1p.go` (`Sqrt2M1`/`Sqrt2HalfM1` = Cephes' `SQRTHF`
   exactly) — two independent sources agreeing on the same constants.
   Validated with a throwaway sweep script *before* touching the repo:
   max absolute error **2.97e-7** vs `math.Log1p` over x ∈ [0, 1e10]
   (`TestLog1pF32Core_matchesMathLog1p`, bound set at 1e-6 with headroom
   — comparable to `GELUF32`'s existing 1e-6 absolute bound). The vector
   kernel (`TestLog1pF32CoreVec_matchesScalar`) came back **bit-identical
   (0 diff)** to the scalar form of the same algorithm on `apple-m1pro`
   — unlike items 7/8's polynomial, this one's `p = p*m + c` reassignment
   shape apparently doesn't trigger arm64's scalar auto-fusion.

   `pooled[v]` is seeded at 0 and only ever raised by `max`, so it is
   never negative, and `log1pF32Core(0) == 0` exactly — the SIMD kernel
   applies unconditionally to every lane (`TestLog1pPoolInto_zeroIsIdentity`),
   no per-element mask needed for the scalar build's `x > 0` skip, which
   was always a call-count optimization, not a correctness requirement.
   `TestSPLADE_parity` (real cosine-vs-Python comparison) stays at
   **1.000000 unchanged** under `GOEXPERIMENT=simd` — the ~3e-7 error
   doesn't move it at all.

   Measured (`BenchmarkLog1pPoolInto`, V=30522, ~15% positive density):
   `apple-m1pro` 2.193 → 1.595 ns/elem (**~1.37x**), `nvidia-rtx2070s`
   3.354 → 2.323 ns/elem (**~1.44x**) — smaller than items 7/8 on both
   boxes because the vector kernel computes BOTH branches
   unconditionally for every lane and mask-selects, paying the full
   large-branch cost (bit manipulation + 9-term polynomial) even for the
   ~85% of lanes that are 0 and would have cost nothing in the scalar
   build's `x > 0` skip. Real on both boxes, but the most modest win of
   the three landed kernels — reported honestly rather than rounded up.
   Also bit-identical vector-vs-scalar on `nvidia-rtx2070s`, matching
   `apple-m1pro` (not the arch-dependent split items 7/8 measured).

10. **`SiLUInto` vectorized via Go 1.27's `simd` package — ✅ DONE, Experimental
    tier, `linalg/exp_simd.go`.** First of the `A3`-`A6` siblings item 7 left
    "next, not started" (`docs/prompts/simd-elementwise-autoresearch.md`'s T1,
    round 1) — chosen first because, unlike GELU/Erf/Tanh's two-branch
    cancellation shape, SiLU's only extra work is the guards, and it already
    has a real caller (`encoder/linalg.go`'s `silu`) that justifies building
    them.

    **The guards ARE the work.** `x/(1+e^-x)` feeds exp an UNBOUNDED
    argument — unlike item 7's `SoftmaxRowInto`, whose domain after
    subtracting the row max is always `x <= 0`, so `expF32CoreVec` could
    stay narrow and skip `ExpF32`'s overflow/NaN guards (that item's own
    entry says so explicitly: "`ExpF32Into` has zero production callers...
    deliberately NOT vectorized"). SiLU is the caller that changes that
    calculus. Two things had to be built, both new:
    - `expF32Vec` — the full vector `ExpF32`, wrapping `expF32CoreVec` with
      overflow (→ `+Inf`, via `IfElse`) and NaN (→ NaN, via the `x != x`
      IEEE identity, also `IfElse`) — the two-branch compute-both-and-select
      pattern this file already uses for Erf/Tanh, just for guards instead
      of algorithm branches.
    - `expF32CoreVec` itself needed a real fix, not just a wrapper: its
      e>=255 two-step scale (the boundary correction scalar `expF32Core` has
      for when `k` reaches 128 while the true result is still finite — `2^k`
      built in one step would encode `+Inf` and poison a finite answer) was
      never replicated, because softmax's `x <= 0` domain can never reach
      it. SiLU's `e^-x` does reach it (`x` near `-88`). Extended in place
      (same function, same existing `TestExpF32CoreVec_matchesScalarULP`
      gate, which never exercised the new branch since it's still
      unreachable for `x <= 0` — fresh coverage added specifically for the
      newly-reachable boundary: `TestExpF32Vec_matchesScalarULP`).

    SiLU's own saturating ends (per `SiLUF32`'s doc comment) fall out of
    `expF32Vec`'s guards with no additional branch: very negative `x` makes
    `e^-x = +Inf`, and IEEE division gives `x/(1+Inf)` a signed zero; very
    positive `x` flushes `e^-x` to 0 (already handled) and the result is
    exactly `x`.

    Measured (`BenchmarkSiLU_vs_scalar`, same `SiLUInto`, only the build tag
    differs — compared against the CURRENT shipped scalar path, not the
    `math.Exp` strawman the benchmark also carries for scale):

    | box | scalar (today) | SIMD | speedup |
    |---|--:|--:|--:|
    | `apple-m1pro` (NEON) | 3.86 ns/elem | 1.34 ns/elem | **~2.9x** |
    | `nvidia-rtx2070s` (AVX2, 128-bit default) | 7.48 ns/elem | 2.61 ns/elem | **~2.86x** |

    Both boxes land within 2% of each other — unusually tight agreement for
    this doc's SIMD entries, likely because the two new guard branches
    (`IfElse`-based, not a polynomial) cost about the same fraction of the
    kernel on both ISAs. Up to 2 ULP vector-vs-scalar drift
    (`TestSiLUIntoRaw_matchesScalar`, within `SiLUF32`'s own 4 ULP contract)
    — NOT bit-identical to the non-experimental build, same class of
    difference as items 7-9. Full gate green on both boxes: accuracy/ULP
    tests, `encoder`+`vision` golden/cosine suites under
    `GOEXPERIMENT=simd`, the `GODEBUG=simd=0` emulation leg, `-race`, and
    the default (non-experimental) build verified byte-for-byte unchanged
    (full `linalg` suite passes with zero diff).

    Next: `GELUF32`/`ErfF32` and `TanhF32`/`GELUTanhF32` (T1's remaining two
    targets — both two-branch cancellation kernels, a different shape from
    SiLU's guards-only case) — autoresearch loop round 2.

11. **`GELUInto`/`ErfF32` vectorized via Go 1.27's `simd` package — ✅ DONE,
    Experimental tier, `linalg/exp_simd.go`.** Second of the `A3`-`A6`
    siblings (autoresearch round 2), and a genuinely different shape from
    item 10's SiLU: Erf is a TWO-BRANCH kernel (series for `|x|<1`, no
    cancellation; the A&S 7.1.26 tail for `|x|>=1`, stable because
    `1-(small)` doesn't cancel there), so the vector form computes BOTH
    branches unconditionally and mask-selects with `IfElse` — exactly the
    pattern this file's SiLU entry named as the alternative shape, now
    built.

    New `erfVec`, its own `erfSIMDConsts` (kept separate from
    `expSIMDConsts` — Erf's ~20 extra broadcasts have no reason to ride
    softmax's or SiLU's hot path). The tail branch's `exp(-x²)` reuses the
    NARROW `expF32CoreVec` (not item 10's full-range `expF32Vec`): for the
    `|x|` in `[1,4]` the tail branch ever runs at (`|x|>4` saturates to ±1
    separately, before reaching it), `-x²` is always in `[-16,-1]` —
    comfortably inside `expF32CoreVec`'s validated domain, no overflow risk,
    same reasoning softmax's own narrow reuse already established. NaN is
    NOT specially guarded here (unlike item 10's guard for SiLU's real,
    NaN-reachable caller): GELU feeds on a forward pass's own activations,
    no production input has ever been NaN, and the existing scalar `ErfF32`
    has no stated or tested NaN contract to preserve — this vectorizes the
    domain the scalar function and its real callers actually see, not a
    hypothetical wider one.

    Measured (`BenchmarkGELU_vs_mathErf`, same `GELUInto`, only the build
    tag differs — the CURRENT shipped scalar path, not the `math.Erf`
    strawman the benchmark also carries for scale):

    | box | scalar (today) | SIMD | speedup |
    |---|--:|--:|--:|
    | `apple-m1pro` (NEON) | 12.97 ns/elem | 2.50 ns/elem | **~5.2x** |
    | `nvidia-rtx2070s` (AVX2, 128-bit default) | 16.64 ns/elem | 3.92 ns/elem | **~4.25x** |

    The biggest speedup of any item in this doc so far — consistent with
    the two-branch premise: the scalar kernel already pays for one branch's
    worth of work per call, but the vector kernel pays for BOTH branches on
    every lane regardless of which one a given element "needs", and the
    branches here (an 11-term polynomial vs. a division-heavy 5-term one
    plus an exp call) are each substantial, so lane-parallelism buys more
    than it did for SiLU's single-branch-plus-guards shape. Up to 2.4e-07
    max absolute vector-vs-scalar drift (`TestErfVec_matchesScalar`,
    `TestGELUIntoRaw_matchesScalar` — both within `GELUF32`'s own 1e-6
    absolute contract) — NOT bit-identical to the non-experimental build,
    same class of difference as items 7-10. Full gate green on both boxes:
    accuracy/ULP-class tests, `encoder`+`vision` golden/cosine suites under
    `GOEXPERIMENT=simd`, the `GODEBUG=simd=0` emulation leg, `-race`, and
    the default (non-experimental) build verified byte-for-byte unchanged.

    Next: `TanhF32`/`GELUTanhF32` (T1's last target, same two-branch
    cancellation shape as Erf) — autoresearch loop round 3.

12. **`TanhInto`/`GELUTanhInto` vectorized via Go 1.27's `simd` package — ✅
    DONE, Experimental tier, `linalg/exp_simd.go`. T1 (the `A3`-`A6`
    siblings) is now COMPLETE.** Third and last of the round-3 targets, same
    two-branch shape as item 11's Erf: Cephes' minimax polynomial for
    `|x|<0.625` (no cancellation there), the exponential form
    `1-2/(e^2x+1)` for `|x|>=0.625` (stable — subtracts at most 0.445 from
    1). New `tanhVec`, its own `tanhSIMDConsts`. The exponential branch's
    `exp(2·|x|)` again reuses the NARROW `expF32CoreVec`: `|x|` is bounded
    to `[0.625,9]` in that branch (`|x|>9` saturates separately, same
    saturating-tail shape as Erf's `|x|>4`), so `2·|x| ∈ [1.25,18]` never
    approaches the overflow/underflow boundaries — the third kernel in a
    row to use this reuse, after softmax (item 7) and Erf's tail (item 11).
    `GELUTanhInto` (the actual SigLIP/Gemma-3-vision-tower activation
    caller) is built directly on `tanhVec` with no guard work of its own,
    same relationship item 10's SiLU has to `expF32Vec`.

    Both `TanhInto` and `GELUTanhInto` have real callers
    (`encoder/crossencoder.go`'s pooling for the former,
    `encoder/bert.go`'s tanh-GELU chunking for the latter) — checked before
    building, per this doc's own item-7 precedent of only vectorizing
    functions something actually calls.

    Measured (`BenchmarkTanh_vs_scalar`/`BenchmarkGELUTanh_vs_scalar`, same
    functions both builds, only the build tag differs):

    | box | kernel | scalar (today) | SIMD | speedup |
    |---|---|--:|--:|--:|
    | `apple-m1pro` | TanhInto | 9.97 ns/elem | 1.94 ns/elem | **~5.1x** |
    | `apple-m1pro` | GELUTanhInto | 15.03 ns/elem | 2.68 ns/elem | **~5.6x** |
    | `nvidia-rtx2070s` | TanhInto | 12.44 ns/elem | 3.45 ns/elem | **~3.6x** |
    | `nvidia-rtx2070s` | GELUTanhInto | 17.22 ns/elem | 4.02 ns/elem | **~4.28x** |

    Same two-branch-pays-for-both-lanes premise as item 11, and the numbers
    confirm it again: large wins on both boxes, `apple-m1pro` consistently
    ahead of `nvidia-rtx2070s` for this whole two-branch family (items
    11-12), unlike item 10's SiLU where the two boxes landed within 2% of
    each other. Up to 4 ULP / 2.38e-07 abs vector-vs-scalar drift
    (`TestTanhVec_matchesScalar`, `TestGELUTanhIntoRaw_matchesScalar` —
    both within the scalar kernels' own contracts). Full gate green on both
    boxes: accuracy/ULP-class tests, `encoder`+`vision` golden/cosine
    suites under `GOEXPERIMENT=simd`, the `GODEBUG=simd=0` emulation leg,
    `-race`, and the default (non-experimental) build verified
    byte-for-byte unchanged.

    **T1 is now fully landed** (items 10-12: SiLU, GELU/Erf, Tanh/GELUTanh
    — all four `A3`-`A6` siblings item 7 named). Next per
    `docs/prompts/simd-elementwise-autoresearch.md`'s own ranking: T2, the
    softmax-scale fusion dead-ends §4.4 reopens now that item 7's SIMD
    softmax makes the scale pass a bigger share of the kernel.

13. **Softmax-scale fusion — ✅ DONE, Experimental tier,
    `linalg/exp_simd.go`.** T2, and dead-ends §4.4's own condition for
    reopening it ("revisit after campaign #13 lands SIMD `expF32` — then it
    becomes ~20% of the softmax instead of 2%") is exactly what items 7-12
    did. §4.4 originally measured the fusion worth nothing (16.0 vs
    16.0 ns/elem at L=80) because scalar `math.Exp` swamped the ~2% the
    separate `scores[i] *= scale` pass cost — that math hasn't changed for
    the DEFAULT build, only for the experiment, so this stays scoped there.

    New `linalg.SoftmaxRowScaledInto(dst, src, scale)` — `softmax(scale·src)`
    in one call. scale MUST be `> 0` (attention's `1/sqrt(headDim)` always
    is): that's what lets the row max be computed on the UNSCALED row (an
    identical max-scan to the unfused kernel) and multiplied by `scale`
    exactly once, rather than needing a max-scan over pre-scaled values —
    `max(scale·x) == scale·max(x)` for `scale > 0`, and that one multiply is
    the EXACT SAME float32 operation whether performed as part of a
    separate full-array pass first or once on the single max value after
    scanning. Same argument for `scaled[i] == src[i]*scale`: the same
    single multiply, same operands, same rounding, regardless of when it
    runs. That's what makes the fusion **provably bit-identical**, not
    merely close — verified directly
    (`TestSoftmaxRowScaledInto_bitIdenticalToTwoPass`,
    `TestSoftmaxRowScaledIntoRaw_tail`,
    `encoder`'s `TestSoftmaxRowsScaled_bitIdentical`), not just argued.

    Default build: still two passes internally (`softmaxRowScaledIntoRaw`
    in `exp_scalar.go` scales, then calls the existing
    `softmaxRowIntoRaw`) — matching §4.4's own finding that fusing them
    isn't worth it at scalar speeds, so nothing there was worth changing.
    SIMD build: the `src[i] *= scale` a caller used to run as its own
    separate O(L²) pass is folded directly into the SAME per-lane pass that
    already computes the exponential (`src[i]*scale - scaledMax`, vectorized)
    — three passes instead of four, eliminating the separate multiply pass
    entirely rather than merely speeding it up.

    Encoder side: all four real call sites that used to do
    `for i { scores[i] *= scale }` then `softmaxRows(...)` —
    `encoder/attention.go`, `encoder/attention_batch.go`, and both of
    `encoder/forward_q8.go`'s (single + batched) — now call a new
    `softmaxRowsScaled(scores, scale, rows, cols)` (`encoder/parallel.go`,
    same row-parallel-split shape as `softmaxRows`) instead. The unscaled
    `softmaxRows`/`softmaxRow` are UNCHANGED and kept (no remaining
    production caller, but real existing test coverage
    (`TestSoftmaxRows_bitIdentical`) — left alone rather than deleted, and
    `softmaxRowsScaled` got the same parallel-split correctness gate
    (`TestSoftmaxRowsScaled_bitIdentical`) rather than trusting the shared
    `parallelRows` machinery by inference.

    Measured (`BenchmarkSoftmaxRowScaled_vs_twoPass`, the actual two-pass
    sequence a caller used to run vs. the fused call, at the two shapes
    §4.4 named):

    | box | build | L=80 | L=691 |
    |---|---|--:|--:|
    | `apple-m1pro` | SIMD | 2.92→2.38 ns/elem (**~18.5%**) | 2.88→2.45 ns/elem (**~15%**) |
    | `apple-m1pro` | default | 5.24→4.92 ns/elem (noise) | 5.61→5.51 ns/elem (noise) |
    | `nvidia-rtx2070s` | SIMD | 3.66→3.09 ns/elem (**~15.3%**) | 3.57→2.95 ns/elem (**~17.4%**) |
    | `nvidia-rtx2070s` | default | 7.20→6.85 ns/elem (noise) | 7.08→6.70 ns/elem (noise) |

    A real, reproducible ~15-19% win under `GOEXPERIMENT=simd` on both
    boxes — smaller than items 10-12's multi-x kernel wins (this eliminates
    one O(L²) pass out of softmax's four, not a whole transcendental), but
    it lands almost exactly where §4.4's own prediction put it ("~20% of
    the softmax"), and it's a smaller, more surgical class of change: no
    new transcendental kernel, just removing a pass whose separate
    existence stopped being justified once the pass it fed got cheap. The
    default build shows only noise-level movement (<6%, inside
    measuring-performance's own "treat as unmeasured" band) — expected and
    correct, since it isn't actually fused there. Full gate green on both
    boxes: bit-identity tests (not just accuracy bounds — this technique's
    whole premise is exactness), `encoder`+`vision` golden/cosine suites
    under `GOEXPERIMENT=simd`, the `GODEBUG=simd=0` emulation leg, `-race`,
    and the default build's full test suite (`linalg`+`encoder`) verified
    unchanged.

    Next per `docs/prompts/simd-elementwise-autoresearch.md`: re-profile
    the L=690 and L=80 forwards now that T1+T2 have both landed — the
    doc's own stop condition triggers here if the elementwise layer's share
    has fallen under ~2-3% of the forward.

14. **Wiring fix: `encoder`'s real hot paths never called the batched
    kernels items 10-12 vectorized — ✅ FOUND AND FIXED, correctness-only,
    no new kernel.** Found while re-profiling for T2's stop-condition
    check (autoresearch loop, immediately after T2): `linalg.SiLUInto`/
    `GELUInto`/`GELUTanhInto` DO have real production callers — but they
    were `vision/qwen_encoder.go` and `vision/encoder.go` (the SigLIP/
    Qwen2-VL towers), NOT `encoder`'s main text-forward path. `encoder`'s
    own biggest activation call sites — `mlp.go`'s `swigluMLP` and
    `swigluMLPQ8` (`val[i] = v * silu(gate[i])`), `bert.go`'s `gelu`/
    `geluTanh` (chunked per-element loops), and `gte.go`'s GeGLU (the
    `~27% of GTE.Encode at L=690` hotspot item 8's own doc cites) — all
    called the SCALAR single-element `SiLUF32`/`GELUF32`/`GELUTanhF32`
    directly, never the batched `Into` functions. **Items 10-12's landed
    SIMD kernels were real, tested, and measured — but genuinely
    unreachable from `encoder`'s own main forward pass until this fix.**

    A `cpu-acceleration.md` profile (`go tool pprof -top`,
    `BenchmarkEncode_singleLong`, `apple-m1pro`) before this fix confirms
    it directly: zero samples in any `*Vec`/`*IntoRaw` SIMD symbol
    anywhere in the trace — only scalar `expF32Core`/`ExpF32`/`SiLUF32`.

    Fixed by rewiring each site to call the batched function on its
    existing chunk/row slice (in place — `GELUInto(x[lo:hi], x[lo:hi])`
    etc.) instead of a per-element scalar loop, keeping every existing
    core-level `parallelRows` split exactly as it was (SIMD now layers
    UNDER the existing multi-core parallelism, not instead of it):
    `bert.go`'s `gelu`/`geluTanh`, `mlp.go`'s `swigluMLP`, `forward_q8.go`'s
    `swigluMLPQ8`, `gte.go`'s GeGLU loop. `geluScalar` (bert.go) had zero
    callers left anywhere (production or test) after the fix and was
    deleted; `silu`/`softmaxRows`(unscaled) were left in place — both have
    real, independent test coverage of their own even though `silu` has no
    remaining production caller either.

    Re-profiling after the fix (same `BenchmarkEncode_singleLong` on
    `apple-m1pro`, plus `BenchmarkGTEEncode/L512` for the GeGLU
    architecture) now shows the SIMD symbols present and load-bearing
    (`siluIntoRaw@simd128`, `softmaxRowScaledIntoRaw@simd128`,
    `geluIntoRaw@simd128`, `erfVec@simd128`), with the matmul kernels
    (`dotNEON2x8`/`blockedFill`) still dominant at 54-69% — the
    elementwise layer's combined share (SiLU/GELU + the T2 softmax-scale
    fusion) landed at **~1.2-1.9% of the forward on both the SwiGLU-model
    and GeGLU (GTE) architectures** — under the autoresearch doc's own
    2-3% stop threshold. Full gate green on both boxes: the default build
    verified bit-identical (existing golden/cosine suites, unchanged), the
    `GOEXPERIMENT=simd` build's `encoder`+`vision` goldens pass, the
    `GODEBUG=simd=0` emulation leg, and `-race`.

    **This closes the autoresearch loop's T1+T2 stop condition.** Not
    because the technique stopped working — because it worked well enough,
    once actually wired up, that there is no longer enough elementwise
    share left to chase with T3-T5. See
    `docs/prompts/simd-elementwise-autoresearch.md`'s STATUS block for the
    stop record.

GPU follow-ups (resident buffers, tiled kernel, batch-tiling) are **goinfer's** —
see `goinfer/gpu` and goinfer's perf docs.

---

## Native K-quant matmul (Q4_K/Q6_K × Q8_K) — evaluated, NOT shipped

**Result: negative for the stated gate.** `docs/internal/archive/task-q8k-integer-accum.md` asked whether a
native integer-accumulation K-quant kernel (the cpubrrr / llama.cpp `ggml_vec_dot_q6_K_q8_K`
algorithm — quantize activations to Q8_K, accumulate sub-block int dot products weighted by
the integer sub-scales, convert to float once per 256-superblock) beats the current decode
path (dequant→int8-requant→`MatmulBTW8A8`, which at decode is just W8A8 over the resident
int8 weight) by **≥1.3× at M=1**. It cannot, for Q6_K — provably.

**What was built and validated** (kept in `linalg/kquant*.go`, Experimental tier, tested and
correct, but deliberately **not wired into `WeightMat`**): `QuantizeActQ8K` (per-256-block
int8 + f32 scale + exact per-16 bsums); `unpackQ6K`/`unpackQ4K`, bit-identical to
`embed/gguf.go`'s dequant (drift-guarded, `TestKQuantUnpackMatchesEmbed`); scalar
integer-accum dots; and an SDOT path (`dotPartials16SDOT`, one Go→asm crossing per superblock)
bit-exact against the scalar reference. So the arithmetic half works and is fast.

**Measured (Apple M-series, this box, `BenchmarkGEMV_*`, M=1):**

| shape | Q6_K native | W8A8 baseline | ratio |
|---|---|---|---|
| K2048 × N2048 | 2.21 ms | 0.111 ms | **20× slower** |
| K4096 × N4096 | 8.79 ms | 0.196 ms | **45× slower** |

The SDOT does the same MAC count as W8A8 (~0.2 ms); ~98% of the native time is the weight
unpack. A SIMD bit-unpack would cut that — but it cannot cross the gate, because there is a
hard ceiling above any unpack:

**The byte-ratio ceiling.** A native K-quant kernel's only advantage is reading fewer weight
bytes. So the speedup cannot exceed the byte-count ratio:
- If W8A8 is **bandwidth-bound**, Q6_K reads 210 B/superblock vs int8's 256 → at best
  256/210 = **1.22×**, and it still adds unpack+SDOT compute → ≤ 1.22×.
- If W8A8 is **compute-bound**, Q6_K does the *same* SDOT MACs *plus* the unpack → strictly
  more work → ≤ 1.0× (loses).

Either way **Q6_K ≤ 1.22× < 1.3×**: the gate is unreachable regardless of kernel quality, so
the SIMD-unpack asm was not written. Q4_K's ratio is 256/144 = **1.78×** (headroom exists),
but cpubrrr's *own* optimized Q4_K/Q6_K kernel landed **~1.12×** over llama.cpp in practice —
below 1.3× — so Q4_K was not pursued either. This confirms `task-native-q6k-kernel.md`'s
original caution on the last axis it had set aside (compute/throughput): the win is real but
thin, and below this project's bar for adding a native-GGUF weight path.

Where the cpubrrr win actually lives is **MXFP4 MoE (~5×)**, which is a different format and a
missing model family — tracked as goinfer's A2 (`task-mxfp4-gptoss.md`), not here.

Reproduce: `go test -bench GEMV -benchmem ./linalg`.

---

## File reference

```
linalg/dot_{arm64,amd64,generic,other}.{go,s}  dot kernels + build-tag dispatch
linalg/dot_i8*_arm64.s, dotprod_arm64_*.go      int8 NEON / SDOT (HWCAP-selected)
linalg/dot_w4a8_arm64.s, quant_w4a8*.go         fused int4×int8 decode kernel + scalar fallback
linalg/dot_amd64.go                             AVX2 dispatch + CPUID/XGETBV detect
linalg/linalg.go                                Dot*/MatmulBT + SetParallelThreshold/Width
linalg/quant.go                                 Q8/Q4/W8A8 matmuls (+ Into/Batch)
linalg/dequant_i8.go, dequant_i8_{arm64.s,amd64.go}  bulk int8→f32 widen (item 22a)
linalg/matmul_blocked.go, matmul_blocked_q8.go  packed f32 GEMM + fused-widen Q8 (22b)
linalg/workspace.go                             reusable scratch (pool.go was pulled — dead-ends §8.1)
linalg/{dot,dot_amd64,width,quant,batch}_test.go   kernel/parity/bench tests
encoder/linalg.go, linalg_q8.go                 encoder's cache-blocked matmul (uses linalg.Dot*)
encoder/parallel.go                             single-forward row-parallel (in-flight gate)
```
