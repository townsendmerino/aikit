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
| arm64 | `linalg/dot_arm64.{go,s}`, `dot_i8*_arm64.s`, `dotprod_arm64_*.go` | NEON; int8 `dotI8` upgrades to `SDOT` on DotProd-capable CPUs (runtime HWCAP) |
| amd64 | `linalg/dot_amd64.{go,s}` | AVX2+FMA (`dotFMA`/`dotFMA4`/`dotFMA8`), runtime CPUID/XGETBV detect, scalar fallback |
| other | `linalg/dot_generic.go`, `dot_other.go` | portable scalar |

On top of the dot kernels, `linalg` provides:
- `Dot`, `Dot4x4`, `Dot8x4`, `MatmulBT` (f32 row-parallel).
- The quant matmuls: `MatmulBTQ8` (int8 weights), `MatmulBTQ4` (int4 group), and
  the **W8A8** path (`MatmulBTW8A8` + the zero-alloc `…Into(ws *Workspace)` and
  the fused `MatmulBTW8A8Batch`) — see `quant.go`, `workspace.go`.
- Dispatch knobs: `SetParallelThreshold` (MAC count to parallelize above) and
  `SetParallelWidth` (cap fan-out shards, for P/E straggler control) — both
  numerically inert (output columns are partitioned). `pool.go` is the optional
  per-`Workspace` spin-then-park worker pool.

**2. `encoder/` — the encoder's own matmul orchestration.** `encoder/linalg.go`
is the cache-blocked `matmulBTInto` (calls `linalg.Dot8x4`/`Dot4x4`);
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

1. **`Dot8x4` large-K crossover heuristic.** `Dot8x4` wins at mid-K (~768) but
   loses to `Dot4x4`/single-row past it (the K=3072 regression above). The
   encoder's blocked matmul should pick the kernel by K (prefer `Dot4x4` / fall
   to single-row at large K). Kernels are correct; only the selection heuristic
   is unwired.
2. **AVX-512 path** (optional, Zen 4 / recent Intel). 16-wide, more registers;
   AVX2 already covers ~all amd64 since 2015 and AVX-512 brings downclocking
   caveats, so low priority. Same shape: CPUID leaf 7 detect, `dot_amd64.s`
   entry points, `hasAVX512` gate.
3. **Per-head attention QK^T parallelization** (encoder). ~17M FLOPs/head wins
   ~3.4× in isolation but recurs 12 heads × 12 layers/forward; needs an
   end-to-end `Model.Encode` benchmark on real weights (not a microbench) to
   decide whether the spawn overhead pays off. Tune the encoder's
   `parallelThreshold` accordingly.

GPU follow-ups (resident buffers, tiled kernel, batch-tiling) are **goinfer's** —
see `goinfer/gpu` and goinfer's perf docs.

---

## File reference

```
linalg/dot_{arm64,amd64,generic,other}.{go,s}  dot kernels + build-tag dispatch
linalg/dot_i8*_arm64.s, dotprod_arm64_*.go      int8 NEON / SDOT (HWCAP-selected)
linalg/dot_amd64.go                             AVX2 dispatch + CPUID/XGETBV detect
linalg/linalg.go                                Dot*/MatmulBT + SetParallelThreshold/Width
linalg/quant.go                                 Q8/Q4/W8A8 matmuls (+ Into/Batch)
linalg/workspace.go, pool.go                    reusable scratch + spin-park worker pool
linalg/{dot,dot_amd64,width,quant,batch,pool}_test.go   kernel/parity/bench tests
encoder/linalg.go, linalg_q8.go                 encoder's cache-blocked matmul (uses linalg.Dot*)
encoder/parallel.go                             single-forward row-parallel (in-flight gate)
```
