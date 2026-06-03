# CPU & GPU acceleration: testing & follow-ups

How to verify and extend the encoder's compute kernels — the amd64 AVX2 path, the
single-forward row-parallel matmul, and the optional WebGPU (Metal/Vulkan/DX12) backend.
Written for the case where you're developing on an Apple-Silicon (arm64) Mac but need to
validate the **amd64** work on a Linux box, since AVX2 can't run on arm64 (not even under
Rosetta 2, which tops out at SSE4.2).

> **Status note.** This doc is kept current as the follow-ups below land. Last updated for:
> Phase A (amd64 AVX2/FMA, first-cut) + Phase B (intra-op row-parallel matmul) +
> Phase 3 foundation (WebGPU matmul offload, `-tags gpu`) on the `main` branch.
>
> **2026-06-02 — amd64 AVX2 path validated on Linux for the first time.** Every
> `AVX2|Dot` test PASSes with `hasAVX2=true`, no SIGILL, and `-race` is clean.
> The single-row and both register-blocked kernels (`dotFMA`/`dotFMA4`/`dotFMA8`)
> bit-match the scalar reference on real AVX2 hardware. Numbers + one tuning
> finding in follow-up §A below. Box: AMD Ryzen 7 3700X (8C/16T, Zen 2), Go 1.26.3.

---

## What's in the encoder now

The transformer's hot inner loop is three arch-neutral kernel functions — `dotNEON`,
`dotNEON4x4`, `dotNEON8x4` — dispatched by build tag:

| Arch | File | Path |
|---|---|---|
| arm64 | `internal/linalg/dot_arm64.{go,s}` | hand-written NEON (M-series, pre-existing) |
| amd64 | `internal/linalg/dot_amd64.{go,s}` | AVX2+FMA `dotFMA`, runtime-detected, scalar fallback |
| other | `internal/linalg/dot_other.go` → `dot_generic.go` | scalar |

> **Moved (M7, 2026-06-02):** the dot kernels were lifted out of `encoder/` into
> `internal/linalg` so the encoder and the Gemma decoder share one copy of the
> assembly (gemma-decoder-plan §1). The encoder imports `linalg.Dot4x4`/`Dot8x4`;
> the decoder's matmul backend calls `linalg.MatmulBT`. See `milestones/M7-perf.md`.

On top of that, `encoder/parallel.go` row-splits the big linear-layer matmuls across cores
**only** when a single forward pass is running alone (a lone `Encode` call) — never under
`EncodeBatch`, which already saturates every core at the document level. The gate is an
atomic in-flight-forward counter; see the package comment for the rationale.

Two things to know before testing:

- **AVX2 detection is runtime.** `hasAVX2` is computed once at init via hand-rolled
  CPUID/XGETBV (no `golang.org/x/sys` dependency). If the CPU/OS lacks AVX2+FMA, the kernels
  silently use the scalar fallback — correct, just unaccelerated.
- **The parallel matmul is numerically exact** vs. the serial path (row-splitting doesn't
  change f32 reduction order), so it must match bit-for-bit, not just within tolerance.

---

## The GPU backend (`encoder/gpu`, `-tags gpu`)

Optional WebGPU compute backend (`encoder/gpu`), built **only** under `-tags gpu`. It links
`github.com/cogentcore/webgpu`, a cgo binding that bundles the wgpu-native Rust library, and
runs on Metal (macOS), Vulkan (Linux), or D3D12 (Windows). Every file except `doc.go` carries
`//go:build gpu`, so without the tag the package is empty and the module stays pure-Go /
no-cgo — `CGO_ENABLED=0 go build ./...` succeeds. `go mod tidy` keeps the dep (verified).

The foundation cut offloads a single `dst = a·bᵀ` matmul (upload → dispatch → readback) with a
naive one-invocation-per-output WGSL kernel. It's correct but **slower than the CPU** at every
forward shape — the expected outcome, and the reason the foundation exists is to quantify the
gap before investing in the resident-buffer follow-up.

**Measured on an M1 Pro (8-core), GFLOP/s, higher is better:**

| Shape           | GPU (naive, full transfer) | CPU serial (NEON) | CPU parallel |
|-----------------|---------------------------:|------------------:|-------------:|
| L80 fc11        | 32.8                       | 43.5              | 129.4        |
| L256 fc11       | 39.6                       | 44.7              | 194.9        |
| L512 fc11       | 42.2                       | 45.0              | 197.0        |
| L512 fc2        | 36.3                       | 39.8              | 174.3        |

Two compounding bottlenecks, both expected: (1) per-call host↔device transfer (each call
re-uploads a+b, reads back dst — ~95 allocs/op); (2) the naive kernel is memory-bound, so its
GFLOP/s *rises* with size as transfer amortizes but plateaus ~42 — the no-tiling ceiling. Even
with zero transfer this kernel would only *match* serial CPU. Beating CPU needs **both**
follow-ups below.

### Testing the GPU backend (needs a GPU; works on your Mac's Metal)

```bash
go test -tags gpu ./encoder/gpu/ -run MatmulBT -v
#   expect "GPU backend: metal" (or vulkan/d3d12) and all PASS, including the
#   ragged shapes (17×65×33 etc.) that exercise the kernel's bounds check.
go test -tags gpu ./encoder/gpu/ -run XXX -bench MatmulBT_GPU -benchmem
#   compare GFLOP/s (MB/s ÷ 1000) against encoder's BenchmarkMatmul{Serial,Parallel}_*.
```
On a headless box with no GPU, `New()` errors and the tests **skip** cleanly. Confirm the
default build is unaffected: `CGO_ENABLED=0 go build ./...`.

---

## Testing on the Linux box (amd64)

```bash
git pull          # or: git clone <repo> && cd aikit
go version        # confirm Go 1.26+
```

### 1. The AVX2 differential test — the main thing to verify

```bash
go test ./encoder/ -run 'AVX2|Dot' -v
```

What to look for:

- `TestAVX2_dotFMA_matchesGeneric` — **PASS**. This directly compares the AVX2 asm kernel
  against the scalar kernel across all tail sizes (the 32-wide body, the 8-wide tail, and the
  scalar remainder). This is the real proof the assembly is correct.
- `TestAVX2_detection` logs `hasAVX2=true (maxLeaf=…)`. If it prints `hasAVX2=false`, that
  box's CPU/OS doesn't expose AVX2 — the asm test will have **SKIPPED**, and you're only
  testing the scalar path. Use a box with AVX2 (anything Haswell-era 2013+ / most cloud VMs).
- `TestDotF32_matchesScalar`, `TestDot4x4_matchesScalar`, `TestDot8x4_matchesScalar` — PASS.
  On an AVX2 box these now exercise the asm path against an independent scalar reference.

**Failure mode that matters:** if any test **SIGILLs** (illegal instruction) rather than
fails an assertion, that's an assembler/detection bug — the dispatch picked the asm path on a
CPU that can't run it, or an instruction is encoded wrong. Capture the exact test name and the
output of `go test -run TestAVX2_detection -v` and `cat /proc/cpuinfo | grep flags | head -1`.

### 2. Full suite, with the race detector

```bash
go test -race ./encoder/...
```

`-race` is the important flag for the parallel matmul (`parallel.go`) — it confirms the
row-splitting goroutines have no data races. Model-dependent tests skip cleanly when the
CodeRankEmbed weights aren't present (see the repo README for `ken download-model --rerank`);
a green run with those skipped is the expected state.

### 3. The payoff — benchmarks

**AVX2 vs scalar** (the per-core SIMD win):

```bash
go test ./encoder/ -run XXX -bench 'Dot' -benchmem
```

Compare `BenchmarkDot8x4_K768` / `BenchmarkDotF32_K768` (dispatched kernel, = AVX2 on this box)
against `BenchmarkDotGo_K768` (the scalar baseline). Expect a multiple-× throughput gain in the
MB/s column (which here reports as GFLOP/s — each element is 2 FLOPs). K64 / K768 / K3072 cover
the attention and linear-layer reduction widths.

**Row-parallel vs serial matmul** (the single-forward latency win):

```bash
go test ./encoder/ -run XXX -bench 'MatmulParallel|MatmulSerial' -benchmem
```

For reference, on an 8-core M1 Pro: L256 fc11 was 4.3×, L512 fc11 4.5×, L80 outproj 2.6×.
Your amd64 numbers will differ with core count and AVX2 throughput — record them; they're the
data for tuning `parallelThreshold` (see follow-ups).

### 4. Cross-check the build the way CI would

```bash
go vet ./encoder/          # asmdecl validates every asm stack offset vs. the Go signatures
go build ./...
```

---

## Follow-ups you can do from the Linux box

These are the deliberately-deferred pieces — each needs real amd64 hardware (or model weights)
to measure, which is exactly what the Linux box gives you. Roughly in priority order.

### A. amd64 register-blocked micro-kernel — DONE & VALIDATED on Linux (2026-06-02)

**Implemented and now executed.** `dotFMA4` / `dotFMA8` in `internal/linalg/dot_amd64.s` load each 8-float
chunk of the shared `a` row ONCE and run 4/8 FMA chains against the b-rows into separate YMM
accumulators — the a-reuse register-blocking arm64's NEON kernel uses. `dotNEON4x4`/`dotNEON8x4` in
`dot_amd64.go` dispatch to them (scalar fallback when `!hasAVX2`). Cross-compiled + `go vet`/
`asmdecl`-clean on the arm64 host; **first real execution was on the Ryzen 7 3700X box below.**

**Correctness — all PASS, no SIGILL, `-race` clean.** `TestAVX2_dotFMA_matchesGeneric`,
`TestDotF32/Dot4x4/Dot8x4_matchesScalar` all bit-match the independent scalar reference;
`TestAVX2_detection` logs `hasAVX2=true (maxLeaf=16)`. The register-blocked asm is correct on real
AVX2 hardware.

**Speedup** (`-bench 'Dot' -benchmem`, Ryzen 7 3700X, Zen 2, Go 1.26.3; MB/s column):

| K     | scalar `DotGo` | single-row `dotFMA` |       Dot4x4 |       Dot8x4 |
|-------|---------------:|--------------------:|-------------:|-------------:|
| 64    |   7.4 GB/s     | 14.3 GB/s (1.9×)    |  30.8 GB/s   |  35.3 GB/s   |
| 768   |   8.1 GB/s     | 44.2 GB/s (5.5×)    |  51.9 GB/s   | **86.5 GB/s**|
| 3072  |   8.4 GB/s     | 49.9 GB/s (5.9×)    |  51.4 GB/s   |  40.5 GB/s   |

Single-row AVX2 lands the expected ~6× over scalar at the linear-layer widths. Register-blocking
adds a-reuse on top: `Dot8x4` peaks at **86.5 GB/s** (K=768).

**Tuning finding → feeds §B/§D.** `Dot8x4` *regresses at K=3072* (40.5 GB/s) — below both `Dot4x4`
(51.4) and even single-row `dotFMA` (49.9) at that width. The 8 live YMM accumulators plus the
streamed b-rows appear to exceed what stays hot at large K (register/L1 pressure), so the 8-row
block is a win only for mid-K (≈768) and loses its lead as K grows. The dispatch should prefer
`dotFMA4` (or fall to single-row) past some K; finding that crossover is the open micro-kernel
tuning task. Not yet wired — the kernels are correct, the selection heuristic is the follow-up.

### B. Tune `parallelThreshold` on actual amd64 silicon

`parallelThreshold` (32M FLOPs) and the win curve in `parallel.go`'s comment were measured on
an M1 Pro. amd64 boxes differ in core count, memory bandwidth, and AVX2 throughput, so the
break-even point will move.

- Run `-bench 'MatmulParallel|MatmulSerial'` (and the `_L80_*` / `_attn512` probes already in
  `parallel_test.go`) on the target box.
- Find the smallest shape where parallel still beats serial net of goroutine spawn; set the
  constant there. Update the comment table with the new numbers.

### C. Decide whether to parallelize per-head attention QK^T

The per-head attention matmul (L512: ~17M FLOPs) wins ~3.4× in isolation but recurs
12 heads × 12 layers = 144×/forward, i.e. 1000+ goroutine spawns. Whether that's a net win
needs an **end-to-end single-forward benchmark on real weights** — not a microbenchmark.

- Get the CodeRankEmbed weights onto the box (`ken download-model --rerank --to testdata/encoder-model`).
- Write a benchmark that calls `Model.Encode` on one long document (L≈512) and compare wall-clock
  with `parallelThreshold` at 32M (attention serial) vs. lowered to ~8M (attention parallel).
- If it wins, lower the threshold; if the spawn overhead dominates, leave it and note it.

### D. (Optional) AVX-512 path

For Zen 4 / recent Intel server parts, an AVX-512 kernel (16-wide, more registers) is a further
step. Lower priority — AVX2 covers virtually all amd64 from ~2015, and AVX-512 brings
downclocking caveats. Same structure: detect via CPUID leaf 7, add `dot_amd64.s` entry points,
gate behind a `hasAVX512` flag.

### E. GPU: resident buffers (the transfer fix — do this before anything else GPU)

The foundation re-uploads a+b and reads back dst on every `MatmulBT` call; the table above shows
transfer dominates. The fix that unlocks the GPU is to keep data resident:

- Upload the 12 layers' weights to GPU storage buffers **once at model load**, not per call.
- Keep activations resident on-GPU **across all 12 layers** — only upload token embeddings at
  the start and read back the final CLS vector. No per-layer round-trip.
- This is also where the full-forward decision (Phase 3 "scope B") gets made: porting layernorm,
  softmax, silu, rope, and attention to WGSL compute passes so the activation never leaves the
  GPU. Measure end-to-end `EncodeBatch` wall-clock at a large batch (the only regime where GPU
  can win) against the CPU parallel path before committing.

### F. GPU: tiled matmul kernel (raise the compute ceiling)

The naive kernel plateaus ~42 GFLOP/s because every invocation streams full a/b rows from global
memory. A tiled kernel stages K-strips of a and b into `var<workgroup>` shared memory and reuses
them across the workgroup's output tile — the standard GEMM optimization, typically several × on
GPU. Needed in addition to (E): resident buffers remove transfer, tiling removes the memory-bound
ceiling. `TestMatmulBT_matchesNaive` already covers correctness for whatever kernel you drop in.

### G. GPU: large-batch storage-buffer tiling

The Metal device reports `maxStorageBufferBindingSize` = 128 MB. Big-batch activations
(e.g. B=32 × L=512 × 3072 f32 ≈ 200 MB) exceed one binding, so the resident-activation design in
(E) must tile the batch dimension across multiple dispatches. Note the cap when sizing batches.

---

## Quick reference: files

```
internal/linalg/dot_generic.go  scalar kernels (build !arm64)
internal/linalg/dot_other.go    aliases for !arm64 && !amd64
internal/linalg/dot_amd64.go    AVX2 dispatch + CPUID/XGETBV detection
internal/linalg/dot_amd64.s     dotFMA asm + cpuid/xgetbv asm
internal/linalg/linalg.go       exported Dot/Dot4x4/Dot8x4 + row-parallel MatmulBT
internal/linalg/dot_amd64_test.go  asm-vs-generic differential test
encoder/parallel.go        in-flight counter + row-parallel blocked matmul
encoder/parallel_test.go   exactness test + threshold benchmarks
encoder/linalg.go          matmulBTInto dispatch (naive / blocked / parallel)
encoder/gpu/doc.go         package doc (untagged — keeps default build non-empty)
encoder/gpu/gpu.go         WebGPU Context + WGSL matmul + MatmulBT  (//go:build gpu)
encoder/gpu/gpu_test.go    GPU-vs-naive correctness + GFLOP/s benchmark (//go:build gpu)
```
