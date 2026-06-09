# Changelog

All notable changes to `aikit` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with the
pre-1.0 caveat documented in [README.md](README.md#versioning): the "Hard,
1.0-committed" surfaces are expected to be stable through the path to 1.0, but
breaking changes may still occur between `0.x` minors if the design requires
it.

## [Unreleased]

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
  Dequant-path fuzzing (`Tensor`/`RowDequantizer`) is a tracked follow-up.

### Documentation

- `linalg` now has a package `doc.go` with the kernel-dispatch map (which kernel
  fires on which CPU, and why), and `Dot8x4` documents its large-K throughput
  cliff with the "tile K to ≤~768" guidance. README's model-fetch quick start no
  longer requires `ken` — it uses the Hugging Face CLI directly.

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

[Unreleased]: https://github.com/townsendmerino/aikit/compare/v1.1.0...HEAD
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
