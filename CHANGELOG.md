# Changelog

All notable changes to `aikit` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with the
pre-1.0 caveat documented in [README.md](README.md#versioning): the "Hard,
1.0-committed" surfaces are expected to be stable through the path to 1.0, but
breaking changes may still occur between `0.x` minors if the design requires
it.

## [Unreleased]

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

[Unreleased]: https://github.com/townsendmerino/aikit/compare/v1.8.1...HEAD
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
