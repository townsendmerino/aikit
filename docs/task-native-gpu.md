# aikit native GPU (Metal + CUDA) — full plan

> **BLUF.** aikit becomes a cgo-free retrieval/embedding toolkit whose **batch** workloads
> — ANN search, vision ViT, text encoding — run on **native GPU (CUDA / Metal), cgo-free end
> to end**, via a GPU-compute substrate aikit *owns* — exactly the way `linalg` already owns
> the CPU substrate. Today aikit's only GPU path is WebGPU (cgo, quarantined behind an
> inversion). This plan gives it native, cgo-free acceleration **where a GPU actually pays
> (batch)**, and unlocks **ANN** — the one hot-path workload with no GPU today.
>
> **Scope note ("as much as possible"):** the plan lays out the *full* arc, but it's phased
> so each phase is independently valuable and independently gated. You can stop after any
> phase with a coherent result. ANN (Phase 2) alone is the headline; the rest compounds it.

## Why — the case, strongest first

1. **Batch is the regime where a GPU pays.** goinfer's single-stream decode is bandwidth-
   bound → a GPU buys parity-to-modest (the whole Metal/CUDA saga). aikit's core work — ANN
   over a whole corpus, a batch of texts, a ViT over hundreds of patches — is **compute-
   bound**, where the fat-slice MMA path delivers **3–9×** (the vision path already reads
   "minutes to seconds"). The substrate built for decode pays a *higher* return pointed here.
2. **ANN is the one workload with no GPU path at all.** `ann.FlatI8.query` is a pure int8
   corpus GEMV (queries × the whole index — the largest single dimension in the system,
   >1M vectors), quantized by construction, calling CPU `linalg` directly with no backend
   seam. Native-GPU ANN takes retrieval — aikit's actual product — from CPU to native-GPU.
3. **cgo-free, end to end — an identity upgrade WebGPU can't give.** aikit gets GPU accel
   today *only* through the cgo WebGPU backend. goinfer's native backends (gocudrv / purego)
   are **cgo-free**. Native Metal/CUDA makes aikit a cgo-free retrieval toolkit that is
   *also* cgo-free-GPU-fast — no cgo anywhere, even for acceleration.
4. **Reuse + unification, not a bolt-on.** The device substrate and the quantized GEMV
   already exist in goinfer's `cuda/` + `metal/`. Extracting them to aikit makes **one GPU
   substrate serve both repos** — aikit's retrieval/vision/encode *and* goinfer's decode —
   mirroring how `linalg` already unifies the CPU path. Cleaner architecture, not more code.

## End-state architecture — the shape

aikit owns the **compute substrate**; the products build kernels on top of it:

```
aikit/linalg   — CPU matmul (MatmulBT / W8A8 / W4A8, WeightMat)          [today]
aikit/gpu      — cgo-free GPU device substrate (Device/Buffer/Pipeline)  [NEW]
                 + generic quantized GEMV (W8A8/W4A8) — the GPU twin of linalg
   ├─ metal impl  = goinfer/metal/metal.go, lifted (already zero-decoder-deps, cgo-free)
   └─ cuda  impl  = a thin wrapper over gocudrv (external, cgo-free by construction)

dispatch seam:  linalg.WeightMat.{Int8,Int4,F32}()  +  per-consumer Backend interfaces
   ├─ encoder.Backend        (exists — webgpu today; add native)
   ├─ ann.Backend            (NEW — ann has no seam today)
   └─ vision.ResidentEncoder (exists — WebGPU SigLIP; add native + Qwen ViT)

built ON TOP of aikit/gpu:
   ├─ aikit consumers:  ANN batch-GEMV, vision ViT, encoder GEMM     (this plan)
   └─ goinfer/cuda, goinfer/metal:  attention / rope / kv / moe      (already built —
                        re-pointed at aikit/gpu, same as decode already uses linalg)
```

Two properties this preserves:

- **cgo-free stays cgo-free.** Native backends are gocudrv/purego (no cgo); the WebGPU
  backend (cgo) stays quarantined behind the `encoder.Backend` inversion (`goinfer/gpu`,
  `-tags gpu`) exactly as today (`aikit/docs/architecture.md`). Default build: pure-Go CPU.
- **goinfer depends on aikit for GPU, as it already does for CPU.** The substrate lift is the
  GPU analogue of the existing `linalg` relationship — not a new dependency shape.

## Phases — each independently valuable, each gated

**Phase 0 — finish the seam (prerequisite; in-flight).** Complete the `linalg.WeightMat`
unification (roadmap §2.8 — migrate `vision.qmat` then `decoder.weightMat`; encoder is
done) and the vision-move goinfer cleanup (§2.9). WeightMat's `Int8/Int4/F32` accessors are
*the* GPU-dispatch seam every consumer routes through; it must be complete and validated
(bit-identical) before anything plugs a GPU into it. No GPU code yet — just the seam.

**Phase 1 — extract the GPU substrate to `aikit/gpu` (the structural move).**
- Define the `Device`/`Buffer`/`Pipeline`/`Kernel` interface (compile-a-kernel, alloc
  buffers, dispatch with args, sync) — the near-mirror-image contract both goinfer backends
  already satisfy.
- **Metal impl:** lift `goinfer/metal/metal.go` verbatim (it's already zero-decoder-deps,
  cgo-free, and proven standalone by `metal/gemv_w4a8_test.go`).
- **CUDA impl:** thin wrapper over `gocudrv` (CUDA's device layer is already the external
  dep — clean by construction) + the `LockOSThread` executor pattern.
- **Generic quantized GEMV (W8A8/W4A8):** extract the standalone kernels. Honest cost — on
  CUDA these are *braided* with LLM kernels in shared `.cu`→PTX blobs (`glue.cu`,
  `gemv_fwd.cu`); on Metal in one concatenated MSL string. Splitting them out is a real
  refactor, not a verbatim lift. Do it as a **blob-split**, kernel-parity-gated vs the CPU
  reference at each step.
- **Re-point goinfer's decode backends at `aikit/gpu`.** This is the proof the extraction is
  clean: goinfer's Metal + CUDA decode must stay **bit-identical green** across its full
  parity suite. If any resident model's numerics move, the extraction changed behavior —
  stop. **This phase is triggered by Phase 2's consumer pull; don't pre-extract speculatively**
  (aikit's own discipline — extract on a named pull, not on spec).

**Phase 2 — ANN-GPU (the headline unlock; the pull that justifies Phase 1).** Give `ann` a
`Backend` seam (like `encoder.Backend`); route `FlatI8.query` / `LoadFlatI8MmapPaged`'s
per-block batch GEMV through `aikit/gpu`'s W8A8. Batch multiple queries → an int8 GEMM, the
GPU's sweet spot, over the biggest N in the system. **Parity-gated:** GPU-ANN top-k ≡ CPU-ANN
top-k on real indexes (argmax/rank-exact within the int8 tolerance). This is *new* coverage
(ANN has none today) and the biggest single-workload win.

**Phase 3 — vision native + the Qwen ViT resident path.** `vision.ResidentEncoder` already
exists (WebGPU SigLIP, "~9×"); add native Metal/CUDA implementations, and build the
**Qwen2.5-VL** resident path (currently a documented follow-on, `goinfer/docs/prompts/
aikit-qwen25vl-vit.md`). Batch patches (M = np, thousands for dynamic-res Qwen) → the fat
GEMM the MMA path was built for. Parity-gated vs the CPU tower (HF-parity already exists).

**Phase 4 — encoder native.** `encoder.Backend` already has `webgpu`; add native Metal/CUDA.
Batch text embedding (M = L tokens / B·Lmax). Incremental over the existing cgo-WebGPU path —
the win here is *cgo-free + native-class*, not GPU-where-there-was-none.

**Not in scope:** `embed` (Model2Vec) — a token→row gather + mean-pool, no GEMM; GPU is the
wrong tool. **WebGPU stays** — it's the portable "any GPU (Vulkan/DX12/Metal)" fallback;
native is the cgo-free-fast path on NVIDIA/Apple. They coexist (native preferred where
present, WebGPU where portable, CPU always).

## Cross-cutting discipline (non-negotiable)

- **Parity on every path.** Each GPU consumer path is gated vs its CPU `linalg` reference
  (ANN top-k, vision logits, encoder vectors), **break-it-first** to prove the gate isn't
  vacuous (per `docs/parity-hunt-playbook.md`). The release-qualification sweep
  (`goinfer/docs/task-release-qualification-sweep.md`) extends to aikit's consumer × backend
  cells.
- **cgo-free preserved and asserted.** Native backends never introduce cgo (the CI guard
  that fails if `webgpu` leaks into aikit's core generalizes to "no cgo in aikit core").
- **Opt-in / additive.** Native backends behind build tags (`-tags cuda` / `-tags metal`),
  `CGO_ENABLED=0`; the default aikit build stays pure-Go CPU and every consumer degrades to
  CPU on a non-GPU build.
- **Device CI is hand-run.** GitHub can't run device tests (the objc-`msgSend` SIGSEGV; no
  GPU runner) — the device/parity gates are the scripted real-hardware tier, same as goinfer.

## Honest risks

- **The blob-split (Phase 1) is the real cost** — the generic GEMV is physically fused with
  the LLM kernels in shared PTX/MSL. The device substrate (metal.go / gocudrv) is the clean
  part; the kernels are the work.
- **Two of three consumers already have cgo-WebGPU** — for encoder + vision the native win is
  incremental (cgo-free + faster); the genuinely new coverage is ANN. Sequence accordingly.
- **Phase 1 is a large structural move** — it must be a pure refactor (goinfer decode
  bit-identical) or it stops. Land it only behind the Phase-2 pull, with goinfer's parity
  suite as the tripwire.
- **Scope creep** — "as much as possible" is the arc, but the value is front-loaded (ANN).
  Don't let Phase 3/4 gate the ANN win.

## Recommended sequence

Phase 0 (finish the WeightMat seam — already in flight) → **Phase 1 + 2 together** (extract
`aikit/gpu` *because* ANN pulls on it, and ship native-GPU ANN — the extraction's
justification and the headline win, in one arc) → Phase 3 (vision native + Qwen ViT) →
Phase 4 (encoder native). Each phase gated, each independently shippable, each leaving aikit
stronger and still cgo-free.
