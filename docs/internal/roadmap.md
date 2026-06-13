# aikit roadmap v4 — post-1.5.0

> Rewritten 2026-06-11, after v1.5.0. History: v1 (at v1.1.0) and v2 (at
> v1.2.0) were engineering roadmaps; v3 (at v1.4.0) predicted the backlog
> would empty into adoption work; v1.5.0 made that literal. Annotated prior
> versions live in git history (`git log -- docs/internal/roadmap.md`).
>
> **There are zero unblocked engineering items.** Four releases in three days
> (v1.2.0 → v1.5.0) shipped the full 2026 hybrid-retrieval bar — dense
> f32/int8 + lexical + learned-sparse, bi- and cross-encoder reranking,
> persistence/mmap/`//go:embed`, parity pins on every model path, fuzzing,
> benchmarks, an automated release gate. This document is now an adoption
> campaign and a list of triggers. **The standing rule: new engineering
> enters only through §2's triggers or §1's feedback. If a future session
> finds itself adding kernels while §1 is unstarted, stop and do §1.**

## Scorecard (cumulative, v1.2.0 → v1.5.0)

Retrieval: Flat / FlatI8 / HNSW (Alg-4, recall@10 0.68→1.00) / int8 HNSW
(Δ0) / BM25 ×2 analyzers / SPLADE end-to-end / RRF+RSF / QueryFilter.
Models: Model2Vec (both formats), CodeRankEmbed, MiniLM BERT, SPLADE head,
ms-marco cross-encoder — all parity ≤1e-5, golden-pinned, cgo-free.
Embedded: versioned blobs + zero-copy mmap + `//go:embed` demo (~50 ms
startup). Proof: README comparative table, BeIR/scifact nDCG@10 0.638,
~21 texts/s pure-Go MiniLM. Health: fuzzing (5 fixes), typed errors, scoped
knobs, pool deleted, surface audit (deliberate keep), format policy
(rebuild-per-minor), release gate (ran 1.4.0 and 1.5.0 unattended).

---

## 1. The adoption campaign — the only live work

In order. Items 1–3 are a sequence, not a menu; each feeds the next.

1. **Post the announcement** — [high / trivial; asset exists]. The r/golang
   draft is written (`r-golang-post.md`, repo root — strip the posting-notes
   footer). Post it; optionally follow with Show HN a few days later (HN
   wants a different lead: the war story or the `//go:embed` demo, not the
   feature list). Submit to awesome-go the same week (its bar: API docs,
   coverage, CI — all already met).
2. **Work the response** — [high / reactive]. Answer every comment and
   issue fast for the first two weeks; the prepped Q&A (footer of the post
   draft) covers the four predictable threads. Triage feature requests
   against §2's triggers rather than accepting them reflexively — but a
   request that *matches* a trigger (e.g. a Windows consumer, a >1M-vector
   corpus) is the trigger firing, and unblocks that item immediately.
3. **Land one named adopter** — [high / not-engineering]. The strategic
   item the last three roadmap versions agreed on. Shapes, best first: an
   external Go service wiring in semantic search (offer hands-on help, as
   the post does); an MCP server `//go:embed`-ing a docs corpus (build it
   *with* someone, not speculatively); ken publicly badged as built-on-aikit
   (first-party, so it half-counts, but it's honest social proof and free).
   Ship v1.6 *with* the adopter named in the README.
4. **The recall war-story blog post** — [medium / low]. Separate asset from
   the announcement: "synthetic vectors can't measure recall" (0.99 → 0.68
   → 1.00) is a standalone engineering post that travels on its own and
   back-links the repo. Write after #1's response settles, using what the
   comments reveal about which framing lands.

## 1b. Unblocked items (from the 2026-06-12 goinfer cross-repo review + external kernel review)

The review's headline: **the split is holding** — goinfer consumes aikit's
loaders/kernels properly, deps point inward only, no container-format
duplication. One deduplication earns immediate work; the rest is gated (§2).

1. **Typed tensor accessors on `embed.SafetensorsFile`** — ✅ **DONE.** [medium / low].
   goinfer hand-writes the same shape-checked typed reads ≥6 times across
   `decoder/weights.go`, `vision/encoder.go`, `decoder/lora.go`,
   `decoder/gptq.go` (`tensorF32`, `loadF32(want…)`, `f16Tensor`,
   `i32Tensor`). Add `TensorF32(name string, want ...int)` (+ I32, and F16
   widening) as methods. Additive to the Hard tier; goinfer deletes its
   helpers at its next aikit bump. Permitted under the standing rule as
   measured deduplication with a named consumer, not new capability.

2. **2×8 register GEMM micro-kernel** — ✅ **DONE.** [high / medium]. External
   review flagged `Dot8x4` as a load-bound 1×8 kernel (each b-load → 1 FMLA, 8
   accumulators < the ~16 to hide FMA latency). The gate — a peak-fraction bench
   against a *measured* f32 ceiling (`fmaPeakARM64` clocked 95.4 GFLOPS, ≈15
   FMA/cyc, settling the 8-vs-16-FMA/cycle question empirically) — put the GEMM at
   40–49% of peak (≤50% ⇒ proceed). `linalg.Dot2x8` (2 a-rows × 8 b-rows, 16
   accumulators) recovered it to **68–73%**: encoder FC matmuls 1.5–1.6×, end-to-end
   encode 1.27–1.36×, bit-identical (same accumulation order as `Dot8x4`, golden
   parity unchanged). arm64 NEON only. Remaining levers, *not* taken (measured win
   already lands the target, and the standing rule discourages speculative kernels):
   the AVX2 port belongs with §2.4 (gated on Zen 4+ access); a 4×8/B-packed
   outer-product kernel could chase the last ~25% to ~90% of peak but needs a real
   throughput trigger to justify the packing path.

3. **Blocked GEMM hoisted into shared `linalg.MatmulBT`** — ✅ **DONE.** [high / medium].
   goinfer's gate (recorded in its `matmulbt-prefill-headroom` note) found the *public*
   `linalg.MatmulBT` was a naive dot-per-output span with **no cache blocking** —
   re-streaming `b` per a-row, **~7% of peak** at prefill shapes. That's a defect every
   kit consumer of `MatmulBT` inherits (goinfer's own f32 vision encoder sat there), not
   just a goinfer concern — so it was fixed despite goinfer deferring its *own* adoption.
   The encoder's blocked + register-tiled GEMM was hoisted into `linalg`
   (`matmul_blocked.go`) as the single shared home; `MatmulBT` (column-parallel) and a new
   Experimental `MatmulBTInto` (serial) both use it, and the encoder delegates
   (bit-identical, golden parity unchanged). Measured **7%→46%** at M=512×4096×4096
   (~6.3×), **68–75%** at the K=768 transformer tiles; width stays numerically inert
   (8-aligned shards). *(Update, v1.7.2: the naive-span threshold for small matmuls was
   removed so `MatmulBT`'s per-output result is M-invariant — the threshold switched
   reduction order at the M=1↔M=K boundary, an avoidable f32-reassociation footgun. All
   M now route through the blocked kernel, which measured faster at small-M decode/
   attention shapes anyway. Gated by `TestMatmulBT_MConsistent`; see §1b.4 for the
   over-attribution that prompted it.)*

   Then the 46% itself was chased and mostly closed: the large-K shortfall wasn't tile
   size but **L1 associativity conflicts** — the 8 b-rows a `Dot2x8` reads are K·4 bytes
   apart and collide in the same cache sets. **B-panel packing** (copy each 8-row group
   into a low-stride buffer first) fixed it **bit-identically**: prefill M=512 **46%→69%**,
   and the encoder's own K=3072 fc2 **+15%** (golden parity unchanged). Gated to K≥2048
   (K=768 dims are already low-stride) and arm64 (packed kernel is NEON `Dot2x8`; amd64
   AVX2 packing → §2.4). Remaining: large M (≥~2048) recovers less (≈53%) because the
   a-panel is re-read per column-group — full 3-level (Goto) blocking with A-packing would
   close it (~70%+) but is a substantial new path, promoted to **§2.12** with its trigger
   (a real large-M f32 prefill that's watched — concretely goinfer's multimodal
   image-prefill). Measured along the way: smaller kBlock and wide n-panels both *hurt* —
   the simple 8-col pack is the sweet spot.

4. **`MatmulBT` made M-invariant — and the mis-attribution that prompted it** — ✅
   **DONE (v1.7.2), with a correction.** [high / low]. A consumer (goinfer) reported a
   same-model speculative-decoding parity failure (`TestSpeculativeGreedyParity`,
   acceptance 0.893 vs ~1.0) after bumping its pin, and it was **mis-attributed to
   aikit**: the theory was that §1b.3's naive/blocked threshold in `MatmulBT` made the
   f32 result M-dependent (M=1 naive vs M=K blocked → ~1e-5 reassociation → flipped
   argmax). We removed the threshold so `MatmulBT` is now **M-invariant** (every output
   bit-identical regardless of M; all M route through the one blocked-kernel order,
   which measured **2–3.8× faster** at small-M decode/attention shapes than the naive
   span it replaced — no perf cost). That is a real improvement and worth keeping —
   `MatmulBT` being M-dependent was an avoidable footgun. **But it did not fix the
   reported bug.** The speculative-parity failure was **consumer-side**: goinfer's
   dense attention computed QKᵀ/AV through two paths (`attendQuery` vs
   `attendBatchedHeads`) that were not bit-identical, and goinfer fixed it there by
   moving attention onto f64 accumulation (`MatmulBTAcc64`, untouched by aikit). The
   threshold removal *transiently* shifted goinfer's f32 attention numerics until that
   f64 move landed; once it did, goinfer's quantized forward stopped calling f32
   `MatmulBT` entirely (W8A8/Q8 weight kernels never did), so the issue is moot.
   `blockedFill`'s internal M-invariance (paired `Dot2x8` vs odd `Dot8x4`) is pinned by
   `TestMatmulBT_MConsistent`; the invariant is documented on `MatmulBT`; the quantized
   kernels (`MatmulBTW4A8`/`Q8`/`W8A8*`) were already M-consistent — untouched.
   **Meta-lesson (recorded so it isn't repeated):** localize a failure in the consumer
   before pointing at the dependency, and never put a downstream-effect claim
   ("fixes X in goinfer") in a release note when only the local property
   (M-invariance) was verified. Chasing the dep cost a release of churn.

   **Follow-up (v1.7.3): the threshold removal exposed a latent amd64 kernel bug —
   found and fixed, not reverted.** Routing *all* shapes through the blocked kernel
   meant small odd-`n4` shapes hit the amd64 AVX2 `Dot8x4`/`Dot4x4` for the first
   time (nothing had since the 1.6.0 hoist), and those kernels were wrong for odd
   `n4`: the trailing single 4-group used an XMM/VEX.128 FMA that zeroed the upper
   128 bits of each YMM accumulator, dropping the loop's lane-4..7 partials (K=13,
   K=300 failed; even-`n4` shapes passed — and arm64 / non-AVX2 were always
   correct). 1.7.3 fixes the kernel (YMM-form tail FMA that preserves the upper
   lanes), keeping 1.7.2's M-invariance — which is now actually correct on amd64.
   A direct `TestAVX2_blockedKernels_oddN4` (odd + even `n4` vs scalar) closes the
   test gap: the prior AVX2 test only exercised the single-row `dotFMA`, never the
   multi-row blocked kernels, which is why the bug stayed latent. Meta-lesson #2:
   a kernel test that doesn't cover the tail × every register-block width leaves a
   blind spot exactly where reuse hides it.

## 2. Gated — unchanged triggers, now the only path for engineering

1. **HNSW zero-copy mmap** — bundle with the next format bump (specced at
   the bump sites), never standalone.
2. **Binary/Hamming pre-filter + rescore** — trigger: an adopter with >1M
   vectors.
3. **Windows real mmap** — trigger: a sizable Windows consumer.
4. **x86 tail (VNNI W4A8, AVX-512)** — trigger: Zen 4+/Cascade Lake+ access
   (cloud c7i/c7a spot remains the cheap unblock); bundle both.
5. **`GOEXPERIMENT=simd` migration** — trigger: archsimd ships arm64 AND
   graduates (amd64-only + gated as of June 2026; arm64/SVE on dev.simd for
   Go 1.27). Then re-spike; within ~10% of hand asm ⇒ migrate, delete `.s`.
6. **In-place index mutation** — trigger: a real consumer proving
   rebuild/delta/swap insufficient. Design rule 4 holds.
7. **Experimental→Hard graduation** — *new, the long-game item:* BERT /
   SPLADE / CrossEncoder / int8 indexes / persistence graduate to the
   semver tier once they survive two quiet consecutive minors under an
   external consumer (the same bar the original Hard tier met). Trigger:
   §1.3's adopter + that stability window. Re-run the surface audit's
   "re-evaluate" notes (unconsumed Workspace methods) at that moment.
   *Note (§2.8):* `WeightMat.MatmulBTInto`'s f32/W8A8 paths now route
   through the `Workspace`-scoped matmul (honoring `SetThreshold`/
   `SetWorkers`), but no in-repo consumer exercises them yet — the encoder
   Q8 migration uses its own baked-scale kernel. The "unconsumed" status
   flips only when the goinfer `decoder.weightMat` f32 path migrates onto
   `MatmulBTInto`; re-evaluate then.
8. **`linalg.WeightMat` — unify the quantized-weight abstraction** —
   🟡 **IN PROGRESS (type + 1 of 3 consumers).** The precision-hiding
   weight-matrix wrapper was implemented **three times** — aikit
   `encoder.LayerWeightsQ8`, goinfer `decoder.weightMat`
   (f32/int8/int4-group/W8A8, the richest), goinfer `vision.qmat`
   (f32/W8A8) — all dispatching into `linalg`. The shared Experimental
   `linalg.WeightMat` now exists (storage-only: precision/scales/dispatch;
   model policy stays with each consumer) and **aikit's `encoder` Q8 path
   is migrated onto it, bit-identically** (cosine 0.997 unchanged, Q8
   golden green, -race clean).
   *Gate note:* the stated trigger (goinfer must change the abstraction
   anyway, or a 4th impl) was **not** actively firing — goinfer is on main,
   clean, no in-flight `weightMat` change. This proceeded as **Francis's
   owner override**, and was scoped to land the type + in-repo consumer
   without disturbing goinfer's fastest-moving internal type.
   *Remaining:* migrate goinfer `vision.qmat` (now an in-aikit refactor
   after §2.9's move; validates the GPU-export accessors) then
   `decoder.weightMat` (richest, keep goinfer's `quantMode.embedding()`
   policy as a thin shim) against the released aikit minor — `go.work
   replace` for dev, goinfer pins on release.
9. **`vision` (SigLIP/ViT encoder) → aikit** — ✅ **DONE (aikit side).**
   The vision tower moved into `aikit/vision` (Experimental), verbatim and
   parity-preserving — decode/preprocess/forward/qmat/resident + the
   import-free GPU-export seam; deps are `embed`+`linalg` only and it adds
   **no** external dependency (image codecs are stdlib). aikit is now the
   only **cgo-free** image-embedding library. The full decision/scoping
   record is `docs/internal/vision-move-decision.md`.
   *Gate:* the stated trigger (launch feedback / an adopter asking for
   image or multimodal retrieval) had **not** fired — the move proceeded as
   **Francis's owner override**, recorded in the decision doc + CHANGELOG.
   *Dependency audit (the one real risk):* the decode path is stdlib
   `image/jpeg`+`png` only — no `x/image`, no cgo — so the "stdlib + x/text
   only" promise holds without a decode quarantine.
   *Scope held:* the Gemma-specific projector (vision→LLM tokens) and the
   image-soft-token sentinels stay in goinfer; the projector↔encoder
   boundary is a plain `[]float32`, no decoder imports `vision`.
   *Remaining (goinfer side):* delete goinfer's encoder copy, rename its
   leftover to `package multimodal`, point `gpu/vision_*.go` at
   `aikit/vision`, bump the pin on aikit release — validated green via
   `go.work replace` first. And see the new §2.13 (text tower).
10. **Explicitly left in goinfer** (reviewed, no move): GPTQ/AWQ
   reconstruction (decoder-checkpoint formats, no aikit consumer);
   BPE/SentencePiece tokenizers (generation-side; note a future
   bge-m3-class multilingual embedder would need SentencePiece *Unigram*,
   which neither repo has); `rmsnorm`/`rope` (small, intentional
   duplication per the split); constrain/chat/sampler/serve/gpu.
11. **AMX** — out of scope.
12. **3-level (Goto) f32 GEMM for large-M prefill** — *new (from the §1b.3 packing
   work).* B-panel packing took large-K shapes to ~69% at M≤~1024, but large M
   (≥~2048) recovers less (≈53%): the a-panel is re-read once per output column-group.
   Closing it needs full Goto blocking — an L2-resident packed B panel reused across
   M-blocks, plus A-packing, plus a 3-level (nc/kc/mc) loop. Substantial new kernel
   path. Trigger: a real large-M f32 prefill that anyone watches — concretely goinfer's
   multimodal **image**-prefill latency (the same trigger as §2.9 vision), since that is
   the one hot f32 `MatmulBT` consumer with M≥2048. Measured dead-ends recorded so they
   aren't re-tried: smaller kBlock and wide n-panels both regress. Bundle the amd64 AVX2
   packed kernel (§2.4) with it.
13. **SigLIP *text* tower** — *new (opened by §2.9's vision move).* The vision
   encoder shipped in `aikit/vision` does image→image and image-as-document
   today, but **text↔image** retrieval needs SigLIP's text tower — which exists
   in neither repo (Gemma drives the text side with its LLM, which aikit
   doesn't have). It's BERT-shaped, so `encoder`'s transformer machinery + the
   parity toolchain make it tractable, but it is its own parity-pinned work: its
   own pin script, its own golden, a shared embedding space verified against the
   HF `SiglipModel` (image+text). Trigger: someone actually needs text↔image (a
   cross-modal search adopter), **not** just image→image — don't build the text
   tower speculatively off the image move alone.

---

## Competitive context (refreshed 2026-06-11)

Unchanged from v3 in substance; deltas only:

- **hugot** — the last pipeline gap closed (cross-encoder, same checkpoint,
  Δ5e-6). The comparison is now fully head-to-head and framed honestly in
  the README (deployment tradeoff, not a drag race). Their Go-SIMD bet
  matures with archsimd (§2.5's trigger) — the announcement window is open
  *now* and narrows when that graduates.
- **Antfly/Termite** (Zig core), **coder/hnsw** (measured 0.22 vs 0.995),
  **Bleve/chromem-go/sqlite-vec** (index-only), **Ollama** (server) — all
  unchanged; the README table and capability matrix carry these.
- **Model2Vec upstream** — supported including standard format; watch for
  new potion releases (a better static model is a free quality bump through
  the parity pipeline).
