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
   (~6.3×), **68–75%** at the K=768 transformer tiles; small matmuls keep the naive span
   (threshold); width stays numerically inert (8-aligned shards).

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
8. **`linalg.WeightMat` — unify the quantized-weight abstraction** —
   *new (goinfer review):* the precision-hiding weight-matrix wrapper is
   now implemented **three times** — aikit `encoder.LayerWeightsQ8`,
   goinfer `decoder.weightMat` (f32/int8/int4-group/W8A8, the richest),
   goinfer `vision.qmat` (f32/W8A8) — all dispatching into `linalg`.
   A shared Experimental `linalg.WeightMat` would collapse them, but
   `weightMat` is goinfer's fastest-moving internal type, so hoisting it
   now adds semver friction exactly where iteration happens. Trigger:
   the next time goinfer must *change* the abstraction anyway, or a
   fourth implementation appears. Not standalone work.
9. **`vision` (SigLIP/ViT encoder) → aikit** — *new (goinfer review):*
   goinfer's vision tower is an *encoder* (bidirectional, emits
   embeddings, parity-pinned, deps already aikit-only: `embed`+`linalg`;
   the GPU export is an import-free seam, same inversion as
   `encoder.Backend`) — by the split's own logic it sits on aikit's side,
   and would make aikit the only pure-Go image-embedding retrieval
   library. But it's capability acquisition, not deduplication: useful
   *retrieval* also needs SigLIP's **text** tower (Gemma's pipeline uses
   the LLM for text), which doesn't exist yet in either repo. Trigger:
   launch feedback / an adopter asking for image or multimodal retrieval.
   The projector + preprocessing-for-generation glue stays in goinfer
   regardless.
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
