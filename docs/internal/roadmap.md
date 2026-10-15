# aikit roadmap v5 — post-1.5.0

> Rewritten 2026-06-11, after v1.5.0; revised 2026-08-14 to drop the
> adoption campaign. History: v1 (at v1.1.0) and v2 (at v1.2.0) were
> engineering roadmaps; v3 (at v1.4.0) predicted the backlog would empty
> into adoption work; v1.5.0 made that literal (v4's §1 was that campaign —
> r/golang, Show HN, awesome-go, a named adopter). **v5 retires it: aikit is
> not separately promoted.** Promotion effort goes to goinfer and ken; aikit
> is their dependency and gains users transitively, not through its own
> outreach. (r/golang specifically had stopped taking anything past the
> small-project category — not a channel worth chasing further.) Annotated
> prior versions, including the full campaign text, live in git history
> (`git log -- docs/internal/roadmap.md`).
>
> **There is no live work.** Four releases in three days (v1.2.0 → v1.5.0)
> shipped the full 2026 hybrid-retrieval bar — dense f32/int8 + lexical +
> learned-sparse, bi- and cross-encoder reranking, persistence/mmap/
> `//go:embed`, parity pins on every model path, fuzzing, benchmarks, an
> automated release gate. **The standing rule: new engineering enters only
> through §2's triggers.** If a future session finds itself adding kernels
> with no trigger fired, stop.

## 0. The asset-blocked cluster — CLEARED 2026-08-29

> **All three records closed.** `testdata/encoder-model` (CodeRankEmbed, 523 MB) and
> `testdata/minilm-model` (all-MiniLM-L6-v2, 87 MB) fetched; both gitignored, so nothing entered
> git. Encoder skips went **67 → 35** (missing-asset **63 → 33**), 41 more tests now execute.
> (1) `task-native-gpu.md`'s last item published at ~1.5× batched on Metal (`a9fd0af`) — and it
> was never blocked on the checkpoint that plan named; `forwardBatch`'s kernel is SwiGLU-only, so
> MiniLM could not drive it and CodeRankEmbed was the one that could. (2)
> `buildWeightsFromSafetensors` refactored, 103 → 69 lines, with the MoE-transpose guard proven
> load-bearing by removal (`d0f6bf7`). (3) The skips are counted rather than silent. The 33
> remaining need *other* fixtures (nomic-embed, crossencoder-model, model) — a smaller,
> separate cluster.

## 0-old. The original entry (2026-08-28)

**Not new engineering, so it does not need a §2 trigger** — this is three things already
recorded as blocked that share a single cause, plus the fetch that clears them. It is listed
here because "aikit has no live work" is true and kept being read as "aikit has nothing worth
doing", which is a different claim.

The cause: `testdata/{encoder-model,model,minilm-model}` are absent on the dev box. Measured
2026-08-28 — the encoder package runs **110 passed / 67 skipped, 63 of them missing-asset**.
What that single absence is currently holding:

1. **`task-native-gpu.md`'s last open item.** Every phase of that plan is ✅ and the doc says
   "nothing is gated behind an unfired trigger any more" — what remains is *one measurement*,
   the batched encoder end-to-end wall-time, and it is checkpoint-blocked, not code-blocked.
2. **`task-code-health.md` §4.5's one blocked item.** `buildWeightsFromSafetensors` carries
   ~20 identical `loadF32` error blocks an accumulator would collapse. It was declined on
   COVERAGE, not taste: the refactor is behaviour-affecting (the accumulator must be re-checked
   before `transposeExpertsW2` or a failed load hands it a nil slice), and no test executes
   over it here.
3. **The 63 skips themselves**, which make a green `go test ./encoder/` mean much less than it
   looks like it means — the false-clean shape the skip census exists to surface.

**The move:** fetch one encoder checkpoint. It is the cheapest thing on any aikit list and it
converts three "blocked" entries into either done or genuinely-decided. Nothing else here is
gated behind it, and nothing new gets built as a result.

**Separately, unblocked but unsettled:** the AVX-512 VNNI cold multiplier (`cpu-acceleration.md`
item 4) is real in 4/4 runs but ranges 1.109–1.270×, and the hot ratio is settled at ±2.2% while
the cold one is not (±6.8%). The lever is method, not hardware: interleave the AVX2 and VNNI
arms **inside one process** and compare medians, rather than separate runs. That is what turned
an inconclusive `scanFlat` measurement into a ±1% answer, and it may settle the cold number on
the shared Xeon without needing quiet silicon.

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

## 1. Unblocked items (from the 2026-06-12 goinfer cross-repo review + external kernel review)

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
   attention shapes anyway. Gated by `TestMatmulBT_MConsistent`; see §1.4 for the
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
   aikit**: the theory was that §1.3's naive/blocked threshold in `MatmulBT` made the
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
   external consumer (the same bar the original Hard tier met). Trigger: an
   external consumer using the tier in production, organically found — not
   chased; aikit is not separately promoted (see the top-of-file note) — plus
   that stability window.
   *Note (§2.8), updated 2026-07-27:* `WeightMat.MatmulBTInto`'s f32/W8A8
   paths route through the `Workspace`-scoped matmul (honoring
   `SetThreshold`/`SetWorkers`), and are now **consumed in-repo** — the
   `aikit/vision` tower threads a `Workspace` through all six per-layer
   projections via `lw.MatmulBTInto(&s.ws, …)` (`vision/encoder.go`, the
   audit-#12 Workspace threading). So the "unconsumed" caveat is resolved;
   the method has real coverage for the graduation audit.
   *Partial ✅, 2026-08-18:* first real audit pass against the trigger, checked
   against goinfer's actual source (not just its `go.mod`) rather than assumed.
   Four surfaces had organically-found production evidence and graduated:
   `linalg` root (`Dot`/`MatmulBT`/int8-int4 quant kernels — 36 non-test files
   in goinfer), `encoder.Backend`/`RegisterBackend`/`NewBackend` (goinfer's
   `gpu/backend.go` + `internal/serveapp/{embeddings,openai,main}.go`),
   `vision` whole package (goinfer's serving path, GPU registration, demo
   agent), `mmap` whole package (4 non-test goinfer files). Checked ken too —
   it contributed no additional evidence (imports only already-Hard-tier
   aikit surface). Everything else in the Experimental list (HNSW, FlatI8,
   BERT, SPLADE, CrossEncoder, `sparse`, `TokenizePlain`, `Truncate`, …) had
   **zero** organic external-production hits in either repo and stays
   Experimental regardless of age — age alone was never the bar. README's
   Stability tiers + Versioning sections rewritten to match (also fixed: the
   Versioning section's own "at 1.0 this tightens" blob-format language had
   gone stale — the milestone passed without the tightening landing; now
   says so honestly instead of implying it's still pending arrival).
8. **`linalg.WeightMat` — unify the quantized-weight abstraction** —
   ✅ **DONE (type + 3 of 3 consumers).** The precision-hiding
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
   *Completed since:* goinfer's `decoder.weightMat` migrated (the type is
   gone; `decoder` now holds `linalg.WeightMat` directly — see `eagle.go`
   fields and `gguf.go`'s `streamMat`), and `vision.qmat` migrated as the
   in-aikit refactor §2.9's move made it. The open-coded wrapper is deleted
   in all three places; `vision` keeps only `newQMat`, which is storage
   *policy* (which weights quantize, and that f32 is copied because the
   source is a released mmap) rather than a second abstraction.
   *Bit-identity is gated, not assumed:* the tower's own parity tests need
   checkpoints and skip without them, so `vision.TestQMat_migrationIsBitIdentical`
   reconstructs the old kernel calls and asserts exact float-bit equality
   against the WeightMat path (plus the no-alias and int8-export contracts).
   Break-it-first: flipping the `w8a8` flag moves the result and turns it red.
   *Now consumed (cf. §2.7), updated 2026-07-27:* `aikit/vision` exercises
   `WeightMat.MatmulBTInto` — the audit-#12 Workspace threading routes all six
   per-layer projections through `lw.MatmulBTInto(&s.ws, …)` (`vision/encoder.go`),
   so the `Workspace`-scoped f32/W8A8 paths have an in-repo consumer. The decoder
   migration landing on `MatmulBT` is no longer what the re-evaluation waits on.
9. **`vision` (SigLIP/ViT encoder) → aikit** — ✅ **DONE (aikit side).**
   The vision tower moved into `aikit/vision` (Experimental), verbatim and
   parity-preserving — decode/preprocess/forward/qmat/resident + the
   import-free GPU-export seam; deps are `embed`+`linalg` only and it adds
   **no** external dependency (image codecs are stdlib). aikit is now the
   only **cgo-free** image-embedding library. The full decision/scoping
   record is `docs/internal/archive/vision-move-decision.md`.
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
12. **3-level (Goto) f32 GEMM for large-M prefill** — *new (from the §1.3 packing
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
14. **`bm25` persistence — a serialized `Index` and/or a `Builder.Add` seam** —
   *new (the 2026-07 perf campaign's N4/N6, deferred as an API decision, not a perf
   item).* The campaign measured the cost of the absent format — `bm25.Build` from
   scratch is **21.5% of the flagship example's cold start** (`nvidia-rtx2070s` W3;
   the original lens heading said 67%, which was measuring against a hand-written
   checkpoint instead of the real 64 MB load), and eliminating it entirely takes
   time-to-first-result from 82.0 → 64.4 ms. Still the largest single cold-start item,
   no longer two thirds of it.
   *Why it is gated, not scheduled:* the cost is a permanent compatibility promise. A
   versioned on-disk format means every future field on `Index` has to be carried,
   defaulted for old files, and gated, for as long as the format exists — a decision
   about what `bm25` **is**, and 17.6 ms is not sufficient grounds for a benchmark to
   make it. Whoever takes it should start from the format and the version-skew policy,
   not the percentage, and should decide alongside it whether `bm25` wants a
   persistence story at all: **rebuilding from an already-embedded corpus may be the
   honest answer** for a package whose build is O(corpus) and fast (the `bm25.Build`
   godoc now says this).
   *N6 (`Builder.Add(tokens)`) is held WITH N4, not separately,* because it is the same
   question from the other side: a serialized `Index` and an incremental `Builder` are
   both about `bm25`'s input/output surface, and committing to either shape first
   constrains the other. Take them together as an API review or not at all.
   *The general lesson (§5.6, "two of the five biggest items are missing APIs"):* N1
   (`StaticModel.EncodeBatch`) shipped inside the campaign and N4 did not, and the split
   is principled — **a perf campaign can measure an absent API's cost but cannot decide
   its shape.** The measurement says how much the absence costs; it never says what the
   presence should look like. N1 was additive with a contract obvious from the serial
   loop it replaced; N4 is a format. Trigger: an adopter who needs warm-restart or
   incremental indexing badly enough to own the format's compatibility surface.
   *Partial ✅, 2026-08-18:* re-measured the split before deciding anything — the
   17.6 ms combined cost is almost exactly 50/50 between `TokenizePlain`
   (8.83 ms) and `Build` itself (8.78 ms), not tokenization-dominant as first
   assumed. Shipped the smaller, reversible half:
   `bm25.MarshalTokens`/`bm25.UnmarshalTokens` cache the tokenized corpus
   (`Build`'s input) so a `//go:embed` deployment skips re-tokenizing raw text,
   while `Build` still runs on load at its own cost. `bm25.Index` itself gains
   **no** format, no version, no compatibility promise — the new surface can be
   rendered obsolete or dropped without touching `Index` at all. This resolves
   roughly half the measured cost (~8.8 ms) as a low-commitment opt-in; full N4
   (serializing `Index` itself, eliminating the other half) and N6
   (`Builder.Add`) remain gated exactly as before, same trigger.

---

## 3. Speculative backlog — not triggered

*Folded in from the former `docs/internal/ideas.md` (2026-08-14): 0 code
citations, not linked from anywhere, drifting as an orphan doc alongside the
one roadmap. Merged here so there is one live-work document, not two.*

A backlog of "AI-cool" additions that fit aikit's ethos: **pure Go, no cgo,
parity-tested, independently importable, composes with what's already here.**
This is deliberately a menu, not a commitment under the standing rule above —
none of these has a trigger; do not start one without finding or declaring
one first, same as §2. Each entry says what it is, why it fits aikit
specifically, how hard it is, and what it composes with.

**Torch-oracle endgame — PARKED, 2026-08-18 (Francis), low priority, no
trigger.** aikit's own ~23-file `scripts/oracle/*.py` cluster (the
BERT/SPLADE/cross-encoder/SigLIP/Qwen2.5-VL-vision parity golden generators)
is the same class of problem as goinfer's much larger decoder-side one — a
pure-Go independent reference (`oracle/`-shaped: own module, own weight
reader, deliberately naive f64 math, periodically re-anchored against real
HF/PyTorch output) could eventually replace routine per-model golden
generation, dropping Python to occasional per-architecture anchor runs
rather than eliminating it. Full design, phases, and open decisions live in
goinfer's `docs/task-oracle-refforward.md` (status line records the parked
decision); if this is ever revisited, do it as **one joint Phase-0
clustering pass across both repos' scripts together**, not as two separate
plans — the independence argument and phase structure apply unchanged to
both.

Context: aikit is the pure-Go **retrieval** toolkit (`topk`, `ann` flat+HNSW,
`bm25`, `fuse` RRF, `embed`, `encoder`, `chunk`, `linalg`); the **generation**
half (decoder / tokenizer / sampler / constrained decoding) lives in goinfer,
which has its own `docs/roadmap.md`. This backlog was originally written when
both halves were one repo and mixed the two scopes with no marker — a reader
skimming just this file couldn't tell which items were aikit's to build.
*(2026-08-14: the six generation-side entries that used to live here — a
`rag` pipeline, constrained generation, speculative decoding, logprobs,
sampler family, KV-cache — were removed. goinfer's own roadmap already
tracks or has shipped every one of them, in more current and more detailed
form than this file ever had: sampler completeness including min-p/
penalties/logprobs, `GrammarFromStruct` JSON-Schema constrained decoding,
a greedy-draft speculative implementation (parked pending a bandwidth-bound
backend), and int8/f16 KV cache with prefix caching.)* What remains below is
aikit-side retrieval and primitive work only.

Rough effort key: **S** = a few days, **M** = a week or two, **L** = a month+.

**Vision-tower scratch aliasing (A-5 of the encoder scratch task) — PARKED, 2026-10-15
(Francis), trigger: CPU-path vision peak RSS becomes a complaint.** The text encoder's
attention and MLP scratch now share one slab (v1.70.2). The same treatment for
`vision.newQwenScratch` (qwen, qwen3, glm_ocr, pixtral) was measured before starting. At
Qwen2.5-VL tower dims and 1024 patches it would alias ≈28 MB of the ≈96 MB scratch (−21% of
the 135.6 MB allocated per Forward), with no speed gain (milliseconds of alloc and zero against
a 0.6–1.1 s forward). Peak RSS is the only real win, and it scales with patches: ≈150 MB at
~5,000. Pooling the per-Forward scratch instead was measured and is not worth it. It cuts B/op
87%, but it doesn't touch peak RSS and removes only 1.9 ms per Forward (see `perf-dead-ends.md`).
Numbers and the original scope:
[`archive/task-encoder-scratch-aliasing.md`](archive/task-encoder-scratch-aliasing.md) §3 A-5.

### Tier 1 — retrieval-side depth

#### 1. Late-interaction retrieval (ColBERT-style multi-vector)  ·  M

**What.** Instead of one vector per document, store per-token embeddings and
score a query against a document by summing each query token's max similarity to
any document token (MaxSim). Much stronger recall than single-vector,
especially for short or keyword-ish queries.

**Why it fits aikit.** It reuses `encoder` for token-level embeddings and slots
in as an alternative retriever behind a RAG pipeline's retrieval interface
(goinfer's, or a caller's own). It's a well-defined algorithm with a clear
correctness story (MaxSim is exact), and it differentiates aikit's retrieval
from "yet another cosine index."

**Design notes.** Storage blows up (N tokens × dim per doc), so this wants the
quantized index (#2) underneath to be practical. Two-stage is natural: cheap
single-vector ANN to get candidates, MaxSim rerank on the survivors.

**Composes with.** `encoder`, `ann`. **Risk:** memory; pair with #2.

#### 2. Quantized / large-scale vector index — PQ, IVF, scalar quant  ·  M

**What.** Make `ann` scale past "fits in RAM as f32": **scalar quantization**
(int8 vectors, 4× smaller), **product quantization** (PQ — split the vector,
quantize subspaces, asymmetric distance), and **IVF** (inverted-file coarse
clustering so a query only scans a few cells). Optionally **binary embeddings**
+ Hamming rerank for a very fast first stage.

**Why it fits aikit.** The current flat + HNSW indexes are great up to a point;
PQ/IVF is what lets a laptop hold millions of vectors. It's classic,
well-specified, pure-Go-friendly numerical code with exact recall/latency
tradeoffs you can measure — squarely aikit's style.

**Composes with.** `ann`, `embed`, late-interaction (#1). **Effort:** M,
more if you want the full IVF-PQ combo with good recall tuning.

#### 3. Retrieval + generation eval harness  ·  S–M

**What.** A small `eval` package with the standard metrics: **nDCG, MRR,
recall@k, MAP** for retrieval; **perplexity, exact-match, ROUGE-ish overlap**
for generation; plus a runner that takes a labeled set and a pipeline and prints
a scoreboard.

**Why it fits aikit.** The project is already parity-obsessed (golden tests
everywhere); giving every feature a *quality* scoreboard, not just a
correctness one, is the natural extension. It makes #1 and #2 comparable
instead of vibes-based, and it's the kind of thing a serious retrieval library
is expected to have. (The generation-side metrics only apply when paired with
a generation backend — e.g. goinfer — the retrieval metrics stand alone.)

**Composes with.** Everything; especially `ann`.

### Tier 2 — classic ML primitives (cool, broadly useful, pure-Go)

#### 4. Embedding-space clustering — k-means(++), mini-batch  ·  S

K-means/k-means++ over the vectors `embed`/`encoder` produce: semantic
clustering, topic discovery, IVF cell construction (#2), and "summarize a corpus
into N themes." Small, classic, and a building block several other ideas reuse.
**Composes with** `ann`, `embed`, IVF (#2).

#### 5. Near-duplicate detection — SimHash / MinHash + LSH  ·  S

Fingerprint documents and find near-duplicates cheaply. Invaluable as a
*pre-indexing* cleaning step for RAG corpora (dedup before you embed and
index), and a genuinely useful standalone utility. Pure bit-twiddling, very
Go-friendly. **Composes with** `chunk`.

#### 6. Dimensionality reduction + Matryoshka embeddings  ·  S–M

**PCA / random projection** for shrinking embeddings and 2-D visualization, plus
first-class support for **Matryoshka** embeddings (use the first k dims of a
model trained for it — instant 2–4× storage cut at a known quality cost).
**Composes with** `embed`, `ann`, #2.

### How these sequence

A reasonable order by dependency and leverage:

1. **#2 quantized index (M)** — the foundation late-interaction needs to be
   practical at scale; do this first.
2. **#1 late-interaction (M)** — retrieval depth on top of the quantized
   index.
3. **#3 eval harness (S–M)** — makes #1 and #2 measurable instead of
   vibes-based.
4. **#5 near-duplicate detection (S)** — cheap, standalone, useful
   immediately as a corpus-cleaning step.
5. **#4 clustering / #6 dim-reduction (S–M)** — fill in as needed; several
   earlier items can reuse them (IVF cell construction, embedding
   visualization).

### Explicitly out of scope (for now)

Training / fine-tuning (aikit is inference-only by design), encoder-decoder and
non-transformer architectures (T5, Whisper, Mamba/SSM — different skeletons),
and anything requiring cgo or a GPU-only path (the WebGPU backend is the one
sanctioned accelerator, behind the existing `Backend` seam).

---

## Competitive context (refreshed 2026-06-11)

Unchanged from v3 in substance; deltas only:

- **hugot** — the last pipeline gap closed (cross-encoder, same checkpoint,
  Δ5e-6). The comparison is now fully head-to-head and framed honestly in
  the README (deployment tradeoff, not a drag race). Their Go-SIMD bet
  matures with archsimd (§2.5's trigger) — worth re-checking the comparison
  when that graduates.
- **Antfly/Termite** (Zig core), **coder/hnsw** (measured 0.22 vs 0.995),
  **Bleve/chromem-go/sqlite-vec** (index-only), **Ollama** (server) — all
  unchanged; the README table and capability matrix carry these.
- **Model2Vec upstream** — supported including standard format; watch for
  new potion releases (a better static model is a free quality bump through
  the parity pipeline).
