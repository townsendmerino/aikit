# aikit roadmap v2 — post-1.2.0

> Rewritten 2026-06-09, the day v1.2.0 shipped, from a fresh external review
> pass. v1 of this roadmap (captured at v1.1.0) was executed almost in full in
> one release cycle; its per-item annotations and measurements live in git
> history (`git log -- docs/internal/roadmap.md`) and the CHANGELOG. Sections
> ordered by importance; items within likewise; **[impact / effort]** tags.
>
> **The binding constraint has moved.** v1's gaps were engineering; those are
> closed. What remains is gated on three external things — an audience,
> a Python parity toolchain, and x86 hardware — plus coherence follow-ups.
> The priority order below reflects that inversion: proof-and-adoption work
> now outranks new engineering.

## Shipped in v1.2.0 (v1 roadmap scorecard)

- §1 perf: amd64 AVX2 W4A8 (closing the marquee cliff), scores·V attention
  vectorization (~2.85× long-seq Encode; QKᵀ premise inverted by profiling),
  ann SIMD scoring (~7× Flat scan), K-crossover (non-issue, doc fix).
- §2 features: `sparse` (index half), HNSW persistence, `bench` harness,
  `FlatI8`, `TokenizePlain`, `QueryFilter`+base/delta/fuse (immutability made
  design rule 4), `embed.Truncate`, pooling seam (internal).
- §3 robustness: complete — parser+dequant fuzzing (5 crashes found/fixed),
  chunk fuzzing, build-tagged kernel contract checks, mmap guardrail, nightly
  long-fuzz CI.
- §4 quality: complete — HNSW Alg-4 default (recall@10 0.68 → 1.00), recall
  regression gate, RSF.
- Measured-and-closed: int8 M-tiling (no win), `GOEXPERIMENT=simd` (deferred,
  see Watch list), AVX-512/VNNI (hardware-gated, see Watch list).
- §6 docs: architecture.md, linalg doc.go, ken-free quickstart.

---

## 1. Adoption & proof — now the top of the roadmap

The code is ahead of its audience (the road-to-1.0 critique's point, still
true two releases later). Everything here is unblocked today.

1. **Comparative benchmark table, published in the README** — [high /
   low-medium]. Now fully unblocked: `bench` exists and earned credibility by
   finding the Alg-4 recall problem. Run aikit vs **hugot's pure-Go backend**
   (now directly comparable: it runs MiniLM-class models and ships a
   CrossEncoders pipeline — it is no longer adjacent, it overlaps),
   coder/hnsw, Bleve vector, chromem-go: recall@k, p50/p95, build time,
   memory, binary size, cgo-or-not. Add a BEIR/VectorDBBench slice for
   absolute numbers people can cross-reference. hugot is publicly betting on
   Go SIMD for a future 3–10× pure-Go win — publish while the perf lead from
   the hand-tuned kernels is large and measurable.
2. **`examples/embedded-corpus` showcase** — [high / low]. Also fully
   unblocked (HNSW `MarshalBinary`+`Load` shipped): one `main.go` that
   `//go:embed`s corpus + Model2Vec model + prebuilt HNSW index and serves
   hybrid search from a single static binary, zero downloads at runtime. This
   is the pattern neither Python nor hugot-on-ONNX can match; it's currently
   prose in the README. Pairs with §3.1 (FlatI8 persistence) for the smallest
   possible blob.
3. **One named external adopter** — [high / not-engineering]. Unchanged from
   v1 and still the highest-leverage non-code item: ship the next minor
   *with* a consumer (an MCP server embedding a docs corpus is the natural
   shape). §1.1 and §1.2 are the marketing assets for recruiting one.
4. **Release-process gate in CI** — [medium / low]. *New; from observed
   misses this cycle:* v1.1.1 was tagged with no CHANGELOG entry, and v1.2.0
   nearly shipped with stale README/DAG docs (caught in review, fixed
   post-tag). Add a tag-triggered CI job: tag has a matching `## [x.y.z]`
   CHANGELOG section; `apidiff` vs previous tag shows no Hard-tier breakage
   (mechanizes the stated release bar); working-tree docs that mention
   package deps match `go list` (a 20-line script). Cheap insurance against
   the only two release defects observed so far.

## 2. Model-blocked track — one toolchain unblocks four items

Every remaining model-side feature is parity-blocked the same way: it needs
torch + sentence-transformers + a checkpoint to generate golden fixtures.
Items 2–5 are sequenced behind item 1.

1. **Stand up the Python parity toolchain** — [enabler / low-medium]. A
   pinned `.venv` (torch-cpu is fine) + the existing `scripts/pin_*.py`
   pattern, runnable on the Mac. Nothing new intellectually — it's the same
   oracle discipline the repo already lives by; it just needs an environment
   that has it.
2. **MiniLM-class encoder support (§2.5 remainder)** — [high / high]. The
   pooling axis is done (internal seam); remaining axes are learned absolute
   positions + GELU FFN + the sentence-transformers config loader, each
   parity-pinned one at a time against all-MiniLM-L6-v2. Raised in priority
   since v1: hugot's CrossEncoders pipeline now serves this exact use case,
   and MiniLM is *the* commodity reranker/embedder family — supporting it
   converts aikit from "two specific models" to "the BERT family you already
   use", cgo-free. Graduating the pooling knob to public comes free with the
   first mean-pooled golden.
3. **SPLADE expansion head (§2.1 remainder)** — [medium-high / high]. The
   index/scorer half shipped; the in-process masked-LM head (logits →
   log(1+ReLU), max-pool) reuses `encoder`'s machinery and needs a SPLADE
   checkpoint + golden. Until then, an interim unlock at [low / low]: a
   `scripts/pin_splade.py` that emits `SparseVec` JSON out-of-band, so the
   shipped index half is *usable end-to-end* today — document the recipe in
   the `sparse` package example.
4. **potion-retrieval-32M parity pin** — [medium / low]. *New.* The README
   quickstart fetches potion-code-16M; upstream's best static *retrieval*
   model is potion-retrieval-32M. `embed` should load it already (same
   format) — add it to the parity matrix and the docs so general-retrieval
   users land on the right model, not the code-tuned one.
5. **forward_q8 scores·V vectorization** — [low / low, fold into #2]. The
   Q8 forward still has the scalar context loop the f32 paths lost; it's
   dormant (off the default path, not model-covered). Fix when the Q8 path
   next gets exercised — not worth touching un-oracled.

## 3. Embedded-index coherence — finish what v1.2.0 started

The `//go:embed` story is the moat; these make it uniform. All unblocked.

1. **`FlatI8.MarshalBinary` + `Load`** — [medium / low]. HNSW persists;
   FlatI8 (the ¼-memory index, the one you'd *most* want to embed) doesn't.
   Same versioned-format + integrity-validation + fuzz discipline as
   `FuzzLoadHNSW`.
2. **Zero-copy mmap `ann.Load`** — [medium / medium]. The v2 format already
   lays vectors out as one contiguous little-endian block; `Load` currently
   copies. An mmap-aliasing variant gives instant startup on big indexes —
   with the same lifetime contract (and guardrail pattern) as `embed`'s mmap
   loaders.
3. **Int8 HNSW** — [medium / medium]. `FlatI8` proved int8 recall holds
   (1.00 real / 0.986 adversarial); HNSW's `sim()` doing int8 dots during
   build+search extends the ¼-memory win to the log-scale index. Gate on the
   recall regression test before flipping anything.
4. **Binary/Hamming pre-filter + f32 rescore** — [low-medium / medium]. The
   third compression tier (32× candidate filter). Only worth it at corpus
   sizes aikit doesn't see yet — keep behind §1.3's adopter signal.
5. **Windows real mmap** (`CreateFileMapping`) — [low / low-medium].
   Carried from v1, unchanged: matters when a sizable Windows consumer
   exists. Slightly more relevant if §3.2 lands (mmap'd indexes).

## 4. API & code health — the untouched v1 section

Carried over intact; all three remain open and got *more* relevant as the
Experimental tier grows toward graduation.

1. **Typed sentinel errors** (`embed.ErrBadMagic`, `ann.ErrFormat`, …) —
   [medium / low]. Additive. The new `Load` paths (HNSW, future FlatI8) make
   this more pressing: callers handling corrupt-blob errors are matching
   strings today.
2. **Scope the global knobs** (`linalg.SetParallelThreshold/Width`) —
   [low-medium / medium]. Per-`Workspace`/per-call overrides, globals as
   defaults. Do before `linalg` graduates from Experimental — cheap now,
   breaking later.
3. **Decide the worker pool's fate** — [low / low]. Still built, still
   measured-neutral, still shipped-but-unused. Delete it (the measurement
   note in git history is the record) or mark deprecated. Pick one this
   cycle.

## 5. Watch list — externally gated, with explicit triggers

| Item | Gate | Trigger to act |
|---|---|---|
| VNNI `VPDPBUSD` W4A8 variant | no Zen 4+/Cascade Lake+ box | hardware access (a cloud c7i/c7a spot instance is a cheap unblock — consider renting instead of waiting) |
| AVX-512 dispatch tier | same hardware gap + downclocking caveats | same box; bundle with VNNI work |
| `GOEXPERIMENT=simd` portable kernels | archsimd is amd64-only + experiment-gated (confirmed still true, June 2026; arm64 planned via SVE on dev.simd for 1.27) | BOTH: arm64 support lands AND experiment graduates. Then re-run the spike; if within ~10% of hand asm, migrate and delete `.s` files. Note: when this graduates, hugot's pure-Go backend gets its predicted 3–10× — aikit's kernel lead narrows, which is *also* why §1.1's benchmark table should publish now |
| In-place index mutation | design rule 4 (immutability cornerstone) | a real consumer demonstrating rebuild/delta/swap insufficiency — not before |
| AMX | out of scope | — |

---

## Competitive context (refreshed 2026-06-09)

- **hugot** (knights-analytics) — *moved the most since v1.* Now ships
  CrossEncoders (reranking — direct overlap with `encoder`), text-generation
  and image-classification pipelines; pure-Go backend targets exactly the
  MiniLM-class models §2.2 would add; publicly expecting 3–10× from Go SIMD.
  Still cgo (ONNX Runtime) for full speed. aikit's answers: no-cgo + hand
  kernels today (§1.1: publish the comparison), MiniLM parity (§2.2), and
  the embedded-binary pattern hugot's ONNX dependency can't follow (§1.2).
- **Antfly/Termite** — core rewritten in **Zig** (May 2026); the Go engine
  story is now historical. Still the feature bar for hybrid retrieval
  (sparse + dense + lexical + published recall numbers) — aikit now matches
  the retrieval-feature list in library form except SPLADE inference (§2.3).
  Their exit from Go arguably *widens* aikit's lane: one fewer serious
  pure-Go retrieval engine.
- **coder/hnsw** — unchanged; aikit now exceeds it (persistence + Alg-4 +
  SIMD scoring + int8). Beat it in the §1.1 table to make that legible.
- **Bleve / chromem-go / sqlite-vec / LanceDB-go** — index-only; no embedded
  inference. Unchanged moat.
- **Ollama** — unchanged: server dependency, awkward rerankers. aikit's
  pitch stays in-process, zero-deploy.
- **Model2Vec upstream (MinishLab)** — potion-retrieval-32M remains the best
  static retrieval model; §2.4 pins it. Parity discipline keeps inheriting
  upstream gains cheaply.
- **Go 1.26 archsimd** — amd64-only, GOEXPERIMENT-gated; arm64 (SVE) and a
  portable high-level API are explicitly future work on dev.simd. The
  watch-list trigger stands.
