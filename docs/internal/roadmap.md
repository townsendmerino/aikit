# aikit roadmap v3 — post-1.4.0

> Rewritten 2026-06-11, after v1.4.0. v1 of this roadmap was captured at
> v1.1.0 from an external review; three release cycles (v1.2.0, v1.3.0,
> v1.4.0 — three days) executed v1 and v2 almost in full. Per-item
> annotations and measurements live in git history
> (`git log -- docs/internal/roadmap.md`) and the CHANGELOG; the v2 file at
> tag v1.4.0 is the last fully-annotated version.
>
> **Where this leaves the project:** the engineering backlog is effectively
> empty. The retrieval stack is feature-complete against the 2026 hybrid-
> retrieval bar (dense f32/int8, lexical, learned-sparse with in-process
> expansion, fusion, reranking, persistence/mmap/`//go:embed`, published
> benchmarks), parity-pinned throughout, fuzzed, and release-gated. What
> remains is almost entirely *adoption* work plus a short pre-graduation
> hygiene list. The roadmap is now correspondingly short — if the next
> session adds engineering items faster than adoption items, that's the
> failure mode to notice.

## The three-cycle scorecard (v1.2.0 → v1.4.0)

- **Perf**: amd64 W4A8 parity, scores·V vectorization (f32 + Q8 + goinfer
  prefill 3.4×), ann SIMD scoring (~7× Flat), HNSW build −20%/-7× allocs.
- **Features**: `sparse` end-to-end (index + in-process SPLADE expansion),
  `encoder.BERT` (MiniLM-class, parity 1.000000), standard-Model2Vec
  loading (potion-retrieval-32M), FlatI8 + int8 HNSW (recall Δ0), HNSW +
  FlatI8 persistence + zero-copy mmap, `QueryFilter`/base+delta+fuse,
  `Truncate`, `TokenizePlain`, RSF.
- **Quality**: HNSW Alg-4 default (recall@10 0.68 → 1.00, found by the
  `bench` harness), recall regression gate.
- **Robustness**: parser/dequant/chunk fuzzing (5 crashes fixed), kernel
  contract checks, mmap guardrails, nightly fuzz.
- **Health**: typed `ErrFormat` sentinels, Workspace-scoped parallel knobs,
  worker pool deleted, release gate automated (and it ran v1.4.0's release).
- **Proof**: comparative benchmarks in the README (aikit ~0.995 recall vs
  coder/hnsw ~0.22, chromem-go ~45× p50), `examples/embedded-corpus`
  (~70 MB single binary, ~50 ms startup, zero external files).
- **Measured-and-closed**: int8 M-tiling, worker pool, K-crossover,
  GOEXPERIMENT=simd spike. **Off-roadmap adds**: `MatmulBTAcc64`, `DotI8`.

---

## 1. Adoption — the entire critical path

Everything high-impact that remains is here, and none of it is blocked.

1. **Cross-encoder reranking head (`encoder.LoadCrossEncoder`)** — ✅ **DONE.**
   As predicted, a small additive step over v1.4.0's BERT trunk: `LoadCrossEncoder`
   reuses `LoadBERT` (prefix-aware) and adds the pooler + a classification head, and
   `hiddenStates` gained token-type segments for the query/document pair
   ([CLS] q [SEP] d [SEP], types 0/1). `Score(query, doc)` → relevance logit; pins
   **ms-marco-MiniLM-L-6-v2** (hugot's CrossEncoders headline + Antfly's reranker
   default) at Δ 5e-6 forward AND end-to-end (aikit's own pair tokenization matches
   HF) — golden via `pin_crossencoder.py`. aikit now covers both halves of the modern
   rerank story (bi-encoder + cross-encoder), cgo-free; closes the last pipeline gap
   vs hugot/Antfly. Additive (apidiff: `CrossEncoder` + `LoadCrossEncoder` added).
2. **One named external adopter** — [high / not-engineering]. Three
   releases of recruiting assets exist (benchmark table, embedded-corpus
   demo, BERT/SPLADE support); the road-to-1.0 critique's warning now
   applies to 1.4. The natural shapes: an MCP server `//go:embed`-ing a
   docs corpus; or ken itself publicly badged as built-on-aikit with the
   embedded-corpus pattern productized. Ship v1.5 *with* one.
3. **Announcement post** — [medium-high / low]. *New:* there is now a
   story worth telling ("a pure-Go, no-cgo retrieval stack: BERT-family
   inference, SPLADE, int8 ANN, one static binary — with parity proofs and
   honest benchmarks") and three days of measure-fix-measure material (the
   0.68→1.00 recall find is a genuinely good engineering-blog arc). A post
   + Show HN / r/golang is the cheapest adopter-recruiting channel that
   exists. Blocked on nothing.
4. **Benchmark remainder** — [medium / medium]. (a) An inference-side
   throughput row vs hugot — **now apples-to-apples** since both run
   all-MiniLM-L6-v2 (aikit pure-Go vs hugot's Go backend vs hugot's ONNX);
   (b) a BEIR/VectorDBBench slice for cross-referenceable absolute numbers.
   Both strengthen items 2–3.

## 2. Pre-graduation hygiene — short, then stop

1. **`linalg` surface audit** — [low-medium / low]. The last v2 engineering
   item still open. The Experimental surface grew opportunistically
   (`DotI8`, `MatmulBTAcc64`, the Workspace methods); do a deliberate
   keep/unexport pass with goinfer's actual usage as consumer evidence
   before any graduation promise. Do it soon — goinfer builds against
   these exports daily and each day raises the cost.
2. **Blob format-stability policy** — [low-medium / low]. *Rises with
   every step toward an adopter:* formats burned v1→v2→v3 in 48 hours,
   fine pre-circulation, but an adopter embedding blobs in *their*
   releases changes the calculus. Decide and document ("Load reads N−1" or
   "rebuild per minor"), reserve header bytes, and bundle the HNSW
   zero-copy alignment fix (§3.1) into whatever bump comes next.

## 3. Gated — do not start without the stated trigger

1. **HNSW zero-copy mmap** — needs a format-v4 alignment bump; bundle with
   the §2.2 policy work, never standalone.
2. **Binary/Hamming pre-filter + f32 rescore** — corpus sizes aikit doesn't
   see yet; trigger: an adopter with >1M vectors.
3. **Windows real mmap** — trigger: a sizable Windows consumer.
4. **Hardware-gated x86 tail** (VNNI W4A8, AVX-512 tier) — trigger: Zen 4+ /
   Cascade Lake+ access; a cloud c7i/c7a spot instance remains the cheap
   unblock. Bundle both when it happens.
5. **`GOEXPERIMENT=simd` portable kernels** — trigger: archsimd ships arm64
   AND graduates (still amd64-only + gated as of June 2026; arm64 via SVE
   planned on dev.simd for Go 1.27). Then re-spike; if within ~10% of the
   hand asm, migrate and delete the `.s` files. Note hugot expects 3–10×
   from this same graduation — another reason §1.3-4 publish now.
6. **In-place index mutation** — trigger: a real consumer proving
   rebuild/delta/swap insufficient. Design rule 4 holds until then.
7. **AMX** — out of scope.

---

## Competitive context (refreshed 2026-06-11)

- **hugot** — the comparison flipped from aspiration to head-to-head:
  aikit now runs the same MiniLM models (and SPLADE, which hugot lacks),
  cgo-free vs ONNX-runtime; §1.1 (cross-encoder) closes the last pipeline
  hugot has that aikit doesn't, and §1.4(a) measures the rest. Their Go
  SIMD bet (3–10×) matures when archsimd graduates — aikit's window for
  publishing a perf lead is now.
- **Antfly/Termite** — core now Zig; feature bar matched in library form,
  including SPLADE end-to-end as of v1.4.0. Their reranker example's
  default model is exactly §1.1's target.
- **coder/hnsw** — measured at recall@10 ~0.22 on real embeddings vs
  aikit ~0.995 (construction-limited; plain selection vs Alg-4). Keep the
  README table honest if upstream improves.
- **Bleve / chromem-go / sqlite-vec / LanceDB-go** — index-only, no
  embedded inference; unchanged moat.
- **Ollama** — unchanged (server dependency); aikit's pitch stays
  in-process, zero-deploy, now with a runnable 70 MB-binary proof.
- **Model2Vec upstream** — potion-retrieval-32M supported as of v1.4.0;
  parity discipline inherits upstream gains. Watch for new potion releases.
