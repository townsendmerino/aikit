# aikit roadmap — post-1.1.0 improvement plan

> Captured 2026-06-09 from an external full-repo review (maintainability, CPU
> perf, competitiveness, docs). Sections are ordered by importance; items
> within each section likewise. Each item carries **[impact / effort]**.
> Competitive context at the bottom informs §2.

## Progress (2026-06-09)

- **§1.1 amd64 fused W4A8 kernel — AVX2 DONE** (`e6f30dd`, landed from the Zen 2
  box; predates this batch). The marquee perf cliff is closed; only the VNNI
  variant remains. See §1.1 below.
- **§1.3 Per-head attention — DONE (pivoted).** Profiled `Model.Encode` on real
  weights: QKᵀ is already SIMD (~2.6%) → closed; the real hotspot was the scalar
  scores·V context loop (~⅓ of `Encode`), now vectorized via the SIMD matmul.
  ~2.85× single `Encode` at long sequences (L²-scaled), bit-exact. See §1.3.
- **`ann` SIMD similarity — DONE** (follow-up to 58c947b's pattern). `Flat.Query`
  and `HNSW.sim` were scalar `float64` dot loops not using `linalg`; now call
  `linalg.Dot`; `Flat.Query` then streams 8 vecs/pass through `Dot8x4` (shared-q
  a-reuse). ~7× Flat scan total (d=128: 5.18→0.72 ms; +1.6× from streaming over
  the per-vector swap, near memory bandwidth), ~1.4× HNSW search. Decision:
  `float32`-precision scores accepted (HNSW approximate → silent; Flat documented,
  recall verified unchanged — 0 tie-flips vs float64; the recall test also caught
  an arm64 Dot8x4 4-lane-partial-sum bug before it shipped). Validation later
  closed to the original spec (`task-ann-simd-dots.md`): a real-embedding recall
  check (Model2Vec, 600 clustered vectors → 0 tie-flips) and the d=768 / N=10k,100k
  benchmark grid (bandwidth-bound, 21–37 GB/s).
- **§1.4 Int8 M-loop register tiling for W8A8 — CLOSED (measured no win).** Built
  + benchmarked the 4-row arm64 SDOT tile (bit-identical); M=64 was a statistical
  tie. W8A8 at M>1 is SDOT-throughput-bound (~3 SDOT/cycle, near M1 peak) and the
  column-outer reblock already took the reuse — nothing for register tiling to
  exploit. Reverted. See §1.4.
- **§1.5 `GOEXPERIMENT=simd` portable kernels — SPIKED → DEFER.** A portable
  `dotF32` over `simd/archsimd` compiles for amd64, but the package is **amd64-only
  (no arm64/NEON)** and **gated behind `GOEXPERIMENT=simd`** (breaks plain
  `go build` for all consumers). Doesn't deliver the cross-arch "write once" win
  and can't gate a 1.0 lib on an experiment. Revisit when arm64 lands + it
  graduates. See §1.5.
- **§3.1 Fuzz binary parsers — DONE (parse + dequant).** Parse: `FuzzParseGGUF` /
  `FuzzParseSafetensors` / `FuzzParseShardIndex` + seed corpus; found & fixed an
  OOM (`make(map, ~5e10)` from an untrusted `tensorCount`), a negative-length
  slice panic, and a safetensors `8+headerLen` overflow. Dequant (`FuzzGGUFDequant`,
  `63c73ea`): found & fixed two more — `∏dims` overflowing `int` → `make([]float32,
  n)` OOM/panic, and `offset+nbytes` overflowing `uint64` → slice panic (same
  class as the safetensors fix). All four repros are committed regression seeds;
  CI runs a 20s/target fuzz smoke over all four targets.
- **§1.2 Dot8x4 K-crossover — DONE (as a doc fix, not a code change).** The
  premise was wrong: the encoder tiles K at `kBlockDefault=768` (Dot8x4's peak),
  so it never feeds a large-K strip — `fc2` (K=3072) runs as 4×768. A call-site
  conditional would be dead code. The real gap was the *public* `Dot8x4` godoc;
  it now documents the cliff + "tile K to ≤~768".
  - **DEFERRED (2026-06-09):** also considered adding a note at the
    `kBlockDefault` const documenting that 768 doubles as Dot8x4's throughput
    peak (so a future tuner doesn't raise kBlock past the cliff). Decided not to
    — the constraint already lives in `linalg.Dot8x4`'s godoc, and the encoder
    comment didn't need the extra coupling. Revisit only if someone retunes the
    tile sizes per arch (which would touch this comment anyway).
- **§6.5 linalg `doc.go` — DONE.** Package doc with the kernel-dispatch map.
- **§6.1 ken-free quick start — DONE.** README model-fetch now uses
  `huggingface-cli` directly; `ken` noted as an optional shortcut.
- **§5.4 minor nits — N/A (already resolved).** Nothing is tracked
  (`git ls-files` clean) and `.gitignore` already covers `.DS_Store` / `*.test` /
  `__pycache__`. The reviewer saw working-tree artifacts, not committed files.

---

## 1. CPU performance — close the amd64 gap

The arm64 story is strong (fused W4A8 SDOT, NEON, cache-blocked matmul). The
amd64 story is not, and x86 Linux servers are where most non-Apple deploys
land. goinfer inherits every one of these.

1. **amd64 fused W4A8 kernel** — ✅ **AVX2 DONE** (`e6f30dd`,
   `dot_w4a8_amd64.s`). The single largest perf cliff is closed: the int4×int8
   decode kernel now has an amd64 AVX2 path (nibble-unpack prologue +
   `dotI8AVX2` sign-extend body), validated on Zen 2 — ~1.7–1.9× of W8A8, ~32×
   faster than the Q4 f32 path, on par with arm64 SDOT. Non-AVX2 amd64 keeps the
   scalar fallback. **Remaining: the VNNI `VPDPBUSD` variant** (one instruction
   for the VPMOVSXBW+VPMADDWD pair) behind the same CPUID gate, for Zen 4+ /
   Cascade Lake+ — drop-in for a VNNI-capable box; the AVX2 path is the proven
   fallback. [low-med / medium] — see also §1.6 (AVX-512/VNNI tier).
2. **Wire the K-dependent kernel crossover** — ✅ **DONE (non-issue + doc fix).**
   The premise didn't survive the code: the encoder tiles K at `kBlockDefault=768`
   (exactly `Dot8x4`'s peak), so the sole caller never feeds it a large-K strip —
   `fc2` (K=3072) runs as 4×768. A call-site conditional would be dead code. The
   real exposure was the *public* `linalg.Dot8x4` godoc, which now documents the
   large-K cliff + "tile K to ≤~768". See the Progress note above.
3. **Per-head attention QKᵀ parallelism — DONE: QKᵀ CLOSED, scores·V vectorized
   instead.** Profiling `Model.Encode` on real weights (the item's own mandate)
   inverted the premise: QKᵀ is already SIMD and ~2.6% of `Encode`, so the
   microbench's 3.4× is irrelevant — closed. The real hotspot was the **scores·V
   context loop** (scalar, ~⅓ of `Encode`), now routed through the SIMD matmul in
   both `selfAttention`/`selfAttentionBatched`. ~2.85× single `Encode` at ~500
   tokens (L²-scaled; neutral at short rerank passages), bit-exact. The
   cross-repo transfer landed: goinfer's **prefill** had the same hotspot — and
   there BOTH terms were scalar (QKᵀ 26% + scores·V 16% = 49% of forward) — now
   vectorized onto `MatmulBT` (`goinfer@7fa82c2`, prefill 12.3→42.1 tok/s, 3.4×,
   parity argmax-exact). Decode (M=1) correctly untouched — scores·V is a gemv
   there. *Remaining follow-up:* aikit's `forward_q8.go` has the same scalar loop
   but is dormant (off the default `Encode` path, not model-test-covered).
4. **Int8 multi-row register tiling (M-loop) for W8A8** — ❌ **CLOSED: measured
   no win.** Built the 4-row arm64 SDOT tile (`dotI8x4SDOT`: weight row held in a
   register, reused across 4 activation rows — one b-load + one reduction per 4
   rows instead of per row), bit-identical to the column-outer path. Benchmarked
   M=64 (K2048,N2048), `-count=5`: tile 1703µs vs per-row 1713µs — a statistical
   tie. W8A8 at M>1 runs at ~3 SDOT/cycle (near the M1 NEON peak): it's
   SDOT-throughput-bound, and the column-outer reblock already captured the
   weight reuse at the cache level, so register-level reuse / fewer reductions
   have nothing to exploit. amd64's int8 dot is *more* instruction-heavy
   (VPMOVSXBW+VPMADDWD, no SDOT), so even less likely load-bound there. Reverted
   the kernel — no complexity for a no-op. Revisit only if a real amd64 prefill
   profile shows W8A8 load-bound (the arm64 evidence says it won't).
5. **Evaluate Go 1.26 `GOEXPERIMENT=simd` (go-highway-style portable kernels)** —
   ⏸️ **SPIKED → DEFER (two hard blockers).** The package exists (`simd/archsimd`,
   Go 1.26.3); a portable `dotF32` over it is clean (`Float32x8`,
   `LoadFloat32x8Slice`, `.Mul`/`.Add`/`.MulAdd`-FMA) and cross-compiles for
   amd64. But:
   1. **amd64-only — no arm64/NEON.** Every impl file is `*_amd64.go`; the spike
      fails to build for `GOARCH=arm64` (`undefined: archsimd.Float32x8`). So it
      does NOT deliver the "write once for all arches" premise — arm64 hand-asm
      stays, and you'd maintain portable-amd64 *plus* hand-NEON (a wash, or worse,
      vs today's two asm sets).
   2. **`GOEXPERIMENT=simd`-gated.** Plain `go build` fails ("build constraints
      exclude all Go files"). Adopting it would force every downstream consumer
      (ken, goinfer, end users) to set the flag — breaking aikit's frictionless
      pure-Go-no-special-flags build, a core selling point. Unacceptable for a 1.0
      lib while it's still an experiment.

   The "within ~10%?" perf question is therefore moot for now — even at parity,
   the blockers prevent adoption. **Revisit when BOTH clear:** archsimd gains
   arm64/NEON AND graduates from GOEXPERIMENT (default-on). Then re-run the spike:
   port `dotF32`/`dotI8`, bench portable-vs-asm on both arches, and if within ~10%
   migrate the kernels and delete the `.s` files (the real bus-factor win). Track
   the upstream Go proposal for arm64 + graduation.
6. **AVX-512/VNNI dispatch tier** — ⏸️ **DEFER (no test hardware + precondition
   failed).** Assessed 2026-06-09. Blocked on the same hardware gap that defers
   the VNNI W4A8 remainder: AVX-512 and VNNI need Zen 4+ / Intel Cascade Lake+,
   and NEITHER available box has it — this host is arm64 (no x86 SIMD), and the
   amd64 validation box is a Zen 2 Ryzen 7 3700X (AVX2 only). Writing the kernels
   + CPUID detection blind = three unvalidatable things in a 1.0 lib (the asm only
   runs on hardware we can't test; the detection can't be confirmed to return
   *true* correctly). And the item's own precondition — "revisit if #5 makes it
   nearly free" — failed: §1.5 deferred. Plus the standing caveats (downclocking,
   AVX2 ubiquity). Revisit when a Zen 4+/Cascade-Lake+ box is available. AMX: out
   of scope.

   **Hardware-gated tail (all unblock on one Zen 4+/Cascade-Lake+ box):** this
   item; the §1.1 VNNI `VPDPBUSD` W4A8 variant; and (separately) §1.5 once archsimd
   also ships arm64 + graduates. The AVX2 paths are the proven fallback for all.
7. **Windows real mmap** (`CreateFileMapping`) instead of heap-read fallback —
   [low / low-medium]. Matters once any sizable consumer runs on Windows.

## 2. Competitive differentiation — features

Positioning note: Antfly/Termite is a *server product* (Elastic License v2,
Raft, Pebble, dashboards) — not a direct competitor to an MIT-licensed,
embeddable, cgo-free *library*. The competitive play is to be the best
building blocks that products like that would otherwise build in-house —
and to cover their headline retrieval features in library form. hugot's full
speed requires ONNX Runtime (cgo); aikit's no-cgo lane stays open.

1. **Learned sparse retrieval (SPLADE-style) package** — 🟡 **INDEX HALF DONE;
   inference half remains.** Shipped the `sparse` package (Experimental tier):
   `SparseVec`, an inverted `Index`, and `Query` (sparse-dot top-k) that fuses
   into `fuse.RRF` via `Hit.Index` (matches `ann.Hit`). Inference-optional — it
   indexes/scores pre-computed vectors from any SPLADE model; validated against a
   brute-force reference, pure Go, concurrent-Query-safe. **Remaining (the bigger
   lift):** the in-process masked-LM expansion head that produces the vectors —
   a small MLM head over `encoder`'s NomicBert machinery (logits → log(1+ReLU)
   max-pool over the vocab), parity-pinned to a Python SPLADE reference like the
   other model paths. That's model-dependent (needs a SPLADE checkpoint + golden
   fixtures); the index half is fully usable without it.
2. **HNSW persistence: serialize/load** — ✅ **DONE.** `HNSW.MarshalBinary` (also
   `encoding.BinaryMarshaler`) + `ann.Load([]byte)` — the `//go:embed`-an-index
   pattern (build offline, embed, load query-ready at startup). Format versioned
   from day one (magic + version); `Load` validates graph integrity (out-of-range
   ids, layer-inconsistent edges, truncation, config bounds) so a hostile/corrupt
   blob errors instead of panicking/OOM-ing (fuzzed, `FuzzLoadHNSW`); round-trip
   reproduces identical `Query` results, `MarshalBinary` deterministic.
   Experimental tier. *Follow-up:* a true **zero-copy mmap `Load`** (vectors
   aliasing the mapped bytes, no copy) — the format is already a contiguous
   little-endian vector block, so it's mmap-friendly; the current `Load` copies.
3. **Published recall/latency benchmark harness** — ✅ **DONE (and it already
   earned its keep).** `bench.Run(corpus, queries, k, cfg)` measures recall@k vs
   exact Flat + p50/p95/p99 latency + build time + memory for Flat/HNSW/FlatI8,
   rendered as a Markdown `Table`; synthetic (reproducible) and real-embedding
   (model-gated) harness tests, with FlatI8 recall as a machine-independent
   regression gate. **First run surfaced a real finding** (see §4.1): on clustered
   real embeddings HNSW recall@10 caps ~0.68 and barely moves with ef, while
   random vectors hit 1.0 at ef=256 — the Alg-3 limitation the old random/d=64
   test (0.99) hid. *Follow-ups:* a cross-library comparison table (§6.2, needs
   hugot/coder-hnsw runs) and a VectorDBBench/BEIR slice for absolute numbers.
4. **Quantized ANN storage (int8 vectors)** — ✅ **DONE (Flat).** `ann.FlatI8`
   stores each unit vector as int8 + a per-vector scale (¼ the float32 footprint)
   and scores via `linalg.MatmulBTW8A8` at M=1 (dynamic query quant, SIMD +
   parallel) — reusing the existing int8 kernels exactly as the roadmap noted.
   Same `Hit`/`Query` shape as `Flat`, fuses identically. Measured: recall@10 vs
   exact float32 Flat = **1.00 on real Model2Vec embeddings**, 0.986 on
   adversarial random; **3.94× smaller**. *Follow-ups:* `FlatI8` persistence
   (MarshalBinary, pairs with §2.2 for an even smaller embedded blob); an int8
   HNSW (its `sim()` would int8-dot during search/build); and the **binary/Hamming
   pre-filter + f32 rescore** the item flagged.
5. **Generalize `encoder` beyond NomicBert (config-driven BERT-family
   loader)** — [medium / high]. Today: CodeRankEmbed only. Termite/hugot run
   arbitrary HF models. Full generality is ONNX-shaped (don't go there);
   instead parameterize the existing forward pass (layers, heads, RoPE vs
   learned positions, pooling) to cover the popular MiniLM / bge /
   mxbai-class rerankers and embedders from safetensors. Each new
   architecture stays parity-pinned like CodeRankEmbed.
6. **General-NLP tokenizer option for `bm25`** — ✅ **DONE.** `bm25.TokenizePlain`
   is a Unicode word tokenizer (lowercase, split on any non-letter/non-digit, no
   snake/camel identifier splitting) alongside the code-tuned `Tokenize`. `Build`/
   `Query` take pre-tokenized docs, so callers pick per corpus; `Tokenize` stays
   the code-RAG default. Tests pin the contract (e.g. `getUserName` → one token
   under plain, split under code) + a runnable Example. Widens the audience to
   natural-language corpora.
7. **Streaming / incremental index updates** — ✅ **DONE the low-risk way (and
   immutability is now a cornerstone, design rule 4).** Rather than make the
   indexes mutable — which would trade away lock-free reads + snapshot consistency
   for a mutable-database concern outside aikit's niche — the changing-corpus cases
   are served WITHOUT mutation: (a) **logical delete** via `QueryFilter(q, k, keep)`
   on `Flat`/`HNSW`/`FlatI8` — a caller-supplied live-set predicate applied at query
   time (Flat/FlatI8 exact; HNSW routes through filtered nodes, recall@10 = 1.00
   under 20% deletion); (b) **base + delta + fuse** — a small frequently-rebuilt
   delta index fused with the big base (`Example_baseDeltaFusion`); (c)
   rebuild-and-swap (the existing ken pattern). True in-place mutation (HNSW
   tombstone graph-repair, concurrent Add-during-Query, incremental BM25 segments)
   stays **explicitly out of scope** per the cornerstone — revisit only if a real
   consumer proves rebuild/delta/swap is insufficient at scale.
8. **Matryoshka dimension truncation support in `embed`/`ann`** — ✅ **DONE.**
   `embed.Truncate(v, dim)` returns the L2-renormalized prefix (fresh slice); the
   indexes are already dim-agnostic. Composes with §2.4's FlatI8 for a compounded
   cut (256→128 int8 = 8× vs 256-d f32). Measured on the bundled Model2Vec slice
   (`TestMatryoshkaRecall`): recall@10 holds at 0.86 down to half the dim (256→128),
   degrading only below — so half-dim truncation is free here. Documented as
   MRL-only (truncating a non-MRL embedding degrades it).

## 3. Robustness & correctness

1. **Fuzz the binary parsers** (`embed/gguf.go`, `embed/safetensors.go`,
   GGUF dequant paths) — ✅ **DONE**. Four native fuzz targets (parse ×3 +
   `FuzzGGUFDequant`); found & fixed 5 crashes (untrusted-count OOM, two int
   overflows wrapping bounds checks, a negative-length and the ∏dims overflow).
   Four committed regression seeds; CI runs a 20s/target smoke. **Remaining:**
   the "longer nightly" run (the PR smoke is the short pass); and `chunk`
   tokenizer/scanner fuzzing (§3.4) is still open.
2. **Debug-build alignment asserts in quant kernels** — [medium / low].
   `DequantizeRow`/W4A8 group paths trust K alignment (caller contract).
   A build-tagged (`//go:build aikit_checks`) assert layer catches misuse
   without taxing the hot path.
3. **Mmap-lifetime guardrail** — [medium / low-medium]. Tensor slices outlive
   `Close()` only by documented contract. Options: a `runtime.SetFinalizer`
   + use-after-close poison in debug builds, or a `vet`-style doc example.
   At minimum, promote the contract into the package doc with a WRONG/RIGHT
   example pair.
4. **Fuzz `chunk` tokenizer/scanner paths** (`bm25.Tokenize`, markdown
   scanner, regex chunkers) — [low-medium / low]. Same mechanism as #1;
   panics in a chunker take down an indexing pipeline.

## 4. Retrieval quality

1. **HNSW Algorithm-4 neighbor-diversity heuristic** — ✅ **DONE, and it's the
   default now.** The §2.3 harness measured the damage (Alg-3 recall@10 capped
   ~0.68 on clustered real embeddings, barely moving with ef), so `selectHeuristic`
   (Algorithm 4) was implemented and — since it won decisively — made the default.
   Measured with the same harness: recall@10 **0.68 → 1.00** on the real Model2Vec
   corpus (and 0.57 → 1.00 on a synthetic clustered set, the model-free regression
   test), at ~2× build cost and unchanged query latency; neutral on random data.
   `Config.SimpleNeighbors` opts back to Alg-3. Persisted format bumped to v2 (one
   selection-mode byte). Textbook measure-fix-measure: the harness found it, drove
   the fix, and verified it. *Build-speed follow-up (done):* profiling the build
   found two pure-overhead hotspots — a per-search `map` and `container/heap`'s
   `interface{}` boxing (~23M allocs/build) — replaced with a gen-stamped visited
   buffer + a concrete typed heap: ~20% faster, 7× fewer allocs, recall identical.
   The residual cost is the Alg-4 diversity dots (inherent). A `BenchmarkHNSWBuild`
   now guards it.
2. **Recall regression tests on a real slice** — ✅ **DONE.** A 50-doc / 5-topic
   real-embedding slice (Model2Vec) is frozen into `testdata/retrieval_eval.json`
   with a hand-curated same-topic relevance set; `TestRetrievalRecall` runs
   model-free in CI and asserts recall@10 vs that relevance: Flat pins the
   embedding+exact-scan quality baseline (0.86), and HNSW / FlatI8 must stay within
   tolerance (both also 0.86 here — Alg-4 + int8 track exact Flat perfectly). So an
   index or scoring change that degrades retrieval quality now fails a test, not
   just a benchmark. `TestGenRetrievalFixture` (model-gated) rebuilds the fixture.
3. **`fuse` extensions: weighted presets / score-aware fusion (RSF)** —
   ✅ **DONE.** `fuse.RSF` / `RSFWeighted` add relative-score fusion (per-ranking
   min-max normalize → sum) alongside the rank-based `RRF`/`RRFWeighted`, with
   `Scored` + the `Scores` projection helper. Package doc now frames the choice
   (RRF for incomparable/noisy scales, RSF when magnitude is calibrated). Tests
   pin the normalization/weighting/negatives/edge cases + a runnable Example.

## 5. API & code health

1. **Typed sentinel errors** (`embed.ErrBadMagic`, `encoder.ErrShape`, …)
   for `errors.Is/As` — [medium / low]. Current string-wrapped chains read
   well but can't be programmatically matched. Additive, no breakage.
2. **Scope the global knobs** — [low-medium / medium]. `linalg.SetParallel
   Threshold/Width` are process-wide; two consumers with different workloads
   in one process can't disagree. Per-`Workspace` (or per-call options)
   overrides, globals kept as defaults. Experimental tier, so still cheap to
   do — gets expensive the moment `linalg` graduates.
3. **Decide the worker pool's fate** — [low / low]. Built, measured
   neutral-to-worse end-to-end, unshipped (`task-perf-linalg.md`). Either
   delete `Workspace.SetWorkers`/pool.go or mark it deprecated-experimental
   with the measurement linked. Dead-but-present code is a maintenance tax.
4. **Fold the review's minor nits** — [low / low]: `.DS_Store` in repo root
   (gitignore it), `linalg/linalg.test` binary committed, `scripts/
   __pycache__` tracked.

## 6. Docs, DX & adoption

Per `road-to-1.0-critique.md`: the code is past 1.0; the gap is audience.

1. **`ken`-free quick start** — ✅ **DONE.** README's model-fetch now uses
   `huggingface-cli` directly (both `minishlab/potion-code-16M` and
   `nomic-ai/CodeRankEmbed`), with `ken download-model` noted as an optional
   shortcut — a fresh checkout can populate `testdata/` without aikit tooling.
2. **Comparative README table + published benchmarks** — [high / low-medium,
   gated on §2.3]. aikit vs hugot (pure-Go backend), Ollama-as-a-service,
   coder/hnsw, Bleve: cgo, model coverage, recall, p95, binary size. The
   parity-test discipline is a trust asset — surface it.
3. **One named external adopter before promoting further surfaces** —
   [high / not-engineering]. Same conclusion as the ken critique: ship the
   next minor *with* a consumer (an MCP server shipping an embedded corpus,
   goinfer counts only partially as it's first-party).
4. **An `examples/embedded-corpus` showcase** — [medium / low]. The
   `//go:embed` corpus+model+index single-binary pattern is the
   differentiator no Python stack matches; today it lives in prose only.
   Needs §2.2 (HNSW persistence) for full effect; a Flat-index version can
   ship now.
5. **Public architecture doc** — ✅ done 2026-06-09: [`docs/architecture.md`](../architecture.md)
   (package DAG, ecosystem map, Backend seam, quarantines, invariant index,
   ADR resolution table). Keep it current as surfaces move.
6. **`doc.go` for `linalg`** with kernel/dispatch map — ✅ **DONE.**
   `linalg/doc.go` carries the package doc with the "which kernel fires on which
   CPU, and why" dispatch table + the bit-identical-parallelism invariant.

---

## Competitive context (June 2026)

- **Antfly/Termite** — distributed hybrid-search server (BM25 + dense +
  SPLADE + graph, local inference via gomlx/ONNX fork, MCP/A2A surfaces).
  Elastic License v2, single static binary, TLA+-checked internals;
  published 0.9975 recall / 10–12 ms p95 on 50K VectorDBBench (M4 Max).
  Core since rewritten in Zig (May 2026) — their Go engine used Go 1.26
  `GOEXPERIMENT=simd` via go-highway (see §1.5). Not a library competitor,
  but the feature bar for "what hybrid retrieval ships in 2026": sparse +
  persistence + published recall numbers (§2.1–2.3).
- **hugot** (knights-analytics) — HF-pipelines-in-Go; pure-Go backend
  explicitly for small models/batches, full speed needs ONNX Runtime (cgo).
  aikit wins the no-cgo lane; loses on model breadth (§2.5).
- **coder/hnsw** — pure-Go HNSW with import/export; no inference. Sets the
  bar for §2.2.
- **Bleve / chromem-go / sqlite-vec / LanceDB-go** — index-only options;
  none ship embedded model inference. aikit's full-pipeline scope is the
  moat.
- **Ollama** — the default Go-shop answer; a server dependency, awkward
  rerankers. aikit's pitch against it is in-process, zero-deploy retrieval.
- **Model2Vec upstream (MinishLab)** — potion-retrieval-32M is the current
  best static retrieval model; parity-testing means aikit inherits upstream
  gains cheaply. Track new potion releases; consider a parity pin for
  potion-retrieval-32M specifically.
