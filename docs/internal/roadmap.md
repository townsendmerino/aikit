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
  an arm64 Dot8x4 4-lane-partial-sum bug before it shipped).
- **§1.4 Int8 M-loop register tiling for W8A8 — CLOSED (measured no win).** Built
  + benchmarked the 4-row arm64 SDOT tile (bit-identical); M=64 was a statistical
  tie. W8A8 at M>1 is SDOT-throughput-bound (~3 SDOT/cycle, near M1 peak) and the
  column-outer reblock already took the reuse — nothing for register tiling to
  exploit. Reverted. See §1.4.
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
5. **Evaluate Go 1.26 `GOEXPERIMENT=simd` (go-highway-style portable
   kernels)** — [medium / medium, strategic]. Antfly's engine got
   AVX2/AVX-512/NEON from one portable kernel source. A spike: port `dotF32`
   and `dotI8` and bench against the hand asm. If within ~10%, new kernels
   (e.g. the amd64 W4A8 above) could be written once instead of per-arch —
   directly attacks the bus-factor cost of `.s` files. Keep the asm where it
   wins.
6. **AVX-512/VNNI dispatch tier** — [low-medium / medium]. Currently absent;
   reasonable to keep deprioritized (downclocking caveats, AVX2 ubiquity), but
   revisit if #5 makes it nearly free. AMX: out of scope.
7. **Windows real mmap** (`CreateFileMapping`) instead of heap-read fallback —
   [low / low-medium]. Matters once any sizable consumer runs on Windows.

## 2. Competitive differentiation — features

Positioning note: Antfly/Termite is a *server product* (Elastic License v2,
Raft, Pebble, dashboards) — not a direct competitor to an MIT-licensed,
embeddable, cgo-free *library*. The competitive play is to be the best
building blocks that products like that would otherwise build in-house —
and to cover their headline retrieval features in library form. hugot's full
speed requires ONNX Runtime (cgo); aikit's no-cgo lane stays open.

1. **Learned sparse retrieval (SPLADE-style) package** — [high / high]. The
   biggest functional gap vs Termite's retrieval stack (dense + sparse +
   BM25). Shape: a `sparse` package — doc/query expansion inference (a small
   masked-LM head, same family as `encoder`'s NomicBert machinery) + an
   inverted-index scorer; fuses into the existing `fuse.RRF` flow. Big lift;
   consider starting with *pre-computed* sparse-vector indexing + scoring
   (inference-optional) to ship the index half first.
2. **HNSW persistence: serialize/load, mmap-friendly** — [high / medium].
   Today the graph is rebuilt per process. Serialization + zero-copy load
   unlocks the `//go:embed`-an-index pattern — the same embedded-corpus story
   the ken critique calls the real differentiator. `coder/hnsw` has
   import/export; aikit shouldn't lack it. Design the format versioned from
   day one (Experimental tier).
3. **Published recall/latency benchmark harness** — [high / medium]. Antfly
   leads with "0.9975 recall, 10–12 ms p95 on 50K VectorDBBench". aikit has
   strong internal perf discipline but no public, reproducible
   recall+latency numbers, and no comparative table vs hugot / Ollama /
   coder/hnsw. A `bench/` harness (VectorDBBench subset or BEIR slice +
   p50/p95/p99 + recall@k) turns "parity-tested" into a marketable number.
   Doubles as the regression gate §3.1 needs.
4. **Quantized ANN storage (int8 vectors; optional binary pre-filter)** —
   [medium / medium]. 4× memory reduction on the dense index; `linalg`
   already has the int8 dot kernels. Matters exactly in aikit's niche
   (embedded, single-binary, RAM-constrained). Binary/Hamming pre-filter +
   f32 rescore as a follow-up.
5. **Generalize `encoder` beyond NomicBert (config-driven BERT-family
   loader)** — [medium / high]. Today: CodeRankEmbed only. Termite/hugot run
   arbitrary HF models. Full generality is ONNX-shaped (don't go there);
   instead parameterize the existing forward pass (layers, heads, RoPE vs
   learned positions, pooling) to cover the popular MiniLM / bge /
   mxbai-class rerankers and embedders from safetensors. Each new
   architecture stays parity-pinned like CodeRankEmbed.
6. **General-NLP tokenizer option for `bm25`** — [medium / low]. The
   code-tuned identifier splitting is a hidden assumption for general-text
   consumers (README carry-over invariant). Ship a plain-text analyzer
   alongside; keeps the code-RAG default, widens the audience.
7. **Streaming / incremental index updates** — [medium / medium]. `ann.Flat`
   and `bm25.Build` are build-once. Add/delete on HNSW (tombstones) and an
   incremental BM25 segment story would cover the "long-running process,
   corpus changes" case Termite/Bleve handle natively. Scope carefully —
   immutability-by-design is part of the current safety story.
8. **Matryoshka dimension truncation support in `embed`/`ann`** — [low /
   low]. Cheap to add (truncate + renormalize helpers, dim-aware index);
   pairs well with #4 for memory.

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

1. **HNSW Algorithm-4 neighbor-diversity heuristic** — [medium / medium].
   Current "M nearest" (Alg-3) is honest and documented (`ann/hnsw.go:21-26`)
   but costs recall-per-ef on clustered data — and code-corpus embeddings are
   clustered. Implement behind `Config`, measure with §2.3's harness, flip
   the default if it wins. (Per the ken critique: recall is the one axis
   that materially improves the *product*.)
2. **Recall regression tests on a real slice** — [medium / low-medium]. The
   golden tests pin numerics; nothing pins end-to-end retrieval quality.
   A small fixed corpus + frozen relevance set, recall@10 asserted with
   tolerance. Prereq for safely landing #1, §2.4, §2.7.
3. **`fuse` extensions: weighted-RRF presets / score-aware fusion (RSF)** —
   [low / low]. Antfly exposes weighted strategy mixing; `fuse.RRFWeighted`
   exists, but a documented recipe (and optionally relative-score fusion)
   closes the gap cheaply.

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
