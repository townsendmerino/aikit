# aikit roadmap — post-1.1.0 improvement plan

> Captured 2026-06-09 from an external full-repo review (maintainability, CPU
> perf, competitiveness, docs). Sections are ordered by importance; items
> within each section likewise. Each item carries **[impact / effort]**.
> Competitive context at the bottom informs §2.

## Progress (2026-06-09)

- **§3.1 Fuzz binary parsers — DONE.** Added `FuzzParseGGUF` /
  `FuzzParseSafetensors` / `FuzzParseShardIndex` + seed corpus; found & fixed an
  OOM (`make(map, ~5e10)` from an untrusted `tensorCount`), a negative-length
  slice panic, and a safetensors `8+headerLen` overflow. OOM repro committed as
  a regression seed; CI runs a 20s/target fuzz smoke. **Follow-up:** dequant-path
  fuzzing (`Tensor`/`RowDequantizer`) — its own OOM-bounding work.
- **§1.2 Dot8x4 K-crossover — DONE (as a doc fix, not a code change).** The
  premise was wrong: the encoder tiles K at `kBlockDefault=768` (Dot8x4's peak),
  so it never feeds a large-K strip — `fc2` (K=3072) runs as 4×768. A call-site
  conditional would be dead code. The real gap was the *public* `Dot8x4` godoc;
  it now documents the cliff + "tile K to ≤~768".
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

1. **amd64 fused W4A8 kernel (AVX2 `VPMADDUBSW`; VNNI `VPDPBUSD` where
   available)** — [high / high]. arm64 W4A8 decode is ~23× faster than the Q4
   f32 path; amd64 currently falls back to the scalar reference
   (`linalg/quant_w4a8.go`). This is the single largest perf cliff in the
   ecosystem. Already noted as follow-up #4 in `cpu-acceleration.md`; validate
   on the Linux Ryzen box as with the original AVX2 work.
2. **Wire the K-dependent kernel crossover** — [medium / low]. `Dot8x4`
   regresses below `Dot4x4` at large K on amd64 (40.5 vs 51.4 GB/s at K=3072,
   `cpu-acceleration.md` §1) but `encoder/linalg.go:206` calls `Dot8x4`
   unconditionally. Add the conditional, benchmark the crossover point per
   arch. Cheapest real win on the list.
3. **Per-head attention QKᵀ parallelism — measure end-to-end, then decide** —
   [medium / medium]. ~3.4× in isolation but recurs 144×/forward; the
   `inflightForwards` gate keeps it serial under `EncodeBatch`. Profile
   `Model.Encode` on real weights; ship or close the question. Don't trust the
   microbench either way.
4. **Int8 multi-row register tiling (M-loop) for W8A8** — [medium / medium].
   Noted as possible follow-up in the 0.5.2 CHANGELOG entry. Benefits prefill
   / speculative-verify / encoder (M>1), on top of the column-outer reblock.
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
   GGUF dequant paths) — [high / low]. Native `go test -fuzz` + seed corpus
   from `testdata/`. These parse untrusted files; bounds-checking discipline
   is good, fuzzing verifies it. CI: short fuzz on PRs, longer nightly. The
   single best robustness-per-hour item in this document.
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

1. **`ken`-free quick start** — [high / low]. README's model-fetch path
   routes through `ken download-model`, a tool a new reader doesn't have. A
   tiny `cmd/aikit-fetch` (or documented `curl`s of the HF files) makes the
   first `go test` green without leaving the repo.
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
6. **`doc.go` for `linalg`** with kernel/dispatch map — [low / low]. The
   asm is well-commented file-by-file; a package-level "which kernel fires
   on which CPU, and why" table is the missing on-ramp for a second
   maintainer.

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
