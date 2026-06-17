# aikit — a pure-Go retrieval toolkit

Composable building blocks for code/document retrieval and reranking, in **pure
Go with no cgo** in the core (stdlib + `golang.org/x/text` only). Chunk text,
embed it, search it lexically and semantically, fuse the rankings, and rerank
with a transformer reranking model — each package is small, independently
importable, and parity-tested against a Python reference.

The dependency DAG is shallow: most packages are leaves; `encoder` requires
`embed` + `linalg` (+ `sparse` for the SPLADE expansion head). The one heavier dependency — `gotreesitter` (pure-Go, but a
large embedded-grammar payload) — is quarantined in the separate
`chunk/treesitter` submodule, so importing the core never pulls it in.

> **Generation lives in [`goinfer`](https://github.com/townsendmerino/goinfer).**
> The decoder-only LLM runtime (Gemma 3 / Qwen / Llama …), its SentencePiece/
> byte-level tokenizers, constrained decoding, and the optional WebGPU (cgo)
> backend were split out so aikit stays a small, cgo-free retrieval library.
> goinfer depends inward on aikit (`embed`, `linalg`).

## Packages

| Package | Purpose | Deps (beyond stdlib) |
|---|---|---|
| `topk` | bounded min-heap top-K selector (generic) | — |
| `ann` | cosine ANN over a dense matrix — exact flat scan + approximate HNSW graph | `linalg`, `topk` |
| `bm25` | identifier-aware BM25 lexical index (Lucene-variant); `Tokenize` (code) + `TokenizePlain` (general text) | `topk` |
| `fuse` | rank fusion (RRF) + relative-score fusion (RSF) — blend lexical + dense rankings for hybrid search | — |
| `sparse` | learned-sparse (SPLADE) retrieval — inverted index + sparse-dot scoring over vectors from `encoder.SPLADE` (in-process) or precomputed | `topk` |
| `bench` | reproducible recall + latency harness for the dense indexes (Flat / HNSW / FlatI8) — Experimental tooling | `ann` |
| `linalg` | SIMD `f32` dot/matmul (NEON on arm64, AVX2/FMA on amd64) + int8/int4 quant kernels | — |
| `mmap` *(Experimental)* | read-only file mapping + `madvise` residency hints + a demand-signal-agnostic `SpanCache` (LRU spans under a byte budget) — the substrate `ann`/`embed` mmap loaders sit on; cgo-free, `!unix` heap fallback | `golang.org/x/sys` *(darwin only)* |
| `embed` | Model2Vec inference: WordPiece tokenizer + safetensors loader + L2-norm | `golang.org/x/text` |
| `encoder` | CodeRankEmbed (NomicBert) + MiniLM-class BERT embedder + SPLADE expansion + cross-encoder reranker — transformer inference scored by cosine / sparse dot / relevance logit; pluggable matmul `Backend` | `embed`, `linalg`, `sparse` |
| `vision` *(Experimental)* | SigLIP / ViT image encoder — decode → preprocess → pure-Go transformer forward → image embeddings (f32 or int8 W8A8), parity-pinned to HF `SiglipVisionModel`; stdlib image codecs, no cgo | `embed`, `linalg` |
| `chunk` | language-aware chunker registry + `regex`, `markdown`, `line` chunkers | — |
| `chunk/treesitter` *(submodule)* | tree-sitter-backed syntactic chunker | `gotreesitter`, `…/aikit` |

`chunk/treesitter` is a **separate Go module** (`…/aikit/chunk/treesitter`) so the
`gotreesitter` dependency is opt-in: `go get …/aikit/chunk/treesitter` only when
you want syntactic chunking; the core stays dependency-light.

## Quick start — hybrid RAG retrieval

A runnable end-to-end pipeline (chunk → embed → ANN + BM25 → RRF fuse →
cross-encoder rerank → top-K) lives in [`examples/rag/`](examples/rag). The shape:

```go
// Lexical (BM25) and dense (ANN over embeddings) each rank the chunks…
lex := bm25Index.TopK(bm25.Tokenize(query), 50)
den := annIndex.Query(queryVec, 50)
// …fuse the two rankings (rank-based, no score-scale juggling)…
fused := fuse.RRF(fuse.DefaultK,
    fuse.Keys(lex, func(r bm25.Result) int { return r.Doc }),
    fuse.Keys(den, func(h ann.Hit) int { return h.Index }))
// …then rerank the fused shortlist with the encoder for final order.
```

`encoder`'s matmul routes through a `Backend`; the default is pure-Go SIMD CPU.
A WebGPU backend can be slotted in by importing `goinfer/gpu` under `-tags gpu`
— without aikit ever importing cgo.

For the zero-deploy story, [`examples/embedded-corpus/`](examples/embedded-corpus)
is a single self-contained binary that `//go:embed`s the Model2Vec model, a prebuilt
int8 index, and the corpus, and answers Go/aikit questions over hybrid (dense +
lexical) search with **no external files** and ~50 ms startup — the
`//go:embed`-a-corpus lane no Python or ONNX stack reaches.

---

## Platforms

The core is pure Go (no cgo) and builds + tests on **Linux, macOS, and Windows**
(amd64 and arm64) — CI covers all three. SIMD acceleration in `linalg` uses NEON
on arm64 and AVX2/FMA on amd64 (runtime-detected, scalar fallback otherwise), on
every OS.

The mmap-backed loaders (`embed.OpenSafetensorsMmap`, `OpenGGUFMmap`) use real
memory-mapping on unix and **fall back to a heap read on Windows** — identical
API and results, just without OS-page-cache sharing (so a large checkpoint costs
heap RAM there). The non-mmap loaders (`OpenSafetensors*`) are heap-backed on
every platform.

The only cgo in the ecosystem is the optional WebGPU backend (`goinfer/gpu`,
`webgpu`), which needs a C toolchain. `chunk/treesitter` (`gotreesitter`) is
pure-Go too — it's a separate opt-in module only because of its large embedded
grammars, not cgo. The core pulls in neither.

---

## How aikit compares

Measured against pure-Go ANN libraries on **real Model2Vec embeddings** (N=8000,
dim 256, M=16, EfSearch=64, k=10; recall@10 vs exact cosine). Reproduce with
[`benchmarks/`](benchmarks/) — `cd benchmarks && GOWORK=off go run .` — which also
documents the methodology and why synthetic vectors can't measure recall@k.

| index | recall@10 | p50 latency | index memory |
|---|---|---|---|
| **aikit HNSW** | **0.995** | 0.085 ms | ~2 MB |
| **aikit FlatI8** (int8) | **0.995** | 0.13 ms | ~2 MB |
| aikit Flat (exact) | 1.000 | 0.28 ms | ~0 MB (zero-copy) |
| [coder/hnsw](https://github.com/coder/hnsw) | 0.22 † | 0.058 ms | ~8 MB |
| [chromem-go](https://github.com/philippgille/chromem-go) (exact) | 1.000 | 3.77 ms | ~4 MB |

**FlatI8 is the standout** — 0.995 recall at near-exact latency and ¼ the float32
memory. † coder/hnsw's recall is *structurally* construction-limited on clustered
real embeddings (flat across search-ef 64→800; only ~0.4 even at M=64); it uses
plain greedy neighbor selection, whereas aikit defaults to the **Algorithm-4
diversity heuristic** built for exactly this case. Verified fair (canonical API,
correct distance, full k, finds the right region) — see the
[benchmark notes](benchmarks/README.md#reading-the-table).

### Capability matrix

| | cgo-free | model inference | image embed | exact | ANN graph | int8 | persistence | lexical + hybrid | learned-sparse | static binary |
|---|---|---|---|---|---|---|---|---|---|---|
| **aikit** | ✅ | ✅ Model2Vec + CodeRankEmbed | ✅ SigLIP/ViT | ✅ Flat | ✅ HNSW (Alg-4) | ✅ FlatI8 | ✅ HNSW | ✅ BM25 + RRF/RSF | ✅ sparse | ✅ **1.8 MB** |
| coder/hnsw | ✅ | — | — | — | ✅ | — | ✅ | — | — | ✅ |
| chromem-go | ✅ | via external API | — | ✅ | — | — | ✅ | — | — | ✅ |
| Bleve v2 | dense needs cgo (faiss) | — | — | — | ✅ vector | — | ✅ | ✅ full-text | — | dense: ✗ |
| hugot | ✗ (ONNX Runtime) | ✅ HF pipelines | ✗ (ONNX) | — | — | — | — | — | — | ✗ |

aikit is the only one of these that ships the **whole pipeline** — local model
inference *and* dense + lexical + sparse retrieval *and* fusion — in a single
**1.8 MB pure-Go static binary** (`CGO_ENABLED=0`, the full `ann`+`bm25`+`fuse`+
`embed` surface). It's also the only **cgo-free image embedder** here: the `vision`
SigLIP/ViT tower runs the whole forward in pure Go, so image→image similarity and
image-as-document indexing need no ONNX runtime or sidecar (hugot can embed images
but only via the ONNX Runtime native library). hugot otherwise covers inference but
needs that cgo backend; the vector DBs cover indexing but not inference. The
`//go:embed`-a-corpus, zero-deploy story is the lane no Python or ONNX stack reaches.

### Retrieval quality on a standard benchmark

On the **BeIR/scifact** test set (a canonical BEIR task), aikit — `potion-retrieval-32M`
embeddings + exact Flat cosine — scores **nDCG@10 0.638** (300 queries, 5183 docs).
That's a cross-referenceable number: SciFact + nDCG@10 is the standard MTEB/BEIR
protocol (the model's overall MTEB retrieval score is 35.06), and 0.638 is right where
a strong static retriever lands — near all-MiniLM-L6-v2's own SciFact nDCG@10, at a
fraction of the cost and pure-Go. Reproduce: `scripts/prep_beir.py`, then
`cd benchmarks && GOWORK=off go run ./beir`.

### Inference throughput (vs hugot)

aikit runs the transformer paths — the MiniLM bi-encoder and the cross-encoder — in
pure Go. all-MiniLM-L6-v2 encodes short queries at **~22 texts/sec (≈46 ms/text, single
thread)**; at the full 256-token context the per-token rate climbs to **~710 tokens/sec
(≈360 ms/text)** as the larger matmuls amortize per-call overhead — the regime aikit's
cache-blocked GEMM (`linalg.MatmulBT`) accelerates. All on CPU with no ONNX Runtime, no
GPU, `CGO_ENABLED=0`; concurrent encoding scales ~linearly across cores. (Primary dense retrieval uses Model2Vec static embeddings —
microseconds per text, the table above; the transformer path is the higher-fidelity
reranking/embedding step over a shortlist.) Measure it: `cd benchmarks && GOWORK=off go
run ./inference`.

The contrast with [hugot](https://github.com/knights-analytics/hugot) is a deployment
tradeoff, not a raw-speed one. hugot's fast CPU backend is ONNX Runtime — a native
shared library + cgo — and *is* faster than pure Go; it also ships a pure-Go GoMLX
backend its docs scope to "simpler workloads / smaller models." aikit's bet runs the
other way: no runtime to install, link, or version — one static binary that already
holds the model. Same checkpoint on both sides, so it's apples-to-apples on quality;
the difference is what you deploy.

---

## Stability tiers

These two tiers define what 1.0 promises. The split is **frozen for v1.0**, and
the Hard tier is verified backward-compatible across the 0.4.x and 0.5.x minors
(`apidiff`, zero incompatible changes).

### Hard — the 1.0 compatibility guarantee

From v1.0 these follow semver: no breaking change before a v2.0. This is the API
to build on.

- `topk.Selector[T]`, `topk.New`
- `ann.New`, `ann.Flat.Query`, `ann.Hit`
- `bm25.Build`, `bm25.Index`, `bm25.Result`, `bm25.Tokenize`
- `fuse.RRF`, `fuse.RRFWeighted`, `fuse.Keys`, `fuse.Result`
- `embed.Load`, `embed.LoadFromFS`, `embed.StaticModel`
- `embed.LoadTokenizer`, `embed.Tokenizer`
- `embed.OpenSafetensors*`
- `encoder.Load`, `encoder.LoadFromFS`, `encoder.Model`, `encoder.Encoder` interface
- `chunk.Chunker` interface; `chunk.{Chunk, Register, Get, Names, ChunkFile, Language}`
- Concrete chunker names registered under `regex`, `markdown`, `treesitter`

### Experimental — outside the 1.0 guarantee

Young, tuning-driven surfaces that ship in 1.0 but are **explicitly excluded
from the compatibility promise**: they may change in any release (minor or
patch). Supported and useful — but pin a version, or prefer the Hard-tier
equivalent, if you need stability. Each graduates to the Hard tier once it
settles.

- `linalg` — promoted to public in v0.4.0 (was `internal/linalg`). `Dot`,
  `MatmulBT` and the int8/int4 quant kernels are stable in shape but the surface
  is young and tuning-driven.
- `encoder.Backend` / `encoder.RegisterBackend` / `encoder.NewBackend` — the
  matmul-provider seam; new in v0.4.0.
- `ann.HNSW` / `ann.NewHNSW` / `ann.BuildHNSW` / `ann.Config` — the `Hit`/`Query`
  surface is stable, but graph internals and `Config` defaults may tune. Neighbor
  selection defaults to the diversity heuristic (Algorithm 4) for high recall on
  clustered data; `Config.SimpleNeighbors` opts back to plain M-nearest.
- `ann.HNSW.MarshalBinary` / `ann.Load` — index persistence (the
  `//go:embed`-an-index pattern). The serialized format is versioned from day one
  but stays Experimental until the graph internals settle.
- `ann.FlatI8` / `ann.NewFlatI8` — int8-quantized dense index (¼ the memory,
  scored via the W8A8 kernel). Same `Hit`/`Query` shape as `Flat`; new surface, so
  Experimental.
- `ann.Config.Int8` — int8-quantized HNSW: ¼ the vector memory, built + searched +
  persisted in the integer domain (uses `linalg.DotI8`). Recall is unchanged on
  real embeddings (measured Δ0 vs f32). New surface, settling.
- `linalg.MatmulBTAcc64` — `MatmulBT` with float64 dot accumulation (bit-identical
  to a scalar f64 reference), for f32 reassociation error amplified downstream
  (attention → discrete MoE router). New surface.
- `ann.FlatI8.MarshalBinary` / `ann.LoadFlatI8` / `ann.LoadFlatI8Mmap` — int8-index
  persistence (the `//go:embed`-an-index pattern). `LoadFlatI8Mmap` is zero-copy
  (aliases the int8 codes from a read-only mapping for instant startup + page-cache
  sharing); `FlatI8.Close` releases it. Versioned format, settling alongside
  `FlatI8`.
- `mmap` — new leaf package: `MapReadOnly`/`Unmap` (the read-only mapping `ann` and
  `embed` previously each kept a private copy of), `Advise` (`madvise` residency
  hints — firm cap on Linux, best-effort elsewhere), and `SpanCache` (a
  demand-signal-agnostic LRU of page-aligned spans under a byte budget) for paging a
  mapping larger than RAM. stdlib-only (plus `golang.org/x/sys` on darwin), cgo-free,
  with a `!unix` heap fallback. New surface, settling.
- `Flat`/`HNSW`/`FlatI8` `.QueryFilter(q, k, keep)` — query-time logical-delete /
  live-set filter (the index stays immutable). New surface, settling.
- `bm25.TokenizePlain` — new general-text (Unicode word) analyzer alongside the
  code-tuned `Tokenize` (which stays the default); pick whichever fits the corpus.
- `fuse.RSF` / `fuse.RSFWeighted` / `fuse.Scored` / `fuse.Scores` — new
  relative-score fusion alongside the rank-based `RRF`; new surface, settling.
- `embed.Truncate` — new Matryoshka (MRL) embedding truncate + L2-renormalize
  helper; pairs with `ann.FlatI8` for compounded memory reduction.
- `sparse` — the whole package is new (learned-sparse / SPLADE retrieval). The
  `SparseVec` / `Index` / `Query` shape is settled, but it ships only the index +
  scorer half (an in-process masked-LM expansion head is a planned follow-up that
  may extend the surface), so it stays Experimental until that lands.
- `encoder.LoadQ8` / `encoder.ModelQ8` (int8 quant) — alternate precision path.
- `encoder.LoadBERT` / `encoder.BERT` / `BERT.Encode` — MiniLM-class BERT encoder
  (learned positions + GELU FFN + mean pooling), cgo-free, parity-pinned to
  all-MiniLM-L6-v2 (cosine 1.0). New surface, settling.
- `encoder.LoadSPLADE` / `encoder.SPLADE` / `SPLADE.Expand` — in-process SPLADE
  learned-sparse expansion (BERT + masked-LM head → `sparse.SparseVec`), parity 1.0
  vs the reference. Closes the `sparse` loop end-to-end. New surface.
- `encoder.LoadCrossEncoder` / `encoder.CrossEncoder` / `CrossEncoder.Score` — BERT
  cross-encoder reranker (scores a query/document pair → relevance logit), parity-
  pinned to ms-marco-MiniLM-L-6-v2. The cross-encoder half of reranking. New surface.
- The mmap variant of `embed.OpenSafetensors`.
- The concrete chunker structs (`regex.Chunker`, `markdown.Chunker`,
  `treesitter.Chunker`) and their `New()` — prefer `chunk.Get("regex")`.
- `chunk/treesitter` — its own opt-in module, **tagged in lockstep with the core
  whenever the submodule itself changes** (`chunk/treesitter/v1.0.0` requires
  `aikit v1.0.0`). When a core release doesn't touch the submodule it gets no new
  tag — the existing one keeps working, since the core's `chunk.Chunker` contract
  is Hard-tier stable (e.g. nothing in 1.1.x or 1.2.0 changed it). Its
  `treesitter.Chunker` API is stable, but it stays Experimental because it depends
  on the pre-1.0, single-maintainer
  [`gotreesitter`](https://github.com/odvcencio/gotreesitter) — a break there
  could force a change here.

---

## Carry-over invariants (read these once)

- `bm25`'s tokenizer is **code-tuned** (identifier splitting: camelCase /
  PascalCase / ACRONYM / digit splits, plus the lowercased run). A feature for
  code/RAG consumers; a hidden assumption for general NLP.
- `encoder`'s CodeRankEmbed weights are **code-tuned**. Same caveat.
- `ann` assumes **L2-normalized** input vectors. The normalization contract
  lives at the `embed` boundary, not in `ann`.
- `embed` accumulates in **float64** during inference and indexes through
  `mapping[]` — both correctness-critical (float32 silently fails the ≥1−1e-5
  cosine bar on longer inputs; non-mapping access produces wrong embeddings).
- **Indexes are immutable after build** (`ann`, `bm25`, `sparse`) — a cornerstone
  that gives lock-free concurrent `Query` and snapshot consistency. Changing
  corpora are handled by rebuild-and-swap, base+delta+`fuse`, or logical delete
  (`QueryFilter`), never by mutating an index. See
  [architecture.md](docs/architecture.md#design-rules) design rule 4.

---

## Testing + golden fixtures

Model-dependent tests skip cleanly when their per-machine assets aren't present,
so a fresh `go test ./...` is green with embed/encoder parity tests skipped.
Populate the assets with the Hugging Face CLI (`pip install -U huggingface_hub`)
— no aikit-specific tooling required:

```bash
# Model2Vec (embed parity tests) → testdata/model
huggingface-cli download minishlab/potion-code-16M \
    tokenizer.json config.json model.safetensors --local-dir testdata/model

# CodeRankEmbed (encoder parity tests) → testdata/encoder-model
huggingface-cli download nomic-ai/CodeRankEmbed \
    tokenizer.json config.json model.safetensors --local-dir testdata/encoder-model
```

`embed.Load` handles both Model2Vec on-disk formats: the vocabulary-quantized
`potion-code-16M` (with `mapping`/`weights` tensors) **and** the standard format
with only an `embeddings` tensor (direct token-id indexing, mean pooling). For
**general (non-code) retrieval**, prefer **`minishlab/potion-retrieval-32M`** — the
strongest static retrieval model — over the code-tuned `potion-code-16M`.

(If you also use [`ken`](https://github.com/townsendmerino/ken), `ken
download-model [--rerank] --to <dir>` fetches the same snapshots.)

Regenerate the committed golden fixtures:

```bash
.venv/bin/python scripts/pin_inference.py    # Model2Vec → testdata/golden.json
.venv/bin/python scripts/pin_encoder.py      # CodeRankEmbed → testdata/encoder_golden.json
```

---

## Versioning

`v0.x` is pre-1.0; breaking changes can still land between `0.x` minors when the
design requires it (the CHANGELOG records each). **v0.4.0** split the LLM runtime
out to `goinfer`, promoted `linalg` to public, and added the `encoder.Backend`
seam — the last hard-tier-affecting break.

The **Hard tier has held backward-compatible across 0.4.x and 0.5.x** (verified
with `apidiff` — zero incompatible changes), meeting the two-consecutive-minors
bar, so it is **frozen for v1.0**. From v1.0 the Hard tier follows semver
(breaking changes only at a v2.0); the Experimental tier is excluded from that
promise and may change in any release until it graduates.

### Serialized blob formats

The persisted index blobs (`ann.HNSW` / `ann.FlatI8` `MarshalBinary`) are
magic-tagged and versioned. **Pre-1.0 policy: rebuild per minor** — a blob is not a
stable cross-version interchange format; re-serialize your index after an aikit minor
upgrade. The safety net is loud, not silent: `Load*` rejects any version it doesn't
recognize with `ann.ErrFormat` (never a crash or a misread), so a stale blob fails
visibly and you regenerate. The format version is bumped freely within `0.x` when the
layout improves. If you `//go:embed` blobs in your own releases, pin the aikit minor
or rebuild in your pipeline (a `go generate` step, as
[`examples/embedded-corpus`](examples/embedded-corpus) does). At 1.0 this tightens to
a stronger guarantee (read N−1, or reserved-field forward-compatibility) — the next
format bump reserves header flag bytes as the mechanism for the latter.

## License

MIT. See [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for upstream
attributions (Model2Vec, semble, gotreesitter, golang.org/x/text).
