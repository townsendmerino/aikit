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
| `hybrid` *(Experimental)* | thin, opt-in wrapper around "retrieve dense + lexical, then `fuse.RRF`" — composes already-built indices, doesn't build/embed/tokenize/rerank | `ann`, `bm25`, `fuse` |
| `sparse` | learned-sparse (SPLADE) retrieval — inverted index + sparse-dot scoring over vectors from `encoder.SPLADE` (in-process) or precomputed | `topk` |
| `late` *(Experimental)* | ColBERT-style late-interaction (MaxSim) reranking over pre-computed per-token vectors (`encoder.Model.EncodeTokens`) — a shortlist reranker, not a corpus-scale index | `linalg`, `topk` |
| `bench` | reproducible recall + latency harness for the dense indexes (Flat / HNSW / FlatI8) — Experimental tooling | `ann` |
| `linalg` | SIMD `f32` dot/matmul (NEON on arm64, AVX2/FMA on amd64) + int8/int4 quant kernels | — |
| `mmap` | read-only file mapping + `madvise` residency hints + a demand-signal-agnostic `SpanCache` (LRU spans under a byte budget) — the substrate `ann`/`embed` mmap loaders sit on; cgo-free, `!unix` heap fallback | `golang.org/x/sys` *(darwin only)* |
| `embed` | Model2Vec inference: WordPiece tokenizer + safetensors loader + L2-norm | `golang.org/x/text` |
| `encoder` | BERT/RoBERTa/XLM-R/GTE/nomic-bert(+MoE) embedder family (12 certified models, `docs/embedder-coverage.md`, incl. CodeRankEmbed) + SPLADE expansion + cross-encoder reranker — transformer inference scored by cosine / sparse dot / relevance logit; pluggable matmul `Backend` | `embed`, `linalg`, `sparse` |
| `vision` | SigLIP / ViT image encoder — decode → preprocess → pure-Go transformer forward → image embeddings (f32 or int8 W8A8), parity-pinned to HF `SiglipVisionModel`; stdlib image codecs, no cgo | `embed`, `linalg` |
| `chunk` | language-aware chunker registry + `regex`, `markdown`, `line` chunkers | — |
| `gpu` *(Experimental, darwin+linux; separate modules)* | cgo-free native-GPU device substrate — `Device`/`Buffer`/`Queue`/`Pipeline`/`Encoder` over Metal (darwin, runtime MSL compiler) or CUDA (linux, embedded PTX); the GPU analogue of `linalg`'s CPU role. 8 one-backend-per-module leaves plug into 3 seams: `anncuda`/`annmetal` → `FlatI8.EnableGPU`; `enccuda`/`encmetal` → `encoder.RegisterBackend("cuda"/"metal", …)`; `qwencuda`/`qwenmetal` + `visioncuda`/`visionmetal` → `vision.RegisterResident`. Device tests are hand-run (no GPU CI); the default aikit build never imports any of them and stays pure-Go. See [`examples/gpu-ann`](examples/gpu-ann) for the `FlatI8.EnableGPU` seam end to end. | `github.com/ebitengine/purego` *(darwin)*, `github.com/eitamring/gocudrv` *(linux)* |
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

The retrieve-then-fuse half of that (everything above the rerank line) is one
call via [`hybrid.Retriever`](hybrid) if you'd rather not hand-wire it:
`hybrid.New(annIndex, bm25Index).Query(queryVec, bm25.Tokenize(query), 50)`.
Purely a convenience — `examples/rag` above is unaffected and still hand-wires
it, and `hybrid` builds neither index nor reranks; see its package doc.

`encoder`'s matmul routes through a `Backend`; the default is pure-Go SIMD CPU.
A WebGPU backend can be slotted in by importing `goinfer/gpu` under `-tags gpu`
— without aikit ever importing cgo.

For the zero-deploy story, [`examples/embedded-corpus/`](examples/embedded-corpus)
is a single self-contained binary that `//go:embed`s the Model2Vec model, a prebuilt
int8 index, and the corpus, and answers Go/aikit questions over hybrid (dense +
lexical) search with **no external files** and ~50 ms startup — the
`//go:embed`-a-corpus lane no Python or ONNX stack reaches.

For the third retrieval signal, [`examples/splade/`](examples/splade) runs
learned-sparse (SPLADE) retrieval on its own (chunk → SPLADE expand → sparse
index → query → top-K) — the in-process, no-Python pipeline behind the
"learned-sparse" column in the capability matrix below. It composes into a
fused hybrid search the same way `examples/rag`'s dense and lexical signals
do, via `fuse.Keys(sparseHits, func(h sparse.Hit) int { return h.Index })`.

For the "only cgo-free image embedder" claim below,
[`examples/vision/`](examples/vision) indexes a corpus that mixes code chunks
and images: each image's caption joins the fused dense+lexical search as just
another chunk (image-as-document indexing), and landing on an image hit pivots
into "visually similar images" via its own SigLIP embedding index
(image→image similarity) — the two capabilities the vision package actually
provides, deliberately not a cross-modal text→image search (aikit has no
joint text/image embedding space).

[`examples/colbert/`](examples/colbert) swaps `examples/rag`'s final rerank
stage for ColBERT-style late interaction: the same fused dense+lexical
shortlist, reranked by `late.Index`'s MaxSim (every candidate keeps its own
per-token vectors — `encoder.Model.EncodeTokens`, built for exactly this — and
each query token independently finds its best-matching document token) instead
of `encoder.CrossEncoder`'s one joint forward per pair. Run both examples on
the same query to compare.

[`examples/gpu-ann/`](examples/gpu-ann) is the native-GPU path
(`docs/task-native-gpu.md`) made visible: the same `ann.FlatI8` int8 index,
queried once on the CPU and once with `EnableGPU()`, checking the two agree
exactly and timing both. It's its own Go module (like `examples/embedded-corpus`)
since the GPU backends pull in `purego`/`gocudrv`, exactly what the core
module's cgo-free promise keeps out — see "Platforms" below. Needs no model:
`FlatI8` just quantizes whatever vectors it's given, so this runs on synthetic
data. Run on both backends this repo ships:

- **Metal** (M1 Pro — the same chip `gpu/annmetal`'s own kernel comments cite
  for their bench numbers, an integrated GPU): `go run ./examples/gpu-ann` (1M
  vectors, dim 256, 256 queries) came out close to parity with the CPU across
  repeated one-shot runs — roughly 0.97–1.2×, not the larger wins
  `docs/task-native-gpu.md` reports from its own (presumably iterated/averaged)
  benchmark harness — while a small corpus (`-n 50000 -queries 8`) has the GPU
  clearly **losing** (dispatch overhead dominates).
- **CUDA** (a discrete RTX 2070 SUPER, verified over SSH): the same run
  measured **~69× over CPU** at 1M vectors, and still **~20× at 50k vectors /
  8 queries** — the small-scale regime where the integrated Metal GPU above
  loses outright. A discrete GPU's dispatch overhead is proportionally far
  smaller against its own much higher raw throughput, so "GPU loses below some
  N" is a property of the *hardware class*, not a fixed threshold in the code.

Both backends are correct, not just directionally plausible: every run's GPU
top-k was bit-identical to the CPU's, on both machines, at every scale tried —
this example checks that itself, and `gpu/anncuda`'s own test suite (run on
the same box) adds a much more exhaustive parity gate on top. The real lesson
here is the tradeoff `gpu/annmetal`'s large-N gating exists for, and that it's
hardware-dependent, not a cherry-picked best number from either box.

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
fraction of the cost and pure-Go. Reproduce: `scripts/fixtures/prep_beir.py`, then
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

Two tiers define aikit's compatibility promise: the Hard tier follows semver
(no breaking change before a v2.0), verified by `apidiff` on every tagged
release (`tools/releasegate`, enforced since v1.0 — see `RELEASING.md`); the
Experimental tier is explicitly excluded from that promise and may change in
any release until it graduates. aikit is at v1.21.0; the Hard tier has held
backward-compatible across every release since 1.0.

### Hard — the semver-covered surface

No breaking change before a v2.0. This is the API to build on.

- `topk.Selector[T]`, `topk.New`, `topk.Selector.Threshold`
- `ann.New`, `ann.Flat.Query`, `ann.Hit`
- `bm25.Build`, `bm25.Index`, `bm25.Result`, `bm25.Tokenize`
- `fuse.RRF`, `fuse.RRFWeighted`, `fuse.Keys`, `fuse.Result`
- `embed.Load`, `embed.LoadFromFS`, `embed.StaticModel`, `embed.StaticModel.EncodeBatch`
- `embed.LoadTokenizer`, `embed.Tokenizer`
- `embed.OpenSafetensors*`
- `encoder.Load`, `encoder.LoadFromFS`, `encoder.Model`, `encoder.Encoder` interface
- `chunk.Chunker` interface; `chunk.{Chunk, Register, Get, Names, ChunkFile, Language}`
- Concrete chunker names registered under `regex`, `markdown`, `treesitter`
- `linalg.Dot`, `linalg.MatmulBT`, and the int8/int4 quant kernels — public
  since v0.4.0, and the busiest Experimental surface by a wide margin: 36
  non-test files in goinfer route through it. **Graduated 2026-08-18** (this
  tier re-curation) on that organically-found production evidence, per
  `docs/internal/roadmap.md` §2.7's trigger.
- `encoder.Backend`, `encoder.RegisterBackend`, `encoder.NewBackend` — the
  matmul-provider seam, public since v0.4.0. goinfer's WebGPU backend
  (`gpu/backend.go`) and serving path (`internal/serveapp/{embeddings,
  openai,main}.go`) register and dispatch through it in production.
  **Graduated 2026-08-18.**
- `vision` — the whole package (SigLIP/ViT image encoder). In aikit since
  v1.7.0 (14 minors survived); goinfer's serving path (`vision_serve.go`),
  GPU backend registration (`gpu/vision_*.go`), and demo agent all depend on
  it directly. **Graduated 2026-08-18.**
- `mmap` — the whole package. In aikit since v1.9.0 (12 minors survived);
  goinfer imports it in 4 non-test files. **Graduated 2026-08-18.**

### Experimental — outside the semver promise

Young, tuning-driven surfaces that are **explicitly excluded from the
compatibility promise**: they may change in any release (minor or patch).
Supported and useful — but pin a version, or prefer the Hard-tier equivalent,
if you need stability. Each graduates to the Hard tier once an external
consumer uses it in production, organically found (not chased), for two-plus
consecutive quiet minors — `docs/internal/roadmap.md` §2.7's trigger, not a
fixed schedule. Age alone doesn't graduate a surface: several entries below
have shipped since v0.4.0/v1.x with no reported break, but stay here because
no such consumer has surfaced yet.

- `ann.HNSW` / `ann.NewHNSW` / `ann.BuildHNSW` / `ann.Config` — the `Hit`/`Query`
  surface is stable, but graph internals and `Config` defaults may tune. Neighbor
  selection defaults to the diversity heuristic (Algorithm 4) for high recall on
  clustered data; `Config.SimpleNeighbors` opts back to plain M-nearest.
- `ann.HNSW.MarshalBinary` / `ann.Load` — index persistence (the
  `//go:embed`-an-index pattern). The serialized format is versioned from day one
  but stays Experimental until the graph internals settle.
- `ann.LoadHNSWMmap` — zero-copy mmap loader for HNSW (aliases the float32
  vector block directly from a read-only mapping), added alongside the v4
  format bump. Covered by `ann.Load`'s existing Experimental status above.
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
- `Flat`/`HNSW`/`FlatI8` `.QueryFilter(q, k, keep)` — query-time logical-delete /
  live-set filter (the index stays immutable). New surface, settling.
- `bm25.TokenizePlain` — new general-text (Unicode word) analyzer alongside the
  code-tuned `Tokenize` (which stays the default); pick whichever fits the corpus.
- `bm25.Index.TopKBatch` — batch query API mirroring `ann.FlatI8.QueryBatch`'s
  work-stealing dispatch, for bulk workloads. New surface.
- `bm25.MarshalTokens` / `bm25.UnmarshalTokens` — a cache for the *tokenized
  corpus* (`Build`'s input), not a serialized `Index`; `bm25.Index` itself
  gains no format or compatibility promise from this. Deliberately the
  smaller half of the deferred N4 persistence question
  (`docs/internal/roadmap.md` §2.14) — caches roughly half of BM25's measured
  cold-start cost (tokenization), while `Build` itself still runs on load at
  its own already-cheap cost. New surface, may be reworked or dropped without
  touching `Index`.
- `fuse.RSF` / `fuse.RSFWeighted` / `fuse.Scored` / `fuse.Scores` — new
  relative-score fusion alongside the rank-based `RRF`; new surface, settling.
- `embed.Truncate` — new Matryoshka (MRL) embedding truncate + L2-renormalize
  helper; pairs with `ann.FlatI8` for compounded memory reduction.
- `sparse` — the whole package is new (learned-sparse / SPLADE retrieval). The
  `SparseVec` / `Index` / `Query` shape is settled; `encoder.SPLADE` (below) now
  provides the in-process masked-LM expansion head, closing the index+scorer
  package end-to-end. Stays Experimental — new surface, settling.
  `sparse.Index.QueryBatch` (batch query API, mirroring `bm25.Index.TopKBatch`
  above) is covered by the same status.
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
- `ann.FlatBinary` / `ann.NewFlatBinary` / `ann.NewFlatBinaryOverquery` /
  `ann.DefaultOverquery` — binary (SimHash) prefilter + exact **float32** rerank,
  13–26× end-to-end over `FlatI8`. Recall is ≈1.0 on real embeddings but it is an
  **approximate** first stage, and `DefaultOverquery = 16` is a tuning constant
  chosen from a measured recall curve — both may move. Same `Hit`/`Query` shape as
  `Flat`; new surface, so Experimental, like `FlatI8` before it.
- `ann.FlatBinaryI8` / `ann.NewFlatBinaryI8` / `ann.NewFlatBinaryI8Overquery` —
  the same binary prefilter composed with `FlatI8`'s int8 rerank instead of
  `FlatBinary`'s float32 one: dim/8 + dim bytes per vector rather than dim/8 +
  4·dim, compounding the memory win on top of the same prefilter throughput gain.
  Recall is 1.0000 on the real Model2Vec corpus at `DefaultOverquery` (int8
  quantization costs essentially nothing beyond the prefilter's own
  approximation on real embeddings — same finding `FlatI8` itself made). Same
  `Hit`/`Query` shape as `FlatBinary`; shares its prefilter code exactly
  (`binaryPrefilter`), so the two never drift independently. New surface,
  Experimental.
- `ann.HNSW.WriteTo` / `ann.FlatI8.WriteTo` — streaming serialization
  (`io.WriterTo`), avoiding `MarshalBinary`'s full second copy of the index. They
  emit byte-identical output to `MarshalBinary` and inherit its tier: the format is
  versioned but stays Experimental until the graph internals settle.
- `embed.LoadMmap` — memory-mapped Model2Vec load. Peak heap falls 5.8× and
  time-to-first-result rises 17%, so it is a deliberate footprint/latency trade
  rather than a better `Load`. Experimental while that default is still being
  argued; the returned `*StaticModel` is the Hard-tier type.
- `embed.SafetensorsFile.ReleaseTensors` — advisory release of a consumed tensor's
  resident pages. **Explicitly Experimental despite living on a Hard-tier type**,
  because its observable effect is platform-conditional: it is a no-op on
  heap-backed files and on every OS but Linux (macOS does not honour
  `MADV_DONTNEED` for a read-only file-backed mapping). Freezing a method whose
  behaviour is "nothing" on some platforms is not a promise worth making yet.
- `embed.Tensor.SubF32` — zero-copy element sub-range of a tensor, for widening or
  quantizing a fused stack one slice at a time. Aliasing and lifetime rules match
  `Float32s`; new surface.
- `encoder.CrossEncoder.ScoreBatch` — the batch form of `CrossEncoder.Score`
  (7.56× over a `Score` loop at 50 documents, bit-identical scores). Covered by
  `CrossEncoder`'s existing Experimental status; listed here so the batch API is
  not read as a separate promise.
- `linalg`'s v1.15.0 additions — the elementwise math kernels (`ExpF32`, `TanhF32`,
  `ErfF32`, `GELUF32`, `GELUTanhF32`, `SiLUF32` and their `*Into` forms,
  `SoftmaxRowInto`), the fused Q8 matmul (`MatmulBTQ8Fused{,Into}`,
  `HasFusedQ8Kernel`, `FusedQ8Applies`), W8A8 activation quantization
  (`QuantizeActivations{,Into}`, `MatmulBTW8A8Pre`, `DequantizeRowsInt8Into`) and
  the Hamming/SimHash primitives (`PackSignBits{,Row}`, `PackedWords`,
  `HammingRows`) — new surface, distinct from `Dot`/`MatmulBT`/the int8/int4
  quant kernels (Hard tier, above): `linalg` is now a mixed-tier package, not
  wholly Experimental.
- `linalg`'s fused attention (v1.35.0) — `AttendTileFused`, `FusedAttnScratch`,
  `NewFusedAttnScratch`, `FusedAttnKeyBlock`, `GatherVBlockMajor`. A
  FlashAttention-style single-tile schedule that keeps the score block resident
  rather than materializing `kt × nKeys`. **Deliberately NOT bit-identical to a
  materialized QK→softmax→AV** — the running-max rescale re-associates both the
  denominator and the AV fold — so it is Experimental in the strong sense: a
  caller needing bit-identity must not use it. Moved from goinfer (P19) body
  verbatim, gated raw-bit against that frozen body.
- `late` — the whole package (ColBERT-style MaxSim reranking). New package.
- `hybrid` — the whole package (thin dense+lexical+`fuse.RRF` wrapper). New
  package.
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
.venv/bin/python scripts/oracle/pin_inference.py    # Model2Vec → testdata/golden.json
.venv/bin/python scripts/oracle/pin_encoder.py      # CodeRankEmbed → testdata/encoder_golden.json
```

---

## Versioning

aikit is past 1.0 — current release: **v1.21.0** (see the
[CHANGELOG](CHANGELOG.md)). **v0.4.0** split the LLM runtime out to `goinfer`,
promoted `linalg` to public, and added the `encoder.Backend` seam — the last
Hard-tier-affecting break before 1.0. The Hard tier has held
backward-compatible ever since, verified by `apidiff` on every tagged release
(`tools/releasegate`; see `RELEASING.md`) — not just the 0.4.x/0.5.x window
that originally justified freezing it — and follows semver: no breaking
change before a v2.0. The Experimental tier (above) is excluded from that
promise and may change in any release until it graduates.

### Serialized blob formats

The persisted index blobs (`ann.HNSW` / `ann.FlatI8` `MarshalBinary`) are
magic-tagged and versioned. **Policy: rebuild per minor** — a blob is not a
stable cross-version interchange format; re-serialize your index after an
aikit minor upgrade. The safety net is loud, not silent: `Load*` rejects any
version it doesn't recognize with `ann.ErrFormat` (never a crash or a
misread), so a stale blob fails visibly and you regenerate. The format
version is bumped freely when the layout improves. If you `//go:embed` blobs
in your own releases, pin the aikit minor or rebuild in your pipeline (a
`go generate` step, as [`examples/embedded-corpus`](examples/embedded-corpus)
does).

**This was originally planned to tighten "at 1.0"** (read N−1, or
reserved-field forward-compatibility). That milestone has passed — aikit is
well past 1.0 now — without the tightening actually happening: rebuild-per-
minor is still the real policy. HNSW's v4 bump reserves a header flags word
as the mechanism for it (`ann/hnsw_persist.go`'s format note), and FlatI8's
own format-bump checklist has the equivalent item pending — the plumbing
exists, the guarantee doesn't yet. Read "at 1.0" here as retired language
describing an unmet trigger, not a promise that was honored on schedule.

That same v4 bump also padded HNSW's header to an 8-byte boundary, which is what lets
[`ann.LoadHNSWMmap`](ann/hnsw_mmap.go) alias the float32 vector block directly from a
read-only mapping instead of copying it — the zero-copy loader `ann.LoadFlatI8Mmap`
already had, now mirrored on the higher-recall index (`ann.FlatI8`'s int8 codes never
needed the alignment fix; HNSW's f32 vectors did).

## License

MIT. See [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for upstream
attributions (Model2Vec, semble, gotreesitter, golang.org/x/text).
