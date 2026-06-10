# aikit — a pure-Go retrieval toolkit

Composable building blocks for code/document retrieval and reranking, in **pure
Go with no cgo** in the core (stdlib + `golang.org/x/text` only). Chunk text,
embed it, search it lexically and semantically, fuse the rankings, and rerank
with a transformer reranking model — each package is small, independently
importable, and parity-tested against a Python reference.

The dependency DAG is shallow: most packages are leaves; `encoder` requires
`embed` + `linalg`. The one heavier dependency — `gotreesitter` (pure-Go, but a
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
| `ann` | cosine ANN over a dense matrix — exact flat scan + approximate HNSW graph | — |
| `bm25` | identifier-aware BM25 lexical index (Lucene-variant) | — |
| `fuse` | reciprocal-rank fusion (RRF) — blend lexical + dense rankings for hybrid search | — |
| `sparse` | learned-sparse (SPLADE-style) retrieval — inverted index + sparse-dot scoring over pre-computed vectors | — |
| `linalg` | SIMD `f32` dot/matmul (NEON on arm64, AVX2/FMA on amd64) + int8/int4 quant kernels | — |
| `embed` | Model2Vec inference: WordPiece tokenizer + safetensors loader + L2-norm | `golang.org/x/text` |
| `encoder` | CodeRankEmbed transformer reranker (NomicBert, 12-layer) — higher-fidelity embeddings scored by cosine; pluggable matmul `Backend` | `embed`, `linalg` |
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
  surface is stable, but graph internals and `Config` defaults may tune.
- `sparse` — the whole package is new (learned-sparse / SPLADE retrieval). The
  `SparseVec` / `Index` / `Query` shape is settled, but it ships only the index +
  scorer half (an in-process masked-LM expansion head is a planned follow-up that
  may extend the surface), so it stays Experimental until that lands.
- `encoder.LoadQ8` / `encoder.ModelQ8` (int8 quant) — alternate precision path.
- The mmap variant of `embed.OpenSafetensors`.
- The concrete chunker structs (`regex.Chunker`, `markdown.Chunker`,
  `treesitter.Chunker`) and their `New()` — prefer `chunk.Get("regex")`.
- `chunk/treesitter` — its own opt-in module, **versioned in lockstep with the
  core** (`chunk/treesitter/v1.0.0` requires `aikit v1.0.0`). Its
  `treesitter.Chunker` API is stable, but it stays Experimental because it
  depends on the pre-1.0, single-maintainer
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

## License

MIT. See [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for upstream
attributions (Model2Vec, semble, gotreesitter, golang.org/x/text).
