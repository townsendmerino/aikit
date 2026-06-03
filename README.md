# aikit — portable AI building blocks for Go

Pure-Go, no-cgo packages extracted from [`ken`](https://github.com/townsendmerino/ken)'s
code-search pipeline. Each package is independently importable; the dependency
DAG is shallow (most are leaves; `encoder` requires `embed`, `tokenizer`
requires `embed` for the GGUF loader, and `decoder` requires `embed` +
`tokenizer`).

| Package | Purpose | Deps (beyond stdlib) |
|---|---|---|
| `topk` | bounded min-heap top-K selector (generic) | — |
| `ann` | cosine ANN over a dense matrix — exact flat scan + approximate HNSW graph | — |
| `bm25` | identifier-aware BM25 lexical index (Lucene-variant) | — |
| `fuse` | reciprocal-rank fusion (RRF) — blend lexical + dense rankings for hybrid search | — |
| `embed` | Model2Vec inference: WordPiece tokenizer + safetensors loader + L2-norm | `golang.org/x/text` |
| `encoder` | CodeRankEmbed transformer encoder (NomicBert, 12-layer; NEON-accelerated on arm64, AVX2/FMA on amd64, with intra-op row-parallel matmul for single-forward latency) | — |
| `decoder` | generic decoder-only LLM forward pass — Gemma 3, Qwen2.5/3, Llama-2/3, Mistral, GPT-2, Mixtral (MoE); f32/bf16/f16 + int8/int4, safetensors/sharded/GGUF; HF logit parity | `embed`, `tokenizer`, `github.com/cogentcore/webgpu` (WebGPU backend) |
| `tokenizer` | BPE tokenizers the decoder LLMs ship — Gemma byte-fallback SentencePiece + GPT-2/Llama-3/Qwen byte-level, from `tokenizer.json` or a bare `.gguf`'s embedded metadata; HF-exact id parity | `golang.org/x/text`, `embed` (GGUF) |
| `chunk` | language-aware code chunker registry + 3 concrete chunkers (`regex`, `markdown`, `treesitter`) | `github.com/odvcencio/gotreesitter` (treesitter only) |

`aikit` is "the parts of ken another project could reuse." The application
glue — atomic-swap snapshot management, MCP server, git-aware walk, database
introspection — stays in [`ken`](https://github.com/townsendmerino/ken/tree/main).

---

## Stability tiers

Mirroring [ken ADR-032](https://github.com/townsendmerino/ken/blob/main/docs/DECISIONS.md):

### Hard, 1.0-committed

- `topk.Selector[T]`, `topk.New`
- `ann.New`, `ann.Flat.Query`, `ann.Hit`
- `bm25.Build`, `bm25.Index`, `bm25.Result`, `bm25.Tokenize`
- `embed.Load`, `embed.LoadFromFS`, `embed.StaticModel`
- `embed.LoadTokenizer`, `embed.Tokenizer`
- `embed.OpenSafetensors*`
- `encoder.Load`, `encoder.LoadFromFS`, `encoder.Model`
- `encoder.Encoder` interface
- `chunk.Chunker` interface
- `chunk.{Chunk, Register, Get, Names, ChunkFile, Language}`
- Concrete chunker names registered under `regex`, `markdown`, `treesitter`

### Best-effort (may shift between minor versions)

- The concrete chunker structs (`regex.Chunker`, `markdown.Chunker`, `treesitter.Chunker`)
  and their `New()` constructors — go through `chunk.Get("regex")` for the
  stable interface
- `chunk/treesitter` — depends on the pre-1.0, single-maintainer
  [`github.com/odvcencio/gotreesitter`](https://github.com/odvcencio/gotreesitter)
- `encoder.LoadQ8` / `encoder.ModelQ8` (int8 quant) — alternate precision path
- The mmap variant of `embed.OpenSafetensors`
- `ann.HNSW` / `ann.NewHNSW` / `ann.BuildHNSW` / `ann.Config` — new approximate index;
  the `Hit` / `Query` surface is stable, but graph internals and Config defaults may tune
- `fuse.RRF` / `fuse.RRFWeighted` / `fuse.Keys` — new; the RRF math is fixed but the
  helper surface is young
- `decoder` (`Load`, `Model.Generate`, `Sampler`, `KVCache`, `Backend`, the
  `Architecture` descriptor) — new in v0.2.0; numerics are parity-gated against
  HuggingFace, but the loader/descriptor surface will keep moving as more
  families and quant formats land
- `tokenizer` (`Load`, `Tokenizer`, `SpecialTokens`, `ChatStyle`) — new in
  v0.2.0; HF-exact id parity is the contract, the API is young
- GPU backend (`encoder/gpu`, `-tags gpu`) — optional, cgo, foundation stage only

---

## Carry-over invariants (read these once)

- `bm25`'s tokenizer is **code-tuned** (identifier splitting: camelCase /
  PascalCase / ACRONYM / digit splits, plus the lowercased run). A feature
  for code/RAG consumers; a hidden assumption for general NLP.
- `encoder`'s CodeRankEmbed weights are **code-tuned**. Same caveat.
- `ann` assumes **L2-normalized** input vectors. The normalization contract
  lives at the `embed` boundary, not in `ann`.
- `embed` accumulates in **float64** during inference and indexes through
  `mapping[]` — both are correctness-critical (float32 silently fails the
  ≥1−1e-5 cosine bar on longer inputs; non-mapping access produces wrong
  embeddings). Documented in each package's doc.comment.

---

## Testing + golden fixtures

The model-dependent tests skip cleanly when their per-machine assets aren't
present. Populate via:

```bash
# Model2Vec (for embed parity tests)
ken download-model --to testdata/model

# CodeRankEmbed (for encoder parity tests)
ken download-model --rerank --to testdata/encoder-model
```

Or symlink to your existing `~/.cache/huggingface/` cache. A green
`go test ./...` with embed/encoder parity tests skipped is the expected
fresh-checkout state.

To regenerate the committed golden fixtures (`testdata/golden.json`,
`testdata/encoder_golden.json`):

```bash
.venv/bin/python scripts/pin_inference.py    # Model2Vec
.venv/bin/python scripts/pin_encoder.py      # CodeRankEmbed
```

---

## Versioning

`v0.x` is pre-1.0 — the surfaces above tagged "Hard, 1.0-committed" are
expected to be stable through the path to 1.0, but breaking changes may still
happen between `0.x` minor versions if the design requires it. The CHANGELOG
records each one. `v1.0.0` cuts when the hard tier has stabilized for two
consecutive minors.

## License

MIT, copied from [`ken`](https://github.com/townsendmerino/ken/blob/main/LICENSE).
See [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for upstream
attributions (Model2Vec, semble, gotreesitter, golang.org/x/text).
