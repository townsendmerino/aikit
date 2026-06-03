# Changelog

All notable changes to `aikit` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with the
pre-1.0 caveat documented in [README.md](README.md#versioning): the "Hard,
1.0-committed" surfaces are expected to be stable through the path to 1.0, but
breaking changes may still occur between `0.x` minors if the design requires
it.

## [Unreleased]

### Changed

- **Streaming int8 quantization at load** — `decoder.Load(…, Quant: "int8")` now
  quantizes each matmul tensor to per-row int8 the moment it is read and frees
  its f32 before the next tensor loads, instead of materializing the whole model
  in f32 and quantizing afterward. The transient footprint drops from the whole
  model in f32 to the int8 model + one tensor's f32 — so a big quantized
  checkpoint loads in roughly a quarter of the RAM. Covers the safetensors,
  GPT-2, and GGUF paths; a quantized `.gguf` lands resident as int8 (the demo
  chats from a bare `.gguf` under `--quant int8`). Forward output is unchanged
  (it quantizes the same weights, just earlier); validated by the new
  `TestGGUF_int8_resident` (argmax + 0.9998 cosine vs the f32 oracle) and the
  unchanged `TestQuantInt8_accuracy`. Public `LoadWeights`/`LoadWeightsFromFS`
  signatures are unchanged. int4 group-quant (≈⅛ f32) is the next step on this
  seam.

### Added

- **`tokenizer.LoadGGUF`** — build a `Tokenizer` from a bare `.gguf` file's
  embedded metadata (vocab + merges + special-token ids), no `tokenizer.json`
  needed. Covers the SentencePiece byte-fallback family
  (`tokenizer.ggml.model == "llama"`: Llama-2/Mistral/TinyLlama), reusing the
  `modeGemma` merge-rank core plus a `▁` dummy-prefix knob (prepend on encode,
  strip one leading space on decode). Parity-gated against HF `tokenizers` on
  TinyLlama (`testdata/tinyllama_tokenizer_golden.json`, pinned by
  `scripts/pin_tinyllama_tokenizer.py`). A bare `.gguf` now chats end-to-end —
  `demo/gemma` detects a `.gguf` path and tokenizes from it.
- `tokenizer.Load` now honors a SentencePiece `Prepend "▁"` normalizer (and the
  paired leading-space strip on decode), so non-Gemma SPM `tokenizer.json`
  files tokenize correctly; Gemma (no Prepend) is unchanged.

### Notes

- Byte-level GGUF tokenizers (`gpt2` family: Llama-3/Qwen/GPT-2) and more GGUF
  K-quant types (Q5_K/Q3_K/IQ*) are deferred until there's a fixture to
  parity-gate them — see [docs/milestones/G7-gguf.md](docs/milestones/G7-gguf.md).

## [0.2.0] — 2026-06-03

Generative half of the toolkit lands. Two new public packages — `decoder` and
`tokenizer` — turn aikit from "embed + retrieve" into "embed + retrieve +
generate", in pure Go with no cgo, validated to HuggingFace parity across a
broad slice of the open-weights ecosystem.

### Added

- **`decoder` package** — autoregressive decoder-only LLM inference as a single
  generic forward pass parameterized by an `Architecture` descriptor resolved
  from the checkpoint. Validated to logit/argmax parity against HuggingFace for:
  - **Families:** Gemma 3, Qwen3, Qwen2.5, Llama-2/3, Mistral, GPT-2, and
    Mixtral (sparse-MoE).
  - **Axes:** RMSNorm/LayerNorm · RoPE (incl. llama3 frequency scaling)/learned
    positions · gated/non-gated/sparse-MoE MLP · full/sliding-window attention ·
    tied/untied heads · optional QKV/output bias · Linear/Conv1D layouts.
  - Public surface: `Load`, `LoadWeights`/`LoadWeightsFromFS`, `Model.Generate`
    (streaming), `Sampler` (temperature/top-k/top-p), `KVCache`, the `Backend`
    seam (`NewBackend`), and the `Config`/`Architecture` descriptors.
- **`tokenizer` package** — the BPE tokenizers the decoder LLMs ship, loaded
  from `tokenizer.json` with HF-exact id parity as the gate:
  - Gemma byte-fallback SentencePiece-style BPE (`▁` space normalize,
    `<0xNN>` fallback).
  - GPT-2 / Llama-3 / Qwen byte-level BPE (NFC normalize, GPT-2 split-regex
    pretokenizer, byte→printable-rune map).
  - Family auto-detected from `tokenizer.json`; special tokens resolved from
    `tokenizer_config.json`. Public surface: `Load`, `Tokenizer`,
    `SpecialTokens`, `ChatStyle`.
- **GGUF support** — self-describing quantized checkpoints (`embed/gguf.go`,
  `decoder/gguf.go`): GGUF v2/v3 container parse + block dequant for F32, F16,
  Q8_0, Q4_0, Q4_K, Q6_K. A bare `.gguf` loads with no sidecar config or
  safetensors. The llama.cpp interleaved q/k RoPE layout is un-permuted back to
  HF `rotate_half`. Validated vs the f32 oracle on TinyLlama: Q8_0 cosine
  0.99996, Q4_0 0.9944, **Q4_K_M 0.9975** (the most-downloaded laptop quant).
- **int8 weight quantization** for the decoder (`--quant int8`).
- **WebGPU backend** for the decoder — resident weights behind the `Backend`
  seam, swappable without touching the forward pass.
- **`internal/linalg`** — shared SIMD matmul/dot kernels (AVX2/FMA on amd64,
  NEON on arm64) and int8 quant helpers, factored out of `encoder` so both
  `encoder` and `decoder` share one accelerated path.
- **`encoder` acceleration** — SIMD/parallel/GPU matmul, plus `ann` HNSW
  approximate index and `fuse` RRF fusion shipped alongside.
- **`demo/gemma` and `demo/gemma-web`** — CLI and stdlib `net/http` + SSE web
  chat front-ends for the decoder.
- **`chunk/treesitter`** — Dart added to the tree-sitter language mapping.

### Changed

- `encoder`'s SIMD dot/matmul kernels moved to `internal/linalg`
  (`dot_arm64.s`, `dot_test.go`); no public-API change for `encoder` consumers.
- Bumped `github.com/odvcencio/gotreesitter` to `v0.20.0-rc3`.
- Applied Go 1.26 modernizers (`go fix ./...`).

## [0.1.1] — 2026-06-02

### Added

- `bm25.Index.IDF(term)` and `bm25.Index.DF(term)` — public read-only accessors
  mirroring the internal `idf` used by query scoring (IDF for ranking, raw DF
  for frequency filtering). Pure additive; no behavior change.

## [0.1.0] — 2026-05-30

### Added

- Initial release, extracted from [`ken`](https://github.com/townsendmerino/ken)
  per ken's ADR-034. Eight packages: `topk`, `ann`, `bm25`, `embed`, `encoder`,
  `chunk` (+ `regex`/`markdown`/`treesitter`).
- Numerical contracts: `embed` golden cosine 1.000000 vs Model2Vec; `encoder`
  golden cosine 1.000000 vs PyTorch+MPS CodeRankEmbed. See
  [README.md](README.md) for stability tiers.

[Unreleased]: https://github.com/townsendmerino/aikit/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/townsendmerino/aikit/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/townsendmerino/aikit/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/townsendmerino/aikit/releases/tag/v0.1.0
