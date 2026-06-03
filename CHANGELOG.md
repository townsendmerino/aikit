# Changelog

All notable changes to `aikit` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with the
pre-1.0 caveat documented in [README.md](README.md#versioning): the "Hard,
1.0-committed" surfaces are expected to be stable through the path to 1.0, but
breaking changes may still occur between `0.x` minors if the design requires
it.

## [Unreleased]

### Added

- **Mellum2 support + YaRN RoPE scaling.** The decoder now runs JetBrains
  Mellum2 (`model_type: "mellum"`, a 12B-A2.5B MoE code model): a `mellum`
  adapter combines axes we already had — a sparse MoE on every layer (64 experts,
  top-8, with the narrower `moe_intermediate_size` for the expert FFN) and a 3:1
  sliding/full attention interleave (`layer_types`) — plus the one new piece,
  **YaRN** RoPE. YaRN is implemented HF-exact (`_compute_yarn_parameters`: the
  NTK-by-parts inv-freq blend + the `attention_factor` mscale), validated against
  a pinned reference (`TestYarn_matchesHF`, rel ≤ 1e-12), and slots into the
  existing dual-table RoPE via a new per-attention-type scaling path
  (`ropeScalingLocal`) and the new nested `rope_parameters` config (YaRN on full
  layers, plain RoPE on sliding layers). Also generally usable for long-context
  Qwen/Llama checkpoints that set `rope_scaling: {"rope_type": "yarn"}`. The
  `mellum` adapter resolves the real config to the right descriptor + RoPE tables
  + mscale (`TestResolveMellum`); full forward parity awaits the 12B checkpoint.
- **`constrain` package — constrained / structured decoding.** A logit mask that
  forces a model's output to satisfy a grammar: at each step every vocab token
  whose bytes would break the grammar is set to −∞, and EOS is masked until the
  output is a complete document. Ships a streaming **JSON** grammar (byte-level
  pushdown automaton, RFC 8259) — so a small model *physically cannot* emit
  malformed JSON. It plugs into the new `decoder.SamplingParams.LogitProcessor`
  hook (`constrain.Masker.Process` matches the signature) and is stdlib-only (the
  vocab→bytes map is injected as a func, e.g. `tokenizer.TokenText`). The guarantee
  is proven structurally: a hard-invariant test drives the masker with *random*
  logits over a synthetic vocab and confirms the output is always valid per
  `encoding/json` (`TestConstrainedDecode_alwaysValidJSON`). `demo/gemma --json`
  shows it end-to-end (a 1B model emits a valid JSON object). `StopWhenComplete`
  ends generation at the first complete document.
- **`decoder.SamplingParams.LogitProcessor`** — an optional per-step hook,
  `func(generated []int, logits []float32)`, called after the forward pass and
  before sampling so a caller can mask/bias logits (the seam for constrained
  decoding; can also gate EOS).
- **`tokenizer.Tokenizer.TokenText(id) []byte`** — the raw surface bytes a single
  token contributes (no whole-sequence post-processing), for mapping a vocabulary
  onto a byte-level grammar.
- **int8×int8 (W8A8) quantization** (`decoder.Load(…, Quant: "int8int8")`) — in
  addition to the weight-only int8, this quantizes the activations to int8 on the
  fly (dynamic per-row scale) and runs a true integer matmul: `linalg.dotI8`
  accumulates int8×int8→int32, with an AVX2 asm kernel (`dotI8AVX2`: VPMOVSXBW
  sign-extend → VPMADDWD → VPADDD, bit-exact to the scalar reference) and a scalar
  fallback off amd64. **~3.4×** faster than the f32-widen weight-only int8 on a
  decode-step shape (428 → 125 µs, K=N=2048). It is lossier (activations are also
  quantized): gemma cosine 0.9979 vs 0.9996, argmax preserved
  (`TestQuantInt8I8_accuracy`) — so it is opt-in; plain `int8` stays weight-only
  (f32 activations) for the higher accuracy.
- **Byte-level GGUF tokenizer** — `tokenizer.LoadGGUF` now also handles the
  byte-level family (`tokenizer.ggml.model == "gpt2"`: Llama-3 / Qwen / GPT-2),
  not just SPM/llama. It dispatches "gpt2" to the existing `modeByteLevel`
  pipeline and reads the pretokenizer knobs (digit-run cap, NFC, ignore_merges)
  from `tokenizer.ggml.pre` — the GGUF analogue of reading them from
  tokenizer.json. So a bare byte-level `.gguf` (the common modern instruct quant)
  now chats end-to-end. Parity-gated against a real Llama-3.2-1B-Instruct GGUF:
  `LoadGGUF` matches `Load` on the same model's tokenizer.json id-for-id
  (`TestLoadGGUF_byteLevelMatchesJSON`), and that json path is HF-golden-validated
  for the family.
- **int4 weight quantization** (`decoder.Load(…, Quant: "int4")`) — group-wise
  symmetric 4-bit on the projections (group size 32: a per-group f32 scale, two
  nibbles per byte; `linalg.QuantizeGroupsInt4` + a dequant-per-tile
  `MatmulBTQ4`), ~⅛ f32 on those weights. The token embedding **and** LM head
  stay int8 (they are the tied head — 4-bit there flips the argmax), mirroring
  how GGUF Q4_K_M keeps `token_embd`/`output` at Q6_K. Streams at load and works
  for safetensors, GPT-2, and GGUF (the demo chats from a bare `.gguf` under
  `--quant int4`). Validated on TinyLlama 1.1B: argmax preserved, cosine 0.994
  vs f32 (on par with Q4_K_M's own 0.9975). int4 is a big-model tool — on a 270M
  it is lossy enough to move the top token, so its strict gate runs on TinyLlama.

### Changed

- **Quantized matmuls are now SIMD** — `linalg.MatmulBTQ4` and `MatmulBTQ8` widen
  each weight group/row into a reused scratch buffer and run the AVX2/NEON
  `dotF32` kernel over it (applying the scale at write-back), instead of a scalar
  multiply-accumulate loop. On a decode-step shape (M=1, K=N=2048): int4 **~6.7×**
  (8.3 → 1.2 ms), int8 **~6.9×** (3.0 → 0.43 ms). Outputs unchanged within float
  reassociation (`TestMatmulBTQ4_matchesDequant` relL2 ≤ 1e-5); decoder quant
  accuracy identical. (An int8×int8→int32 fixed-point kernel could go further.)
- **`embed.OpenGGUFMmap`** — memory-map a `.gguf` (read-only, MAP_PRIVATE)
  instead of `os.ReadFile`-ing it onto the heap, so the raw quantized bytes live
  in reclaimable page cache. `decoder` and `tokenizer` GGUF loads now use it:
  the decoder dequantizes tensor-by-tensor off the mapping then `Close`s it
  (weights are fresh copies, so nothing dangles), and `tokenizer.LoadGGUF` no
  longer pages in the multi-GB weights at all to read head-of-file metadata (its
  GGUF test dropped from ~0.5 s to ~0.03 s). Parse is bit-identical to the heap
  path (`TestGGUFMmap_matchesHeap`). Combined with streaming int8 below, a big
  quantized `.gguf` no longer needs the whole file on the heap *plus* the model
  in f32 to load. Unix only (`syscall.Mmap`), like `OpenSafetensorsMmap`;
  `OpenGGUF` (heap) remains for other platforms.
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
  signatures are unchanged.

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
