# Embedding-model coverage — broaden `encoder` to the popular embedders

> **BLUF.** The serving surface already exists (goinfer's `/v1/embeddings` + `--embed-model`
> + `aikit/encoder`), so this is **coverage breadth, not new plumbing**. aikit *is* the
> retrieval/embedding toolkit, and today it runs essentially **one** embedder
> (CodeRankEmbed). Of the ~12 popular embedding models, **~9 are loader + tokenizer +
> pooling work in `encoder` with no new architecture**, 2 don't touch `encoder` at all, and
> 1 needs a new primitive.

## Why this is on-mission

- "Runs the popular embedding models, cgo-free, in-process" is aikit's **core claim** — and
  right now it can't back it. Unlike an LLM family add (which serves goinfer), this serves
  aikit's own product.
- It **compounds with** [`task-native-gpu.md`](task-native-gpu.md): batch encoding is the #3
  GPU-fit workload there, so coverage × GPU-batch-encode multiply rather than compete.

## Bucket A — BERT-family bi-encoders (~9 of 12): pure `encoder` breadth

Entries: `all-minilm`, `mxbai-embed-large`, `bge-large`, `snowflake-arctic-embed`,
`granite-embedding`, `nomic-embed-text` — plus the XLM-R-based `bge-m3`,
`paraphrase-multilingual`, `snowflake-arctic-embed2` (same shape, different embedding
layout). All in `aikit/encoder`:

1. **Loader variants — make the implicit explicit.** BERT vs nomic-bert vs XLM-R differ in:
   position-embedding style (learned-absolute vs rotary), **position-id offset** (XLM-R
   starts at `padding_idx + 1` — a classic silent-wrong if missed), `token_type_embeddings`
   presence, and pre- vs post-LayerNorm placement. These are config-driven; declare them as
   fields rather than assuming one shape.
2. **Pooling as a declared property.** CLS vs mean vs last-token is currently implicit. Add
   a `Pooling` enum on the model, set from config, and assert it in the parity test.
3. **Tokenizer coverage.** aikit now covers **both** WordPiece (BERT / MiniLM / BGE / arctic)
   and **SentencePiece/Unigram** (XLM-R / bge-m3 / multilingual-e5). The Unigram path
   (`embed/tokenize_unigram.go`) is a pure-Go port of HF's pipeline — Precompiled charsmap
   normalizer (darts-clone double-array trie), Metaspace pre-tokenizer, Unigram Viterbi
   (`fuse_unk`, `unk_score = min_score - 10`), and TemplateProcessing — dispatched from
   `LoadTokenizer` by tokenizer.json shape (no public API change). It holds the **HF-exact
   id-parity** bar: byte-exact normalization over a per-codepoint sweep (U+0000..U+2FFFF) and
   id-exact `encode` over Latin/CJK/RTL/Devanagari/fullwidth/emoji/code, with break-it-first.
4. **Normalization + dims.** L2-normalize and `dimensions` truncate+renormalize already ship
   in the serve layer — confirm they compose per-model (see Matryoshka caveat below).

This is the cheap bulk: three-quarters of the list for work you do once.

## Bucket B — Decoder-as-embedder (2): no `encoder` change at all

`qwen3-embedding`, `embeddinggemma` are **causal decoders used as embedders**, and goinfer
already runs `qwen3` and `gemma3`. So this is a small **goinfer-side** path:

> existing decoder forward → pool (last-token / mean) → L2-normalize → out the existing
> `/v1/embeddings`

Reuses the decoder, its tokenizer, and the serve surface. The only new pieces are the
pooling head and the instruction-prefix convention — mirror `encoder`'s existing
query/document asymmetry (`input_type`).

**This bucket is tracked in goinfer, not here** — see goinfer
`docs/task-decoder-as-embedder.md` for the spec (the seam is aikit's `encoder.Encoder`
interface, which a decoder-backed embedder implements) and its live status. Nothing in
aikit needs to change for it.

## Bucket C — One new primitive (1), deferred

`nomic-embed-text-v2-moe` is a BERT encoder with an **MoE FFN**; `encoder` has no MoE path.
Bounded but real, and it's a single entry — **defer until A and B land**.

> **Landed — and the sketch above was wrong in two ways.** It is a `nomic_bert`, not a plain
> BERT, so the RoPE/post-norm base was already supported; and it needed **three** new pieces,
> not one (MoE FFN, dense GELU MLP with biases, attention biases). See the Phase 4 status entry.

## Parity discipline (per family, non-negotiable)

An embedder that silently pools the wrong token still returns *plausible-looking* vectors.
That's the silent-wrong class, and only a reference comparison catches it:

- **Vector gate:** cosine vs the HF sentence-transformers reference over a fixed sentence
  set (the CodeRankEmbed path already holds ~0.997 — extend the pattern per family).
- **Retrieval gate:** encode a small fixed corpus + queries and assert **top-k ordering**
  matches the reference. Cosine can be high while the *ranking* is wrong — the vector gate
  alone is not sufficient.
- **Break-it-first:** perturb the pooling mode or the position-id offset, confirm both gates
  go **RED**, revert. A gate that can't fail isn't a gate
  (see goinfer's `docs/parity-hunt-playbook.md`).

## Phasing

- **Phase 1 —** the `encoder` API work (loader variants, declared pooling, WordPiece), landing
  3 representatives with gates: `all-minilm`, `bge-large`, `nomic-embed-text`.
- **Phase 2 —** the XLM-R trio (`bge-m3`, `paraphrase-multilingual`, `arctic-embed2`) —
  multilingual tokenizer + the position offset.
- **Phase 3 —** decoder-as-embedder, goinfer side (`qwen3-embedding`, `embeddinggemma`).
- **Phase 4 (deferred) —** MoE encoder for `nomic-embed-text-v2-moe`.

Each phase independently shippable; **no model is listed as supported until its gates are
green.**

### Status (2026-07-20)

- **Phase 1 — done, gates green.** Loader variants (position-id offset, optional
  `token_type_embeddings`, model-name tensor prefix), declared pooling (CLS/mean read from
  `1_Pooling/config.json`). Certified against real HF references at cosine 1.000000 with
  break-it-first: `all-MiniLM-L6-v2` (mean BERT), `bge-small-en-v1.5` (CLS BERT),
  `nomic-embed-text-v1.5` (mean nomic-bert/RoPE). See `encoder/{bge,nomic_embed}_test.go`.
- **Phase 2 — XLM-R certified end-to-end.** Both halves now green against `xlm-roberta-base`:
  (a) the **Unigram/SentencePiece tokenizer** (`embed/tokenize_unigram.go`) reproduces HF
  id-for-id — byte-exact Precompiled normalization over the full U+0000..U+2FFFF sweep and
  id-exact `encode` over a broad multilingual/emoji/code set, with break-it-first
  (`embed/tokenize_unigram_test.go`, `scripts/pin_xlmr_tokenizer.py`); (b) the **position-id
  offset** forward holds at `posOff=2`, hidden-state maxΔ 1.7e-05, offset-zeroing break-it-first
  (`encoder/xlmr_test.go`). `LoadBERT` wires the tokenizer, so `Encode(text)→hidden` is certified.
- **Phase 2 — first multilingual embedder certified full-stack.** `intfloat/multilingual-e5-base`
  (genuine XLM-R + SentencePiece + mean-pool + a real sentence-transformers head) is certified
  end-to-end at **cosine 1.000000** over 11 cases (Latin, CJK, Cyrillic, Arabic, German ß/umlaut):
  hidden-state parity AND `Encode(text)` — tokenizer + `posOff=2` + mean pooling + forward in one
  gate — with CLS-vs-mean and offset break-it-first (`encoder/e5_test.go`, `scripts/pin_e5.py`).
- **Phase 2 — `bge-m3` certified full-stack (CLS).** The flagship multilingual retriever (24-layer
  XLM-R, 1024-dim, CLS pooling) is certified at **cosine 1.000000** over 13 cases
  (`encoder/bge_m3_test.go`, `scripts/pin_bge_m3.py`). It exercises two config-driven tokenizer
  variations the parser now handles generically: a normalizer `Sequence[Precompiled,
  Replace(" {2,}"→" ")]` and a **bare Metaspace** pre-tokenizer (no WhitespaceSplit) — validated
  id-exact against the raw HF tokenizer including trailing-▁ / multi-space edge cases
  (`embed/tokenize_unigram_bgem3_test.go`). Note: bge-m3 ships only `pytorch_model.bin`; convert
  to safetensors for aikit's loader. With this, every same-repo Phase-1/2 model is certified.
- **Phase 3 (Bucket B) — done, in goinfer.** Decoder-as-embedder landed and is certified there
  (`goinfer` commit `9168f82`): Qwen3-Embedding-0.6B at **cosine 1.0000000**, last-token pooling
  over a new `decoder.HiddenLast` seam, behind a `decoderEmbedder` implementing aikit's
  `encoder.Encoder`. No aikit change was required, as predicted. `embeddinggemma` is deferred
  with the reason recorded (HF repo still gated, HTTP 401). See
  `goinfer/docs/task-decoder-as-embedder.md`.
- **Phase 4 (Bucket C, MoE) — done, certified.** `nomic-embed-text-v2-moe` is certified full-stack
  at **cosine 1.000000** over 9 cases, worst hidden maxΔ 3.2e-05 (`encoder/nomic_moe_test.go`,
  `scripts/pin_nomic_moe.py`), with break-it-first on the routing itself (top-1 instead of top-2,
  and forcing every token to one expert — both diverge by ~2–7 vs 2e-05).

  Reading the config first paid off again: the bucket called this "one new primitive," but it is
  **three**, and the base architecture was already supported. It is a `nomic_bert` (not a plain
  BERT), so RoPE/post-norm came free from the certified v1.5 path, and its XLM-R SentencePiece
  tokenizer came free from Phase 2 — but it needed (a) the **top-2-of-8 MoE FFN** on odd layers
  (`i%moe_every_n_layers == 1`), (b) a **dense GELU `fc1`/`fc2` MLP with biases** on even layers
  (v1.5 is SwiGLU, bias-free), and (c) **attention qkv/out_proj biases**. Two MoE details are
  load-bearing and easy to get subtly wrong: `W2` is applied **untransposed**, and the router
  softmax runs over **all** experts with the top-k weights taken as-is (`moe_normalize_expert_weights=false`),
  so they do not sum to 1.

  The padded-batch and q8 kernels are SwiGLU-only: batch falls back to the per-sequence forward
  for these checkpoints, and `LoadWeightsQ8` refuses them outright rather than quantize absent
  tensors into a plausible-looking wrong model.

## Coverage claim, generated not hand-maintained

Once models land, emit a generated embedder-coverage table (model → validated / pending,
pooling, tokenizer, dims), freshness-gated the same way `hardware-matrix.md` is. The
stale-cell lesson applies here too — a hand-maintained support list rots.

## Honest caveats

- These were bucketed **by name and reputation, not by reading each `config.json`.** The
  shapes are right; verify the per-model specifics before building.
- **`bge-m3` ships three retrieval heads** (dense / sparse / ColBERT). Scope to the **dense**
  vector unless there's a concrete pull for the others.
- **Matryoshka `dimensions` semantics differ per model** — honor each model's documented
  supported dims rather than truncating blindly, or you'll return degraded vectors that look
  fine.
