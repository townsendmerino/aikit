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
3. **Tokenizer coverage.** aikit's tokenizer is **WordPiece-only** today
   (`embed/tokenize.go` rejects any `model.type != "WordPiece"`). WordPiece (BERT / MiniLM /
   BGE / arctic) is covered; the multilingual set (XLM-R, bge-m3, multilingual-e5) is
   **SentencePiece/Unigram, which does NOT yet exist** — this is the real Phase 2 blocker,
   not the byte-fallback that was assumed here. Hold the same **HF-exact id-parity** bar the
   decoder tokenizers already meet. Until it lands, XLM-R models load **forward-only** (run on
   pre-tokenized ids; `Encode(text)` errors — see the loader's best-effort tokenizer).
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

## Bucket C — One new primitive (1), deferred

`nomic-embed-text-v2-moe` is a BERT encoder with an **MoE FFN**; `encoder` has no MoE path.
Bounded but real, and it's a single entry — **defer until A and B land**.

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
- **Phase 2 — position offset certified, tokenizer blocked.** The XLM-R **position-id-offset
  forward** is certified against `xlm-roberta-base` (`posOff=2`), hidden-state maxΔ 1.7e-05
  over 6 cases incl. non-Latin, with break-it-first zeroing the offset (`encoder/xlmr_test.go`,
  `scripts/pin_xlmr.py`). What remains for end-to-end XLM-R is the **Unigram/SentencePiece
  tokenizer** (item 3 above) — the single missing component. `LoadBERT` is now best-effort on
  the tokenizer, so these models load and run forward on pre-tokenized ids today.

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
