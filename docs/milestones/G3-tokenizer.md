# Milestone G3 — byte-level BPE tokenizer (Qwen / Llama-3 family)

Parent plan: [`docs/multi-model-plan.md`](../multi-model-plan.md) §5.2, §7 (G3).
Touches: `tokenizer/bytelevel.go` (new — byte-level pipeline), `tokenizer/
sentencepiece.go` (shared merge core + mode detection + generic specials),
`tokenizer/doc.go`, `scripts/pin_qwen3_tokenizer.py` (new oracle),
`tokenizer/bytelevel_test.go` (new parity test).

Status: **DONE & validated on Linux 2026-06-02.** The pure-Go byte-level BPE
reproduces HF `tokenizers` **id-for-id** on Qwen3-1.7B: the committed golden (20
edge-case prompts, encode ± special and decode) and a throwaway adversarial
differential of **161,972 inputs** (every BMP scalar value + 80k random unicode
strings + whitespace/digit/contraction edges + repo Go source lines + embedded
special tokens) → **0 mismatches**. With G2's forward pass, **Qwen3-1.7B now
chats end-to-end in pure Go** — no HF-supplied token ids:

```
$ go run ./demo/gemma --model testdata/qwen3-1.7b --prompt "The capital of France is" --max 12
loaded 28-layer model (hidden 2048, vocab 151936) in 4.4s [backend=cpu quant=f32]
The capital of France is Paris. The capital of Italy is Rome. The capital of
```

## What it proves

The M2 merge machinery is family-agnostic. Generalizing to the largest bucket
of community checkpoints (Llama / Mistral / Qwen) was **swap the pre/post
processing, keep the merge core** — the ordered lowest-rank-pair merge loop is
literally shared (`mergeSymbols`). Only how text becomes initial symbols (and
ids become text) differs.

| stage | Gemma (M2, modeGemma) | byte-level (G3, modeByteLevel) |
|---|---|---|
| normalizer | ASCII space → `▁` | **NFC** (`x/text/unicode/norm`) |
| pretokenizer | none (whole gap → BPE) | **GPT-2 split regex** |
| initial symbols | per-rune, `<0xNN>` byte-fallback | **byte → printable rune** (no fallback) |
| whole-word shortcut | — | **`ignore_merges`** (vocab hit wins pre-merge) |
| merge core | `mergeSymbols` | **`mergeSymbols`** (same) |
| decode | `▁` → space, fuse `<0xNN>` | **rune → byte**, interpret as UTF-8 |
| specials | required (`<bos>`…) | from `tokenizer_config.json`, default none |

## The hard parts

- **The pretokenizer is hand-written, not a regexp.** The Qwen/Llama-3 split is
  the GPT-2 regex
  `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`.
  Go's `regexp` (RE2) **can't express the `\s+(?!\S)` negative lookahead**, so
  `splitGPT2` walks the runes applying the seven alternatives in priority
  order, each greedy — the same ordered leftmost-match semantics HF's regex
  engine uses. The lookahead is the mechanism that makes a single leading space
  attach to the *following* word (`"  two"` → `["Ġ", "Ġtwo"]`) while a trailing
  whitespace run stands alone (`"trailing   "` → `["trailing", "ĠĠĠ"]`); both
  are reproduced exactly. Anchored against HF's `pre_tokenize_str` on every edge
  (contractions case-insensitive, tab-as-prefix, per-digit split, newline runs)
  before writing the Go.
- **`ignore_merges=true`.** Qwen emits a pretoken whose byte-encoded form is
  itself a vocab entry *directly*, before any merging. Skipping this drifts on
  common whole-word tokens.
- **Specials live outside `model.vocab`.** Qwen keeps its 26 `<|im_*|>` /
  `<|endoftext|>` tokens only in `added_tokens`, with ids *above* the vocab
  range. `Load` now folds `added_tokens` into `idToPiece` (and `vocab`) so an
  emitted special id can be decoded — the added-token trie (M2) already handled
  the encode side.
- **No BOS at encode.** Qwen's post-processor is plain `ByteLevel` and
  `bos_token` is null, so `encode(add_special_tokens=True) == encode(False)`.
  `Encode(addBOS=true)` is a no-op when `special.BOS < 0`.

## Acceptance criteria — all met

- [x] `go build ./...` / `go vet ./...` clean; default checkout stays green
      (parity test **skips** when the checkpoint/golden are absent).
- [x] `scripts/pin_qwen3_tokenizer.py` dumps `testdata/qwen3_tokenizer_golden.json`
      (committable, ~9 KB) from HF via the `tokenizers` lib.
- [x] `TestByteLevel_qwen3GoldenParity`: every golden prompt encodes id-for-id
      (± add-special) and `Decode` reproduces HF's rendering exactly.
- [x] M2 Gemma golden **byte-identical** (regression intact — shared core).
- [x] Adversarial differential (throwaway, not committed): 161,972 inputs → 0
      id-mismatches vs HF.
- [x] End-to-end: `demo/gemma --model testdata/qwen3-1.7b` generates coherent
      text from a real prompt through the pure-Go tokenizer + forward pass.

## Pointers

- Golden oracle: `scripts/pin_qwen3_tokenizer.py` (uses `tokenizers`, no torch).
- Reference behavior: HF `tokenizers` `Tokenizer.from_file(tokenizer.json)`.
- Files: `tokenizer/bytelevel.go` (byte-level pipeline + `splitGPT2`),
  `tokenizer/sentencepiece.go` (`mergeSymbols`, `Load` mode detection),
  `tokenizer/added.go` (added-token trie, shared from M2),
  `tokenizer/bytelevel_test.go` (parity test).

## Known follow-ups (not blocking G3)

- **Chat template + multi-EOS in the demo.** ✅ **DONE 2026-06-02.**
  `tokenizer.ChatStyle()` detects ChatML vs Gemma from the vocab; `demo/gemma-web`
  renders the matching template (`buildChatML`/`buildGemmaPrompt`). Multi-EOS is
  general: `decoder.resolveEOSIDs` now folds `generation_config.json`'s
  `eos_token_id` into config.json's, so Qwen3 stops on either `<|im_end|>`
  (151645) or `<|endoftext|>` (151643). Verified end-to-end: Qwen3-1.7B answers
  and stops cleanly at `<|im_end|>` in the web chat.
- **Llama-3 parity.** Same byte-level pipeline; the split regex differs only
  slightly. Drop in a Llama-3 checkpoint + golden to confirm (no code expected).
- **Unigram (T5) family.** A different segment algorithm (Viterbi) — out of
  scope for G3, noted in the plan §5.2.
- **Perf.** `mergeSymbols` is the M2 O(n²)-per-piece loop; byte-level pretokens
  are short so it's sub-ms, but the M7 priority-queue merge would lift the
  worst case (a long no-space blob) if profiling ever flags it.
