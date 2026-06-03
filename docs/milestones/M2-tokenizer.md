# Milestone M2 — Gemma 3 tokenizer (byte-fallback BPE, HF-exact)

Parent plan: [`docs/gemma-decoder-plan.md`](../gemma-decoder-plan.md) §6 (M2), §9, §10.
Scaffold this fills in: `tokenizer/sentencepiece.go` (+ new `tokenizer/added.go`).

Status: **DONE & validated on Linux 2026-06-02.** Parity test green against the
real `google/gemma-3-270m` tokenizer; verified id-for-id over 215k+ stress
inputs (every BMP codepoint + astral sweep + random/adversarial) with **zero
mismatches**.

## Goal

Pure-Go `Encode`/`Decode` that reproduces HF's tokenizer **exactly** (ids must
match, BOS included) — a one-token drift silently wrecks generation, so the bar
is byte-exact equality, not a tolerance.

## The key correction to the plan

The plan (§9) hedged "unigram/BPE." The shipped Gemma 3 `tokenizer.json` is a
**byte-fallback BPE** model — an explicit ordered merge table (~515k merges),
not a unigram Viterbi scorer. So the algorithm is greedy merge-by-rank, not
best-path segmentation. We load `tokenizer.json` (HF format) rather than the
legacy `tokenizer.model` SP protobuf: the JSON carries the same 262k vocab
*plus* the explicit merges and the normalizer/decoder pipeline, which is what
makes exact HF parity tractable.

## The pipeline (must match HF `tokenizers` step-for-step)

1. **Added-vocabulary split (on raw text, before everything).** All 6415 added
   tokens — control tokens (`<bos>`, `<start_of_turn>`, …) *and* the
   newline-run tokens (`\n`, `\n\n`, …) — are matched leftmost-**longest** at
   each byte position and emitted as their id directly; the gaps between them go
   through normalize+BPE. (Longest match matters: `\n\n` → one id, not two.)
   Implemented as a byte trie (`tokenizer/added.go`).
2. **Normalize (gaps only).** Replace every ASCII space with `▁` (U+2581).
   That is the *whole* normalizer — no NFC/NFKC, no dummy-prefix, and tabs /
   newlines are left literal.
3. **BPE (per gap).** Initial symbols are per-rune; a rune absent from the vocab
   is decomposed to its UTF-8 bytes as `<0xNN>` tokens (byte fallback). Then
   repeatedly merge the lowest-rank adjacent pair (leftmost on ties) until none
   remain; map final symbols → ids.
4. **Decode.** id → piece, `▁` → space, and fuse runs of `<0xNN>` pieces back
   into raw UTF-8 bytes. Special tokens render as their literal surface form
   (`<eos>`); stopping at EOS is the generation loop's job, not Decode's.

## Acceptance criteria — all met

- [x] `go build ./...` / `go vet ./...` clean; default checkout stays green
      (parity test **skips** when the checkpoint/golden are absent).
- [x] `scripts/pin_gemma_tokenizer.py` dumps `testdata/gemma_tokenizer_golden.json`
      (committable, ~8 KB) from HF via the `tokenizers` lib.
- [x] `TestEncodeDecode_goldenParity`: every golden prompt encodes id-for-id
      with **and** without BOS, and `Decode` reproduces HF's rendering exactly.
- [x] `Special()` resolves BOS/EOS/Pad/StartOfTurn/EndOfTurn from the vocab.
- [x] Adversarial differential (throwaway, not committed): 215,285 inputs incl.
      every BMP codepoint → 0 id-mismatches vs HF.

## Pointers

- Golden oracle: `scripts/pin_gemma_tokenizer.py` (uses `tokenizers`, no torch).
- Reference behavior: HF `tokenizers` `Tokenizer.from_file(tokenizer.json)`.
- Files: `tokenizer/sentencepiece.go` (Load/Encode/Decode/BPE),
  `tokenizer/added.go` (added-token trie),
  `tokenizer/sentencepiece_test.go` (parity test).

## Known follow-ups (not blocking M2)

- **Perf (defer to M7).** The merge loop is naive O(n²) per gap — fine for
  prompts (sub-ms), but a long no-space blob is the worst case. Swap for the
  linked-list + priority-queue merge if profiling flags it.
- **Streaming decode.** `DecodePiece` returns a lone byte-fallback piece as a
  possibly-incomplete UTF-8 byte; a streaming caller must buffer across calls
  (a demo concern, wired in M6).
