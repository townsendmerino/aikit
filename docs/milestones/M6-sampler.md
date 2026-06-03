# Milestone M6 — sampler + streaming (the demo generates)

Parent plan: [`docs/gemma-decoder-plan.md`](../gemma-decoder-plan.md) §6 (M6).
Scaffold this fills in: `decoder/sampler.go` (`Sample`), `demo/gemma/main.go`.

Status: **DONE & validated on Linux 2026-06-02.** `demo/gemma` runs a real
checkpoint end to end — greedy or temperature/top-k/top-p sampling — streaming
the completion to stdout.

## Goal

Turn the validated forward pass into a usable generator: temperature, top-k,
top-p, seeded reproducibility, EOS/stop handling, and a streaming demo CLI.

## What was implemented

- **`Sampler.Sample`** — `Temperature ≤ 0` is greedy (argmax, ignores
  top-k/top-p); otherwise softmax at temperature → optional top-k and/or top-p
  nucleus filter → multinomial draw from the seeded RNG. (`softmaxStable` and
  `topFilter` were already present; this wires the draw.)
- **Demo streaming** — `demo/gemma` was already wired end to end (tokenizer →
  model → `Generate` → channel). Fixed the output path: instead of
  per-token `DecodePiece` (which emits a lone, possibly-partial byte-fallback
  byte), it decodes the running id slice and prints only up to the last
  complete UTF-8 rune, holding back partial multibyte sequences until they
  finish (`completeUTF8Len`).
- EOS/stop already wired at M4 (`isStop`, `Config.EOSIDs`).

## Validation

- **It runs.** `go run ./demo/gemma --model testdata/gemma-3-270m --prompt "The
  capital of France is" --max 20` →
  `"… Paris. It is the most visited city in the world. …"` (greedy).
  With `--temp 0.8 --top-k 40 --seed 1` it samples, and the same seed reproduces
  the same text.
- **Sampler unit tests** (`sampler_test.go`, no model needed): greedy reduction;
  `temp 0` stays greedy even with top-k/top-p set; top-k=1 ⇒ argmax; top-k/top-p
  restrict the draw to the correct support; same seed ⇒ identical sequence,
  different seed ⇒ differs; and the empirical draw distribution matches
  `softmax` within 0.01 over 200k draws.
- **Streaming integration** (`TestGenerate_streamMatchesGolden`): the public
  `Generate` channel path, greedy, reproduces the M4 continuation ids exactly —
  covering the prefill → decode-loop → sampler → channel wiring the M4 test's
  hand-rolled loop bypassed.

## Acceptance criteria — all met

- [x] `go build`/`vet`/`gofmt` clean; `demo/gemma` generates from a real prompt.
- [x] Greedy + temperature/top-k/top-p, seed-reproducible.
- [x] Sampler mechanics + distribution unit-tested; streaming path integration-tested.
- [x] Slow model-backed tests gated behind `-short`.

## Known follow-ups (not blocking M6)

- **Perf (M7).** Naive backend; ~0.2 s/token. The §1 shared-linalg refactor
  (SIMD/parallel matmul) makes it interactive.
- **Repetition penalty / chat template.** The plan mentions repetition handling;
  a 270M base model loops on greedy (expected). Repetition penalty and the
  `<start_of_turn>` chat template are easy follow-ups when wanted.
