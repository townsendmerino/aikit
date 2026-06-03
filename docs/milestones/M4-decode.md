# Milestone M4 — KV-cache multi-token decode (greedy continuation parity)

Parent plan: [`docs/gemma-decoder-plan.md`](../gemma-decoder-plan.md) §6 (M4), §10.
Scaffold this fills in: `decoder/model.go` (decode loop + `isStop`),
`decoder/config.go` (EOS ids), `decoder/sampler.go` (`StopIDs`).

Status: **DONE & validated on Linux 2026-06-02.** A 48-token greedy continuation
matches HF **id-for-id**, and the decoded string matches exactly end to end.

## Goal

Prove the decode *loop*, not just one step: append K/V per token, advance the
RoPE position, and attend over the growing cache — across many steps — stay
bit-faithful to HF greedy decoding. This is the check M3's single forward can't
make (cache, position, and causal-mask bugs only surface across steps).

## What was implemented

- **Decode loop** already shaped in M3 (cache-based `forward`); M4 adds the
  oracle + parity test and the stop wiring below.
- **EOS / stop wiring.** `Config.EOSIDs()` parses `eos_token_id` (scalar *or*
  list); `Model` caches the ids at load; `isStop` ends generation on a config
  EOS or a caller `SamplingParams.StopIDs` entry (e.g. `<end_of_turn>` for
  chat). `Generate` now greedy-decodes a usable stream (non-greedy sampling is
  still M6).

## The oracle

`scripts/pin_gemma_generate.py` runs HF greedy (float32, `do_sample=False`) and
**forces exactly N new tokens** (`min_new_tokens == max_new_tokens`, EOS
suppressed) so the sequence is deterministic and the Go loop reproduces it
without EOS-truncation ambiguity. Writes `testdata/gemma_generate_golden.json`
(committed, ~1 KB): prompt + ids, the N continuation ids, and the decoded text.

For "The capital of France is" the greedy continuation is repetitive (expected
for a 270M base model): `" Paris. It is the most visited city in the world. …"`.

## Acceptance criteria — all met

- [x] `go build ./...` / `go vet ./...` / `gofmt` clean; default checkout green
      (M4 test **skips** when checkpoint/golden absent).
- [x] `TestDecode_greedyContinuationParity`: 48 greedy tokens match
      `continuation_ids` id-for-id off the KV cache.
- [x] Decoding those ids through the **M2 tokenizer** reproduces
      `continuation_text` exactly (end-to-end M2+M3+M4).
- [x] `isStop` honors config EOS ids and `SamplingParams.StopIDs`.

## Known follow-ups (not blocking M4)

- **Sliding window (M5).** The sequence (≤54 positions) is far below the 512
  window, so local==causal here. `cache.WindowStart` is wired but the
  window-eviction path is unverified past 512 tokens — that's M5's golden.
- **Perf (M7).** ~11 s for 48 steps on the naive backend; the vocab-sized LM
  head per step dominates. Wire the SIMD/parallel linalg (plan §1).
- **Sampler (M6).** Temperature/top-k/top-p still stubbed; greedy only.
