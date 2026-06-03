# Milestone M5 — sliding-window attention (parity past 512 tokens)

Parent plan: [`docs/gemma-decoder-plan.md`](../gemma-decoder-plan.md) §6 (M5), §10.
Scaffold this fills in: `decoder/kvcache.go` (`WindowStart`), `attention.go`.

Status: **DONE & validated on Linux 2026-06-02.** A 748-token prompt (window
512) matches the HF float32 oracle: **cosine 0.99999999999**, argmax identical
(`' A'`).

## Goal

Local (`sliding_attention`) layers must attend only the last 512 keys; global
(`full_attention`) layers attend everything. M3/M4 used sequences ≤ ~54
positions — far below the window — so local behaved identically to full causal
attention and this path was never actually exercised. M5 drives a prompt past
512 tokens and checks the logits still match.

## The two bugs this milestone caught (and fixed)

Both were latent in the M3/M4 code and invisible until a sequence exceeded the
window:

1. **Mutable position.** Attention called `cache.WindowStart(global)`, which
   read `c.pos`. But `c.pos` only advances on the *last* layer's `Append`
   within a forward, so layers 0–16 saw `pos` while layer 17 saw `pos+1` — the
   window boundary shifted by one on a single layer. Fixed by passing the
   query position explicitly: `WindowStart(pos, global)`.
2. **Off-by-one.** The start was `pos − window`, admitting `window+1` keys.
   Gemma's mask admits key `j` iff `pos − j < window`, i.e. `[pos−W+1, pos]` =
   exactly `W` keys. Fixed to `pos − window + 1`.

Neither could surface in M3/M4 (windows never engaged); the M5 long-prompt
golden is what made them observable. With both fixed, cosine went to ≈1 − 1e-11.

## The oracle

`scripts/pin_gemma_window.py` builds a **deterministic, varied** >512-token
prompt (varied so the evicted early tokens genuinely differ from the recent
ones — otherwise eviction would be a no-op and the test couldn't tell correct
windowing from none), runs HF float32, and dumps the last-position logits
(compact `testdata/gemma_window_golden.json`, committed; full
`testdata/gemma_window_full.json`, gitignored, for exact cosine).

## Acceptance criteria — all met

- [x] `go build`/`vet`/`gofmt` clean; default checkout green (test **skips**
      when assets absent, and under `-short`).
- [x] `TestForward_slidingWindowParity`: 748-token prompt, argmax identical,
      sample/top-k within 5e-3, full cosine ≥ 1 − 1e-4 (got 1 − 9e-12).
- [x] The slow model tests (this one ~63 s, M4 decode ~11 s on the naive
      backend) are gated behind `testing.Short()`.

## Known follow-ups (not blocking M5)

- **Perf (M7).** ~63 s for the 748-token prefill is all naive-backend matmul;
  the §1 SIMD/parallel linalg makes this interactive.
- **Sampler (M6).** Temperature/top-k/top-p still stubbed; greedy only.
