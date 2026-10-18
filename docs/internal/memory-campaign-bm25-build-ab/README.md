# BM25 `Build` A/B, v1.70.2 vs 8ba2b3c (memory campaign, job 5)

**VERDICT: ACCEPT** — `Build` within 1.10× of v1.70.2 on both corpora. nobara (Ryzen 7 3700X, linux/amd64,
go1.27.2), night queue, 2026-10-17 18:30–18:39 PDT, under the timing lock.

Why it exists: item 3 (the postings arena) measured `Build` +5.6% in process on the M1 Pro, and `perfgate` has no
BM25 shape. The rule was registered before the harness was written
([`../task-memory-campaign.md`](../task-memory-campaign.md), "Job 5"; commit `0e1b717`): per corpus, the median of
the paired new ÷ old ratios; accept at ≤ 1.10× on both corpora, investigate above that, no automatic revert.

| corpus | v1.70.2 `Build` | 8ba2b3c `Build` | new ÷ old, median (rounds) | 8ba2b3c `Builder.Add` + `Build` |
|---|---|---|---|---|
| synthetic, 100,000 chunks (25.6 M tokens) | 1,698 ms | 1,731 ms | **1.026×** (0.995–1.088) | 1,727 ms — 1.000× new `Build`, 1.020× v1.70.2 |
| aikit's tree, 4,361 chunks (1.19 M tokens) | 83.8 ms | 82.6 ms | **1.030×** (0.716–1.339) | 86.7 ms — 1.020× new `Build`, 1.022× v1.70.2 |

Medians of 15 paired rounds; each round's value is one process's median of 5 timed builds after a warm-up. The tree
corpus is small enough to be noisy (builds of 62–102 ms); the synthetic corpus is the tight measurement.

## Method

- Both arms are built from git worktrees of the same clone (`v1.70.2` = `8eb0987`, and `8ba2b3c`), each linking the
  same benchmark source against its own `bm25`.
- Both read identical input: each corpus is written once as a `bm25.MarshalTokens` blob, by a generator linked
  against `8ba2b3c`. Tree: every `.go` and `.md` file at `8ba2b3c`, minus `.git`, `testdata` and `assets`, in 40-line
  chunks through `bm25.Tokenize`. Synthetic: 100,000 seeded documents of 64–448 tokens, Zipf (s = 1.1) over 50,000
  terms.
- 15 rounds, ABBA order (v1.70.2 first in odd rounds), one process per arm per round, then the ratio per round.

## Files

- [`results.jsonl`](results.jsonl) — every process's samples (60 lines: 15 rounds × 2 corpora × 2 arms).
- [`run.log`](run.log) — the night job's log.
- [`_harness/`](_harness) — the harness exactly as it ran (byte-identical to nobara's copy):
  `run-bm25ab.sh` (driver), `gen/` (corpora), `bench/` (timer; `Builder.Add` behind `-tags aikitnew`),
  `analyze.py` (the rule). The leading underscore keeps the Go tool from building it as part of the module. It is a
  record, not a gate: re-run it with `REPO=<clone> WORK=<dir> OLD=<ref> NEW=<ref> _harness/run-bm25ab.sh`, and
  `python3 _harness/analyze.py results.jsonl` reproduces the table above.
