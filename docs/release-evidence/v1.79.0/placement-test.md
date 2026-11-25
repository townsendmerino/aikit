# v1.79.0: is perfgate's contract-exp regression code placement? Registered 2026-11-24 17:44 PST PST, before the run

**Why.** Perfgate run 2 (`perfgate-run2.log`, nobara, Ryzen 7 3700X, linux/amd64, settled box, 13/47 shapes resolving 5%) read FAIL:
`BenchmarkAttendTileFused_expKind/contract` slower by 6.20% at np576 (floor 2.00%), 6.87% at np1024 (2.98%) and 6.95% at np2048 (3.61%),
its `f64math` siblings flat. None of the code it runs changed since v1.78.0 (`fusedattn.go`, `exp_contract*.go`, `matmul.go`); in
the new binary every contract-exp function sits 32 B further mod 64 (`go tool nm`), after the dispatch functions #39 added to
`attn_acc64_amd64.go`. Run 1 (`perfgate-run1-under-load.log`) started under load and is not evidence.

**Arms**, both v1.78.0 source, built with Go 1.27 and `GOWORK=off`: **plain** (v1.78.0), and **padded** (v1.78.0 plus one 32-byte
non-inlined function in `attn_acc64_amd64.go`, kept linked by an `init`, which moves `expContractImpl`, `softmaxContractWithMaxImpl`
and `attendTileFused` from 0 to 32 mod 64 and `AttendTileFusedContractExp` from 32 to 0, exactly v1.79.0's placement; `go tool nm`).

**Measure.** `BenchmarkAttendTileFused_expKind/np{576,1024,2048}/contract`, `-benchtime 1s`, one process a cell, A B B A × 5
blocks (10 pairs a shape, A = plain, B = padded), under the exclusive lock, after the 1-minute load has held under 0.3 for two minutes.

**Rule.** **REPRODUCES** if, at all three shapes, the median pair ratio padded ÷ plain (ns/op) is ≥ 1 + (run-2 Δ − its floor):
≥ 1.0420 at np576, ≥ 1.0389 at np1024, ≥ 1.0334 at np2048. **Anything else: DOESN'T**, and the regression is treated as real.
REPRODUCES: the 1.79.0 section carries a `> **PERFGATE EXCEPTION: …**` blockquote citing run 2 and this test, and the release goes on.

## Result, read 2026-11-24 17:51 PST: **REPRODUCES**

One run, 17:47–17:51 PST under the exclusive lock, start load 0.07 / 0.10 / 0.37 (1/5/15 min; the benchmark's own threads raised the
1-minute figure to 2.8 by the end, on both arms alike, interleaved). Cells `placement-cells.tsv`, log `placement-run.log`.

| shape | median padded ÷ plain (10 pairs) | range | bar | |
|---|---|---|---|---|
| np576 | **1.0676** | 1.028–1.071 | ≥ 1.0420 | meets |
| np1024 | **1.0656** | 1.057–1.076 | ≥ 1.0389 | meets |
| np2048 | **1.0700** | 1.061–1.085 | ≥ 1.0334 | meets |

v1.78.0's own source with one 32-byte function added ahead of the contract-exp code is 6.6–7.0% slower on these shapes, the size of
perfgate run 2's 6.2–7.0%: the "regression" is where the linker places unchanged code, not anything this release changed.

**Note.** Before the run, a 3-iteration smoke run of the plain binary (well under a second, to check the output parses) ran without the
timing lock's shared hold. No timed run was in progress then, so no cell of this test or of anyone else's is affected.
