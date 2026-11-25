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
