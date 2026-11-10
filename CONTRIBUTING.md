# Contributing to aikit

This is the entry point for the conventions that only live as tribal
knowledge today — scattered across [`RELEASING.md`](RELEASING.md),
[`docs/architecture.md`](docs/architecture.md), and code comments. It does
not repeat what those already say well; it points at them and fills the
gaps. aikit is [MIT-licensed](LICENSE).

## Quick start

```bash
GOWORK=off go build ./...
GOWORK=off go vet ./...
GOWORK=off go test ./...
```

**`GOWORK=off` is not optional.** `go.work` (gitignored, yours to keep —
see below) pins a Go version older than the modules require, and a
workspace overrides module resolution with your local directories, so a
command run without it can pass locally and fail for everyone else — or
the reverse. Every gate in `tools/` sets this for you automatically
(`tools/gpumod.Exec`); if you're invoking `go` directly, set it yourself.

aikit is **several Go modules in one repo**, not one:

- the root module (`github.com/townsendmerino/aikit`) — the pure-Go core;
- `tools/` — the gate commands this file describes, its own module so
  nothing it needs shows up in the root module's own `go.mod`/`go.sum`;
- `chunk/treesitter` — quarantines the `gotreesitter` dependency;
- `gpu` + its backend modules (`gpu/anncuda`, `gpu/annmetal`, …) —
  quarantine cgo (CUDA/Metal).

See [`docs/architecture.md`](docs/architecture.md) for the full package DAG
and why each boundary exists. If you only touch the root module, you can
ignore the others; `cd` into a module before running `go` commands in it
(`go run -C tools ./preflight`, not `go run ./tools/preflight`).

If you have a local dev workspace need, create your own `go.work` — it's
already gitignored, and nobody else's build depends on it. Never commit
one.

## Before you push

Run `preflight` — it mirrors what CI's core job runs, minus the parts that
need the network, a GPU, or `-race`'s triple runtime:

```bash
go run -C tools ./preflight            # gofmt, build, vet, lint, cgo-free build, go test
go run -C tools ./preflight --fast     # skip go test (CI still runs it with -race)
```

A clean preflight does not mean CI will be clean — it explicitly does not
run `-race`, the cgo-dependency guard, `aikit_checks`, fuzzing, or the gpu
jobs — but a red preflight means CI will be red too, so it is worth
running before every push, not just before a release.

**Every gate command in this repo ends its output with one line in the
same shape:**

```
VERDICT: PASS — 6/6 clean
VERDICT: FAIL — 2 check(s) failed — fix before pushing
VERDICT: INCONCLUSIVE — a required external tool was unavailable
```

That's [`tools/gate.Verdict`](tools/gate/gate.go) — every gate command
calls it (`preflight`, `releasegate`, `consumergate`, `gpugate`,
`gpudevice`, `gpupins`, `perfgate`), so grepping for `VERDICT:` across a
CI log or a terminal always finds the line that matters, in the same
format, regardless of which gate produced it. `INCONCLUSIVE` is never a
pass — it means the gate could not judge (a missing tool, an unreachable
network), and it is never silently folded into green.

`vulncheck` is the one deliberate exception: it never prints `VERDICT:`
at all, only `STATEMENT:` — a release-notes line, not a pass/fail
verdict, per [`RELEASING.md`](RELEASING.md) ("put its `STATEMENT:` line
in the release notes"). If you're adding a new gate, default to
`gate.Verdict`; only skip it if your gate's output is genuinely a
different kind of thing, the way vulncheck's is.

If you're touching a `gpu/*` backend, `tools/gpugate` and
`tools/gpudevice` are the mechanical and on-hardware halves of that
ritual — see their own doc comments. The full pre-tag sequence (which
gates run in which order, what's hand-run vs CI-run) is
[`RELEASING.md`](RELEASING.md); this file is about day-to-day
contribution, not cutting a release.

## API stability

aikit makes a semver promise on its **Hard tier** — most of the retrieval
core (`ann`, `bm25`, `fuse`, `topk`, `chunk`, `embed`'s stable surface,
`encoder`'s Hard-tier types) — and explicitly does not on **Experimental**
(new surface still settling, or acknowledged as likely to move — `sparse`
in full, plus named parts of `ann`/`embed`/`encoder`/`linalg`: see [README
§ Stability tiers](README.md#stability-tiers) for the current,
authoritative list — it changes as things graduate, so it is not
duplicated here).

If you're adding a new public symbol, decide which tier it's in *before*
shipping it, and say so in the PR. Two consumers (`ken`, `goinfer`) and
the public Hard-tier API are the reason this matters: a Hard-tier
breaking change is a bigger deal than an Experimental one, and `tools/
releasegate`'s `apidiff` check enforces the Hard tier automatically — an
incompatible change there fails the gate unless it's declared
Experimental in `experimentalSyms`/`experimentalMembers`
(`tools/releasegate/main.go`).

## Code conventions

- **Short variable names are deliberate, not sloppiness.**
  [`docs/why-short-variable-names.md`](docs/why-short-variable-names.md)
  is the philosophy — read it before flagging `M`, `q`, `sb`, or similar
  in review; it explains what to actually flag instead.
- **Numbered citations in comments** (`audit #16`, `item 11`, `perf-
  campaign A5`, `lens doc §4.3`) point at specific findings in this
  repo's own historical record — the 2026-07-25 engineering audit and the
  2026-07 performance campaign, both archived under
  [`docs/internal/archive/`](docs/internal/archive/) once their live work
  finished. `AUDIT-2026-07-25.md`'s findings are numbered `### N.`;
  `perf-campaign-2026-07/`'s plain-numbered items (`item 27`) are all in
  `perf-campaign-2026-07-28.md`, but its LETTERED items (`A5`) are a
  **different file**, `task-perf-handoff-linux.md` — the letter picks the
  doc, not just the item within one. When you cite one, cite it the same
  way: a short tag in the comment, the full story in the archived doc —
  not restated inline. If your own
  work produces a finding worth citing later, it deserves the same
  treatment: a durable record under `docs/internal/` (archived once
  closed), not just a comment that stops making sense once the context
  that produced it is gone. You'll also see `plan §N.M` and milestone
  tags (`M0`…`M24`, `M8c`) in older comments — those predate aikit's split
  out of ken and don't resolve to anything in this repo; see
  [`docs/architecture.md` § Numbered-citation
  index](docs/architecture.md#numbered-citation-index) for the full table
  of which style resolves where (or doesn't).
- **ADRs** (`ADR-NNN` in comments) live in the `ken` repo, not here —
  aikit's packages originated there. See [`docs/architecture.md` § ADR
  index](docs/architecture.md#adr-index) for the table of which ADR
  covers what, cited from where.

## Tests

- **`package foo` (internal) is the default** for a new `_test.go` file —
  it can see unexported state, which most tests need. Reach for
  **`package foo_test`** (external) only when that's the point: `Example`
  functions must live in the external package (a godoc/pkg.go.dev
  requirement, and it forces the example to read exactly as a consumer
  would write it), and some integration/recall gates (e.g.
  `ann/hnsw_int8_recall_test.go`) deliberately test only the public
  surface so they can't accidentally pass by reaching into an internal
  that a real caller can't touch.
- **Naming**: `_test.go` for correctness, `_bench_test.go` for
  benchmarks-only files, `example_*_test.go` for godoc `Example`s — kept
  in separate files so `go test` output isn't a wall of benchmark noise
  when you're only running correctness tests.
- **Break-it-first.** A gate that has never seen its own check fail is
  unproven — it might be structurally unable to fail (see
  [`docs/analyzer-canaries.md`](docs/analyzer-canaries.md) for the
  sharpest version of this problem). Where practical, a new correctness
  test should deliberately perturb the invariant it's protecting (negate
  a score, flip a config flag, corrupt an offset) and assert the test
  *would* have caught it — not just that the happy path passes. Grep
  `break-it-first` for worked examples across the tree.
- **Golden fixtures** live under `testdata/`, produced by the Python
  oracle scripts under `scripts/oracle/` (see
  [`testdata/README.md`](testdata/README.md) and
  [`scripts/README.md`](scripts/README.md)) — Go tests only ever *read*
  them. `scripts/` holds nothing else: every deciding-shell gate lives in
  `tools/` as a Go command, and `tools/scriptsguard` enforces both of
  those boundaries mechanically. It's a separate CI job
  (`scripts-boundary`), not part of `preflight` — run it directly if
  you're adding anything under `scripts/`: `go run -C tools
  ./scriptsguard`.
- **`-tags aikit_checks`** builds `linalg`'s quant-kernel contract checks
  (shape/argument validation the hot kernels otherwise trust silently in
  production builds). Run it when touching `linalg`'s quantized kernels:
  `go test -tags aikit_checks ./linalg/`.

## Where to look next

- [`README.md`](README.md) — package table, stability tiers, versioning.
- [`docs/architecture.md`](docs/architecture.md) — the package DAG, the
  Backend seam, the quarantines, the ADR index.
- [`RELEASING.md`](RELEASING.md) — the full pre-tag ritual.
- [`docs/internal/roadmap.md`](docs/internal/roadmap.md) — what's live,
  what's gated behind a trigger, what's speculative backlog.
