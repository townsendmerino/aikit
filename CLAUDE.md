# aikit — agent notes

Rules here are true for any contributor. A maintainer's machine names, home-directory paths and session procedures live
in a gitignored `CLAUDE.local.md` that Claude Code loads beside this file; nothing here depends on it.

**`CONTRIBUTING.md` is the authority for day-to-day work** and is loaded with this file:

@CONTRIBUTING.md

It covers `GOWORK=off`, the module layout, `preflight` and the `VERDICT:` line, the API tiers, the code and test
conventions, and the scripts boundary. Read `docs/architecture.md` before changing a package boundary, and `RELEASING.md`
before any tag; do not restate their version numbers. What follows is what `CONTRIBUTING.md` does not say.

## Changes reach `main` through a pull request

- **Every change lands through a pull request; nobody pushes to `main`.** `main` is protected by CI's required checks.
  Push your branch, open the PR, and at once, with checks still pending, run `gh pr merge <n> --repo townsendmerino/aikit
  --auto --merge`. GitHub merges it when the required checks pass. Do not hold a turn open on CI or poll it; read the
  result at the next natural point and report a red the moment you see it.
- **A flake gets one rerun; nothing merges past a red that is not understood.**
- **`main` moves under you.** `git fetch` and merge `origin/main` into your branch before pushing. No rebase and no
  force-push, because a cited commit id must stay valid. Changes you did not make in a file you are editing are someone
  else's in-flight work: commit your hunks, leave theirs.
- **Never `git add -A` or `git add .`.** Stage explicit paths.
- **If a hook holds a push you believe is wrong, stop and report it. Never `--no-verify`.**
- **Run the linter only through `preflight`.** It runs the pinned golangci-lint behind a canary. An empty lint result is
  indistinguishable from one that never ran, and a linter built by another Go version drifts.

## The consumers

aikit exists to serve goinfer and ken. Breaking an Experimental-tier API is acceptable when it makes the API clearer; its
consumers update in their own pull requests. A Hard-tier break goes through `tools/releasegate`'s `apidiff` check as
`CONTRIBUTING.md` describes. A kernel or numeric change that goinfer depends on for bit-identity says so in the PR, and
names the goinfer gate that checks it.

## Public text

Public docs, comments, commit messages and PR bodies name no session, no coordinator and no machine nickname; describe a
box by its hardware ("an Apple M1 Pro, 16 GB"). Record a decision as what was decided, with the date and the reason in a
sentence, not as a quoted owner message. Commit messages and PR bodies end with the `Co-Authored-By` line; no session URL
line. Existing text is not rewritten for this rule.

## Code comments

A Go comment says what is true now: what the code does, its contract (ownership, units, concurrency, what a caller must
check), the reason in a sentence where the shape is not obvious, and every guardrail, with a pointer to its evidence. It
does not carry the history: how a bug was found, dates, commit ids, sessions and measured figures go to the commit message
or the record under `docs/`. A pointer to a finding is a short tag in the form `CONTRIBUTING.md` gives ("Numbered
citations in comments"). Cite code by declaration name, never `file.go:NNN`. Do not rewrite comments in a file you are
only passing through.

## Tests

- **A skip is not a pass.** `go test` prints `ok` when every test skipped; confirm with `-v` and read for `--- PASS`.
- **A unit test that supplies its own calling convention proves nothing about the callers.** When a contract depends on
  how often or under what condition a component is called, test it through its caller.
- **A doc comment claiming coverage is not coverage:** the test body must contain an assertion naming the thing claimed.
- **A minimal repro can be minimal in exactly the dimension that hides the bug:** ask what is identically zero at that
  size (one row, one head, K below a tile's width).
- **`go test -v ./a/ ./b/` prints nothing until `./a/` finishes:** one package per invocation when you need to watch.
- **An assembly kernel is checked on every ISA it targets.** A kernel's bit-identity gate runs under emulation where the
  host lacks the feature (`qemu-user` for arm64 without DotProd, the forced-fallback CI jobs for amd64); CI's job list in
  `.github/workflows/ci.yml` names them.

## Rigor matched to stakes

Maintainer decision, 2026-10-30: match the process to what the result decides.
- **Full measurement discipline** (pre-registration, registered gates, stopping rules, a measurement record) only when a
  number will be published, or will settle a direction: adopt or kill a kernel, change a default or a dispatch threshold.
- **Ordinary engineering** (refactors, moves, tooling, bug fixes, docs): a red-first test where behavior changes,
  `preflight` passing, and a PR. No registration section and no measurement record.
- **A registration is short:** about one screen (question, metric, threshold, stopping rule, what each outcome means).
- **When unsure which applies, take the lighter one and say so in the PR.**

## Timed runs

- **`docs/internal/measuring-performance.md` is the method,** and `docs/internal/perf-dead-ends.md` is read before
  proposing a kernel idea: it records what was tried and why it lost.
- **One timed run per box, by day, holding that machine's timing lock** (goinfer's `scripts/timing_lock.py run`, when the
  box has a goinfer checkout), with the expected wall time stated first. Night is only for a job over about two hours.
- **Comparisons are interleaved** A B B A on the same input in one session, and matched observations are differenced,
  never pooled. A cell started under heavy load is void, not averaged.
- **A bit-identical speed change ships** when its outputs are bit-identical, its memory is not higher, it is faster on at
  least 9 of 10 timed pairs, and it replaces code rather than adding a second path.
- **Model files for timed runs are read from fast local disk on the machine doing the timing,** never from an archive
  or a network share.

## Task docs

New task docs, queues and plans for aikit live in the maintainer's private notes checkout (a sibling `../goinfer-notes`,
`tasks/`), in goinfer's ledger format: `<!-- ledger -->`, then `| id | state | next | evidence |`, one row per item. A
commit subject names the item by the ID its doc owns. The task docs already under `docs/` stay where they are.

## Long-running work

Anything over a few minutes is detached (`setsid nohup … </dev/null &` on Linux; `launchctl submit` on macOS), with its
log somewhere durable, never `/tmp`. Prefer committed increments to one big commit.
