# AX4: deprecate and retire what moved to goinfer — 2026-10

Track AX's aikit side (goinfer's `docs/tasks/task-subsystems-2026-10.md` §7.3, AX4, and AX5; owner decisions SD1,
SD2 and SD5, 2026-10-26; Francis, 17:38: "AX4, go"). The moved code is goinfer's since AX1 and goinfer stopped
importing aikit's copies at AX2 and AX3 (goinfer PRs #274, #278, #282 to #284, #289).

## What changes

| path | change |
|---|---|
| `vision` (Hard tier) | a `Deprecated:` package note pointing to goinfer; frozen; removed only at a v2 |
| `audio` (Experimental) | a `Deprecated:` package note; removed in the minor after the one that first ships the note |
| `gpu/qwencuda`, `gpu/qwenmetal`, `gpu/visioncuda`, `gpu/visionmetal`, `gpu/webgpu` | a retirement note in each `go.mod`; no further tags |
| `gpu` (the device module) | stays, frozen on `gpu/v0.35.x` (fixes only) for `gpu/anncuda`, `gpu/annmetal`, `gpu/enccuda` and `gpu/encmetal` (SD2) |
| `tools/gpupins` | judges only the four staying backends, against the latest `gpu/v0.35.x` tag, and stops judging their root `aikit` pin (AX5) |
| `.golangci.yml` | SA1019 excluded for the retired backends and `examples/vision`, which import `vision` and stay as they are |
| `README.md`, `RELEASING.md`, `CHANGELOG.md` | the tiers, the release steps (no `gpupins --fix` after a root tag) and an Unreleased entry |

No tag is pushed by this branch. A tag is a separate step that waits for Francis (DECISION NEEDED with the exact name
and notes).

## Gates (registered 2026-10-26 17:43 PDT by mac-ax01, before AX4's code)

Tier 0; the Mac by day.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| AX4-L (aikit's preflight, CI's root job) | `go run -C tools ./preflight` reads green: gofmt, build, vet, the pinned golangci-lint (SA1019 now live for the deprecations), cgo-free build, tests. A missing linter is INCONCLUSIVE, not green | — | complete | about a minute (RELEASING.md) |
| AX4-G (aikit's GPU mechanical gate) | `go run -C tools ./gpugate` reads green on the Mac across the gpu modules (PTX n/a on darwin, as documented) | — | complete | minutes |
| AX4-P (the pin gate) | `go run -C tools ./gpupins` reads green with the four staying backends on the latest `gpu/v0.35.x`; planted once, a staying backend's `aikit/gpu` require set to `v0.34.0` reads red, and a retired backend's drift is not judged; the tool's own tests pass | — | complete | seconds |
| AX4-D (the note reaches a consumer) | a scratch module importing `aikit/vision` and `aikit/audio` from this branch (a `replace`), linted with the pinned golangci-lint, reports SA1019 naming each package as deprecated | — | once | seconds |
| AX4-C (the downstream-consumer gate) | `ken` and `tack` at their current main, each in a scratch workspace with this branch's aikit, build, vet and run their unit tests green, as they do against their pinned aikit; neither imports a deprecated package | a repo's own baseline is red at its pinned aikit (listed, compared by name) | once each | minutes |
| AX4-R (aikit's release gate) | at the tag step only, after Francis's decision: `go run -C tools ./releasegate X.Y.Z`, the perfgate evidence line and the vulncheck statement, as RELEASING.md requires | — | — | per RELEASING.md |
