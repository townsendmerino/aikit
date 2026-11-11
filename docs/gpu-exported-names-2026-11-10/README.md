# Exported names of the `gpu` module and who uses them (2026-11-10)

Taken after the five retired modules (`gpu/qwencuda`, `gpu/qwenmetal`,
`gpu/visioncuda`, `gpu/visionmetal`, `gpu/webgpu`) were deleted, to record how
much of package `gpu`'s exported surface the four backends that stay
(`gpu/anncuda`, `gpu/annmetal`, `gpu/enccuda`, `gpu/encmetal`) still use.
Nothing was unexported or deleted on the strength of it (owner decision,
2026-11-10): the module is frozen on `gpu/v0.35.x` for fixes, and unexporting
is an API break for anyone pinned to a tag. The `gpu` package doc points here.

## The two lists

- [`darwin-arm64.tsv`](darwin-arm64.tsv): the Metal build's exported names (160).
- [`linux-amd64.tsv`](linux-amd64.tsv): the CUDA build's exported names (176).

One exported name per line, tab-separated:

| Column | Meaning |
|---|---|
| 1 | `USED` if a backend or `examples/gpu-ann` references the name, `FREE` if nothing outside package `gpu` does |
| 2 | the name: `Name` at package level, `Type.Member` for a method, a struct field or an interface method |
| 3 | `func`, `type`, `const`, `var`, `method`, `field` or `ifacemethod` |
| 4 | for `USED`: which module references it; `(tests)` means only from that module's test files |

## The figures

Across the two builds there are 233 distinct exported names.

| | Names |
|---|---|
| Referenced by a backend's non-test code | 54 |
| Referenced only by a backend's tests | 9 |
| Referenced by nothing outside package `gpu` | 170 |

The 170 are 65 methods, 44 constants, 30 fields, 18 functions, 11 types and 2
variables. About 100 are the ViT tower surface the deleted tower backends
used: the fields of `ViT`, the `Kernel*` and tile-size constants, the attention
and grid helpers, `ViTPTX`. The rest are device features goinfer used before it
took its own copy of the device code: on Metal, residency sets, shared events
and autorelease pools, and a set of `Encoder` methods; on CUDA, graphs, events,
mapped host buffers and the quantized GEMV. One of the 170, the type name
`Encoder`, could not sensibly be unexported: nothing spells it, but backend
tests call five of its methods.

No Go file in goinfer, ken or tack imports package `gpu` (checked the same
day; goinfer has a test that forbids it).

## How it was produced

[`analysis.go.txt`](analysis.go.txt) is the program (stdlib only; stored as
`.txt` so it is not part of any package). It type-checks each consumer with
`go/types` and the source importer, once per platform, and records every use
of an object that belongs to package `gpu`, including uses through a selector
(methods and fields). There were no type errors on either platform. Each
`FREE` name was then searched for as text in the consumers; the only hits were
comments and string literals.

```
go build -o gpuapi analysis.go        # after renaming it to .go in a scratch module
GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 GOWORK=off ./gpuapi <repo> gpu/anncuda gpu/annmetal gpu/enccuda gpu/encmetal examples/gpu-ann
GOOS=linux  GOARCH=amd64 CGO_ENABLED=0 GOWORK=off ./gpuapi <repo> gpu/anncuda gpu/annmetal gpu/enccuda gpu/encmetal examples/gpu-ann
```

Run it from a directory outside the repository. Run from the repository root
with `GOFLAGS=-mod=mod`, it adds `require` lines to the root `go.mod`.

## Limits

- **Consumers outside these repositories are not counted.** `gpu` is a
  published module; a build pinned to one of its tags may use any exported name.
- **The package was read as tagged at `gpu/v0.35.0`.** Its non-test Go source on
  `main` was identical to that tag when the lists were made.
- **The package's own tests are in-package** and use many `FREE` names; that
  does not need them exported.
- **A use through an interface declared in a consumer would be missed.** No
  backend declares an interface, so there is none today.
