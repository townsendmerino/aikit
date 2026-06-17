# aikit task: an mmap + madvise + span-residency leaf (and page the FlatI8 index with it)

## Context

goinfer built a weight-paging substrate (mmap a read-only `.giw`, then page
weights in/out under a RAM budget via `madvise`) to run a 35B-A3B MoE on ~16 GB.
The *mechanism* is generic and cgo-free; only the *demand signal* (which expert /
layer to fault, when) is goinfer-specific. This task **lifts the generic mechanism
into a new `aikit/mmap` leaf**, dedups the mmap primitive aikit already duplicates,
and — the payoff — **gives `ann.FlatI8Mmap` a paged mode** so aikit can query an
int8 index larger than RAM. goinfer then refactors its pagers onto this leaf
(goinfer side: `goinfer/docs/task-aikit-substrate-extraction.md`).

This is **new public surface ⇒ Experimental tier** (like `FlatI8`, `vision`,
`linalg`). Ship it in the next aikit minor after **v1.8.1**; goinfer bumps its
`require` and refactors only after you tag.

## Why now / why aikit

- The read-only mmap primitive is **already byte-duplicated inside aikit**:
  `ann/mmap_unix.go` and `embed/mmap_unix.go` are identical (`ann`'s own comment
  says it copies rather than import `embed` to avoid an `ann→embed` edge). goinfer
  has a third identical copy. A zero-dep leaf both `ann` and `embed` import kills
  the duplication without the bad edge.
- aikit has **no `madvise` anywhere**, and `ann.LoadFlatI8Mmap` (int8 codes aliased
  from a read-only mapping — the *same substrate* as goinfer's expert weights) has
  `Close()` but **no residency control**: a large embedded index relies entirely on
  the OS page cache, with no prefetch and no RAM-budget cap. The `SpanCache` below
  fixes that.

## Deliverable 1 — the `aikit/mmap` leaf (stdlib-only, no cgo, no aikit imports)

A new package `github.com/townsendmerino/aikit/mmap`. **Leaf invariant: imports
only stdlib** (so `ann` and `embed` can both import it with zero cycle risk). Build
tags `unix` / `!unix`, mirroring the existing loaders; CI must build the `!unix`
fallback (software path) as aikit already does.

- **`MapReadOnly(path string) ([]byte, error)` / `Unmap([]byte) error`** — lift the
  identical `mmapReadOnly`/`munmap` from `ann/mmap_unix.go` (+ `mmap_other.go`).
  Refactor `ann` and `embed` to call these and **delete their local copies**.
- **`Advise(span []byte, willNeed bool) error`** — `MADV_WILLNEED` / `MADV_DONTNEED`
  over a page-aligned span. Port goinfer's `madvise_unix.go` **and its
  `madvise_darwin.go`** — the darwin nuance matters: `MADV_DONTNEED` is firm on
  Linux but weaker on darwin (`MADV_FREE`), so document "firm on Linux, best-effort
  elsewhere." `!unix` is a no-op.
- **`SpanCache`** — the generic core of goinfer's `expertPager` (`moepaging.go`):
  an **LRU of page-aligned spans within a mapping, bounded by a byte budget**.
  `Touch(spans)` faults a member in (WILLNEED) and releases the LRU tail (DONTNEED)
  to stay under budget; tracks resident bytes by summed span length.
  **Demand-signal-agnostic — the caller supplies spans and decides when to Touch.**
  Do NOT bake in MoE / layer / any model logic (that stays in goinfer). This is the
  hard-won contract: goinfer's router-driven and layer-order policies are thin
  wrappers over this.
- **`PageAlignedInterior(raw []byte) []byte`** (round a byte slice up/down to page
  boundaries — goinfer's alignment math) and a small **available-RAM helper**
  (goinfer's `/proc/meminfo` reader + `~half-of-RAM` auto-budget; 0 elsewhere).

## Deliverable 2 — `linalg.WeightMat.MappedSpan(base, end uintptr) []byte`

goinfer's `alignedMappedSpan` reaches into `WeightMat.Int8()/Int4()` to find a
tensor's quantized backing bytes and tests whether they lie inside a `[base,end)`
mapping. Because aikit **owns `WeightMat`**, that extraction belongs in `linalg` as
a method: return the page-aligned interior of the WeightMat's backing bytes *iff*
they alias the given mapping, else nil (heap-backed → skip). The page-rounding it
calls is `mmap.PageAlignedInterior`. (linalg may import the leaf — leaf is stdlib-
only, no cycle.)

## Deliverable 3 — the payoff: page the FlatI8 index

Wire `SpanCache` into `FlatI8` (or a paged sibling loader) so a `LoadFlatI8Mmap`
index can be queried under a RAM budget: the int8 code block is the span set, the
query path `Touch`es the rows/blocks it scores, cold blocks re-fault from the
read-only mapping. Result: **a larger-than-RAM int8 ANN index** — the capability
`FlatI8Mmap` has been missing, delivered by the same mechanism goinfer proved on
35B-A3B experts. Keep the existing non-paged `LoadFlatI8Mmap` behavior the default;
paging is opt-in (a budget arg / option).

## Gates

- **Pure-refactor for `ann`/`embed`:** lifting `mmapReadOnly` must be byte-for-byte
  behavior-preserving — existing `OpenSafetensorsMmap` / `FlatI8Mmap` tests stay
  green **unchanged**.
- **`SpanCache`:** an LRU/eviction unit test (eviction count > 0 when over budget) +
  a **model-free property test**: `MADV_DONTNEED` a span, re-read, assert
  byte-identical, then `MADV_WILLNEED` + read again (port goinfer's always-on
  `TestMadvise_dontneedRefaultsIntact` — this is the correctness keystone: a
  read-only file-backed mapping re-faults identical bytes, so eviction is lossless).
- **FlatI8 paged == resident:** recall@k of a budget-paged index is **identical** to
  the fully-resident index over the same queries, with eviction count > 0 to prove
  the budget fired (paging is lossless — the cap costs faults, never wrong codes).
- **`!unix` builds** (the fallback path), as aikit CI already enforces.

## Non-goals

- No model/LLM logic in the leaf (no MoE, no layers) — `SpanCache` is span+budget
  only; the demand signal stays in goinfer.
- Not Hard-tier — ships Experimental; note it in the README package table.
- Don't change the default `LoadFlatI8Mmap` behavior — paging is additive/opt-in.

## Reference (goinfer source to mirror)

- `goinfer/decoder/mmap_unix.go` + `mmap_other.go` — the `MapReadOnly`/`Unmap` body.
- `goinfer/decoder/madvise_unix.go` + `madvise_darwin.go` + `madvise_other.go` —
  the `Advise` split (mind the darwin `MADV_FREE` nuance).
- `goinfer/decoder/moepaging.go` — `expertPager` is the `SpanCache` to generalize
  (LRU + byte budget + Touch); `alignedMappedSpan` + `availableRAMBytes` /
  `autoWeightBudget` are the helpers to lift (`MappedSpan` → linalg, the rest → leaf).
- aikit: `ann/mmap_unix.go` (the dup to delete), `ann/flat_i8_mmap.go` (the paging
  target — `bq`/`scales`/`mmap` fields, `Close`/`finalizeMmap`).
