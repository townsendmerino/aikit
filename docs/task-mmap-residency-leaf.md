# Plan (aikit): an mmap + residency leaf — and page the FlatI8 index with it

> **Status: DEFERRED — written now, executed later.** This is the aikit half of a
> two-repo extraction (the goinfer half:
> `goinfer/docs/task-aikit-substrate-extraction.md`). The code to be lifted is
> goinfer's weight-memory substrate (mmap + madvise + a span-residency cache),
> which **shipped to goinfer days ago and is still settling** as ideas #2/#4 get
> tuned and GGUF Phase 2 lands. Promoting a moving API into aikit — where it
> becomes a public, version-contracted surface — is the expensive thing to get
> wrong. **Fire only when the trigger below is met.**

## Trigger (when to execute)

Execute when **both** hold:

1. **goinfer weight-memory Phase 1 is stable** — `expertPager` / `layerPager` have
   gone one goinfer minor without an API-shape change, and GGUF Phase 2 (#1
   zero-copy GGUF) has either landed or been explicitly shelved (so we know
   whether the span-cache needs to serve heap-backed *and* mapping-backed spans).
2. **aikit has a reuse pull** — concretely, a "FlatI8 index too big for RAM"
   ask (the paid-off payoff in §3), or a third in-tree duplication of the mmap
   primitive lands. Until one of those is real, the duplication is cheaper to
   carry than the wrong abstraction.

Re-read the goinfer pager code at trigger time before designing the API — don't
design it from this doc's snapshot.

## 1. Why aikit, and why now-the-plan

The mmap read-only primitive is **already duplicated inside aikit** — `ann/mmap_unix.go`
and `embed/mmap_unix.go` are byte-for-byte identical, and `ann`'s own comment says
why:

> *"This mirrors embed's mmap loader. ann keeps its own copy rather than importing
> embed — an ann→embed edge would couple the index package to the embedding one."*

So aikit consciously accepted duplication to avoid a bad dependency edge. goinfer's
`decoder/mmap_unix.go` is now a **third** identical copy across the ecosystem. A
zero-dependency **leaf** package is the clean resolution: `ann` and `embed` both
import it (no `ann→embed` edge), and goinfer drops its copy. Three copies → one.

Two things ride in alongside it that aikit **does not have today**: `madvise`
(residency hints — absent ecosystem-wide) and a generic **span-residency cache**
(the demand-signal-agnostic core of goinfer's `expertPager`). Both are generic OS
mechanism with nothing model-specific in them.

## 2. Scope — what lands in aikit

A new leaf, proposed `github.com/townsendmerino/aikit/mmap` (stdlib `syscall`
only, no cgo, `//go:build unix` + `!unix` fallback — exactly aikit's existing
shape, so the pure-Go promise is unaffected):

- **`MapReadOnly(path) ([]byte, error)` / `Unmap([]byte) error`** — lifted from the
  three identical `mmapReadOnly`/`munmap` copies. `ann` and `embed` refactor to
  call it; their local copies delete.
- **`Advise(span []byte, willNeed bool) error`** — `MADV_WILLNEED` / `MADV_DONTNEED`
  over a page-aligned span (goinfer's `madviseBytes`). Firm on Linux, best-effort
  on darwin (`MADV_FREE`) — document it, as goinfer already does.
- **`SpanCache`** — the generic core of `expertPager`: an LRU of page-aligned spans
  within a mapping, bounded by a byte budget, faulting in on `Touch` (WILLNEED) and
  releasing the tail (DONTNEED). **Demand-signal-agnostic** — the caller supplies
  spans and decides when to `Touch`. goinfer's MoE-router and layer-order logic do
  **not** come along.
- **`PageAlignedInterior(start uintptr, raw []byte) []byte`** + a small system
  helper for available-RAM budgeting (goinfer's `availableRAMBytes` /proc/meminfo
  reader + auto-budget). Generic; co-locate.

A companion in **`linalg`** (not the leaf, since it touches the aikit type):
`WeightMat.MappedSpan(base, end uintptr) []byte` — goinfer's `alignedMappedSpan`
reaches into `WeightMat.Int8()/Int4()` to find a tensor's quantized backing bytes
inside a mapping. Because aikit **owns `WeightMat`**, that extraction belongs in
`linalg`; the page-rounding math it calls belongs in the leaf.

## 3. The payoff that justifies the lift: page the FlatI8 index

This is the part that makes the extraction *aikit's* win, not just goinfer
hygiene. `ann.LoadFlatI8Mmap` already aliases int8 codes from a read-only mapping —
**the identical substrate as goinfer's expert weights** — but today it has only
`Close()`: **no residency control at all.** A large embedded index relies purely on
the OS page cache, with no prefetch and no RAM-budget cap.

Wire `SpanCache` into FlatI8 (or a paged sibling loader) and aikit gains a
**larger-than-RAM int8 ANN index**: query an index whose codes exceed RAM, paged
under a budget, the cold rows re-faulting from the read-only mapping. FlatI8Mmap is
aikit's flagship "embed a huge index" feature; this is the capability it's been
missing, delivered by the same mechanism goinfer already proved on 35B-A3B experts.

## 4. Stability tier + release coordination

- **New public surface ⇒ Experimental.** The leaf and `WeightMat.MappedSpan` ship
  *outside* the v1.0 Hard-tier guarantee (like `FlatI8`, `vision`, `linalg`
  itself) — settling, may change in a minor. Say so in the README package table.
- **Leaf invariant:** stdlib-only, no aikit imports, no cgo — so `ann` and `embed`
  importing it can never create a cycle. CI builds the `!unix` fallback (the
  software path) as aikit already does.
- **Release:** the leaf ships in an aikit minor (call it the next after v1.7.3).
  goinfer bumps its `require` to that version and refactors **after** it's tagged —
  never against an untagged aikit (the goinfer plan gates on this).

## 5. Non-goals (stay out of aikit)

- The transformer-specific pagers themselves (router demand signal, layer-order
  prefetch, `LayerWeights`/`expertWeights`) — those stay in goinfer; only the
  `SpanCache` substrate they sit on moves.
- The `.giw` weight format / goinfer's `serialize.go` payload. **Soft watch only:**
  the magic/version/CRC/lazy-fallback *envelope* is triplicated (aikit `ann` persist
  + `format_errors.go`, goinfer's `serialize.go` "mirrors ken's index_serialize.go",
  ken). A shared `blob` envelope could de-dup the scaffolding — but the payloads
  genuinely differ and aikit already has `ErrFormat`, so it's lower value than the
  mmap/madvise/span work. Don't bundle it into this lift; note it and move on.

## 6. Gates

- **Pure-refactor for ann/embed:** lifting `mmapReadOnly` into the leaf must be
  byte-for-byte behavior-preserving — existing `OpenSafetensorsMmap` / `FlatI8Mmap`
  tests stay green unchanged.
- **`SpanCache` correctness:** an LRU/eviction unit test + a model-free "DONTNEED
  then re-read is byte-identical" property test (goinfer already has the latter as
  `TestMadvise_dontneedRefaultsIntact` — bring it down with the code).
- **FlatI8 paged == resident:** recall@k of a budget-paged FlatI8 index is
  **identical** to the fully-resident index over the same queries (paging is
  lossless — the cap costs faults, never wrong codes), with eviction count > 0 to
  prove the budget fired.
