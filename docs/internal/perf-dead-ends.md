# Perf dead ends — the things that were tried and did NOT ship

The single most reusable artifact of the 2026-07 perf campaign. Every entry here is
a change that was **built and measured** (or reasoned out against a hard ceiling) and
did not earn its place. A dead end is only useful if the **mechanism** and the
**number** are here — a bare "tried X, didn't help" gets re-tried, so those are
recorded in full. Grouped by root cause: several share one, and the root cause is the
transferable lesson.

**Machine labels are load-bearing** (see measuring-performance §1.35–36). Two boxes:
- **`nvidia-rtx2070s`** — Ryzen 7 3700X, Zen 2, amd64, AVX2 (no VNNI), Linux. The
  Phase A/B / lens-scan box.
- **`apple-m1pro`** — M1 Pro, arm64, 6 P + 2 E cores, no SMT, macOS. The arbiter of
  record and the representative laptop.

A number without a box is a bug; where a source did not name the box it is flagged as
inferred (from the kernel named — `dotFMA8`=amd64, `dotNEON2x8`=arm64 — or the date
cluster; the 2026-07-30/31 work is the `apple-m1pro` arbiter phase).

> **Recovering an unattributed box.** When a commit's prose does not name the machine,
> two signals recover it, and across this campaign they agree 10/10:
> 1. **The `Co-Authored-By` trailer is a machine fingerprint.** The two boxes ran
>    different configured models — **Claude Opus 5** on `nvidia-rtx2070s`, **Claude
>    Opus 4.8** on `apple-m1pro` — so the trailer separates them cleanly
>    (`git log -1 --format=%b <sha> | grep Co-Authored-By`). *Caveat:* it records the
>    session's configured model, not the machine directly, so it stops separating if
>    the same model is ever run on both boxes; it held for the whole 2026-07 campaign.
> 2. **Third-person references to the OTHER box.** A commit that says "the amd64 box's
>    instinct… was right" while reporting its own build-measure-revert was written on
>    the arm64 box — you don't write that about yourself.
> Worked example: `0269de1` (MaxScore, §5.2) carries the Opus 4.8 trailer AND says "The
> amd64 box's instinct to leave it undone was right" → `apple-m1pro`, both signals
> agreeing. Use the same method on any other entry marked "inferred."

> **Read this first — a negative needs its numbers as fully as a positive.** Two
> entries below (HNSW heap pooling, §8.3; item 24 packed-stride pad, §6.1) were
> recorded as dead ends and then **measured the other way** on a different box or with
> a better method. Both were originally filed without a rigorous in-process A/B. That
> is the meta-lesson of this whole file: *a dead end recorded without its measurement
> is a landmine, not a signpost* — it either gets re-chased or, worse, blocks a real
> win. When you add to this file, write the mechanism and the number, name the box,
> and prefer an in-process A/B to a remembered impression.

---

## Group 1 — darwin `MADV_DONTNEED` is a no-op for RSS: every madvise-reclaim lever is inert on macOS

The single sharpest OS-transfer finding. macOS does not reclaim clean file-backed
pages on `madvise`, so anything whose win is "drop resident pages" or "hide a fault
that only exists because a page was dropped" does nothing on `apple-m1pro`. See
measuring-performance §1.35.

### 1.1 · Item 9 `Touch(b+1)` paging prefetch — **+12%**, `apple-m1pro`, 2026-07-30 (`f6e6a33`)
- **Tried:** a state-free `SpanCache.Prefetch` issuing a one-block-ahead
  `MADV_WILLNEED` in `scorePaged`, hinting block b+1 while block b is being scored.
- **Mechanism:** overlap b+1's page fault with b's int8 scoring compute, hiding the
  fault latency.
- **Number / box:** **+12% SLOWER**, base 1.95 ms/query → prefetch 2.18 ms, budget
  64/1875 blocks, on `apple-m1pro`.
- **Why:** on darwin `MADV_DONTNEED` is a no-op for RSS, so "evicted" blocks stay
  resident and every prefetch `WILLNEED` is a syscall on an already-resident page —
  pure overhead, no fault to hide. The win is Linux-only and **unmeasurable on the
  only box available**; even granting Linux, one block (~24K int8 MACs, ~µs) is too
  little compute to cover a ~10 µs fault without a multi-block-ahead window.

### 1.2 · §4.5 peak-RSS release, re-measured — **726.2 MiB, no change**, `apple-m1pro`, Phase D (`cc1133b`)
- **Tried:** the arbiter pass on the `nvidia-rtx2070s`-shipped `ReleaseTensors` (one
  `MADV_DONTNEED` per f32 tensor after quantizing it; `LoadWeightsQ8` peak RSS
  727.6 → 242.3 MiB, 3.00× on amd64).
- **Mechanism:** drop each f32 checkpoint tensor's pages as soon as it is quantized,
  cutting peak resident set.
- **Number / box:** **726.2 MiB on `apple-m1pro`** — lands on the amd64 *unreleased*
  727.6 MiB, not the released 242.3. No win at all.
- **Why:** same as 1.1 — `ReleaseTensors`→`mmap.Advise(DONTNEED)` is a documented
  no-op on darwin, so the 547 MB f32 checkpoint stays resident until `Close`. The
  peak-RSS number the footprint story quotes is a **Linux-only artifact**. Correctness
  is identical everywhere (released pages re-fault identical bytes); only the footprint
  drop is OS-specific. *This is not a revert (the Linux code shipped) — it is an
  arbiter finding that the headline does not transfer.*

### 1.3 · `mmap` eager fault-in (`Advise(WILLNEED)` on map) — **not attempted / unmeasurable**, `apple-m1pro`
- **Tried:** nothing built — advise `WILLNEED` on the whole mapping so a
  larger-than-RAM index faults in eagerly.
- **Mechanism:** the same overlap-the-fault idea, index-wide.
- **Number / box:** none. Declared unmeasurable: exhibiting the win needs a genuinely
  larger-than-RAM index (measuring-performance §1.23), which `apple-m1pro` cannot show
  for the same DONTNEED-no-op reason. Stated "could not measure" rather than shipped
  blind.

---

## Group 2 — `dotNEON2x8` is ~95% of FMLA peak on `apple-m1pro`: the arm64 f32 kernel is compute-bound, so every load-reduction lever is dead

On the M1 P-core the f32 micro-kernel runs at its arithmetic ceiling (~51 GMAC/s,
~95% of single-core FMLA peak). On `nvidia-rtx2070s` the same-shaped kernel
(`dotFMA8`) sits at ~40% of peak, i.e. load/latency-bound — which is why the amd64
analysis kept predicting load-reduction wins that do not exist on arm64.

### 2.1 · Item 37 — outer-product f32 microkernel — **1.10× raw / ~4× slower full path**, `apple-m1pro` (`5aeab4e`)
- **Tried:** transpose-pack both operands into K×8 panels + an 8×8 by-element-`FMLA`
  tile (raw-`WORD`-encoded, bit-gated vs scalar oracle, mutation-checked).
- **Mechanism:** the outer product does 4.0 FMLA per load vs `dotNEON2x8`'s 1.6, so on
  a load-bound kernel it should be ~2× (the amd64-shaped ~1.45× prediction).
- **Number / box:** **raw kernel only 1.10×** (46.1 vs 41.7 GMAC/s) on `apple-m1pro`;
  the mandatory transpose-pack of both a and b made the **full serial path ~4× slower**.
- **Why:** `dotNEON2x8` already runs at ~95% of FMLA peak here — compute-bound, so
  cutting loads 10→4 buys almost nothing, and it is `≠` (a different f32 order) so it
  would perturb every encoder golden for a loss.

### 2.2 · Item 25 — arm64 `Dot2x8` 4×4 re-tile — **likely dead (not separately built)**, `apple-m1pro`
- **Tried:** re-tile to a 4×4 register block (8 loads per 16 FMLAs vs today's 10).
  Measured-out *by inheritance* from 37, not separately built.
- **Mechanism:** fewer loads per FMLA → predicted 1.1–1.25× arm64 f32 GEMM.
- **Number / box:** same 95%-FMLA-peak cause as 2.1 — cutting loads 10→8 cannot help a
  compute-bound kernel. Same finding as 37.

### 2.3 · Pre-packing weights to kill `packedFill`'s per-forward b-copy — **~0–3%, noise**, `apple-m1pro` (`7197c66`)
- **Tried:** `PackWeightBT` + `MatmulBTPacked` — pack the immutable weights once at
  load instead of `packedFill` rebuilding the conflict-free b-layout every forward
  (bit-identical, mutation-checked).
- **Mechanism:** `packedFill` profiles at **13%** of a GTE forward (`runtime.memmove`,
  98% from `packedFill`); the weights are immutable so the copy should be removable.
- **Number / box:** **~0–3%, within noise** (fc2/down, M=91–690) on `apple-m1pro`.
- **Why:** the 13% is pprof *self-time* the copy spends **overlapping** the
  compute-bound `Dot2x8` of adjacent tiles on an out-of-order core — not wall-clock you
  can remove. With `dotNEON2x8` at ~95% FMLA peak, the f32 encoder is at its arm64
  floor here.

### 2.4 · Lens §4.2 gate column-blocking in `swigluMLP` — **+3.5–6.9%**, `apple-m1pro`, 2026-07-31 (`6a8a368`)
- **Tried:** column-block the gate into `val[:, j0:j1]`, dropping the full `[L,I]` gate
  buffer to one jb-wide tile (bit-identical, `TestSwigluMLP_colBlockBitIdentical`,
  mutation-checked).
- **Mechanism:** on `nvidia-rtx2070s` this was **latency-neutral** (2126→2123 ms) while
  cutting gate scratch 44.0 → 3.67 MB/worker — a "free" footprint win expected to
  transfer.
- **Number / box:** **+3.5–6.9%** on isolated `swigluMLP` L=3584 (+5.6% jb=256 / +3.5%
  jb=512 best / +6.9% jb=768 / +5.4% jb=1024, min-of-8) on `apple-m1pro`; full batch
  forward +0–6%, too noisy to resolve.
- **Why:** the amd64 "free" reading is spent by the **6P+2E fork/join** — the full gate
  is one parallel matmul, the blocked gate is I/jb of them, each barrier paying the
  E-core straggler tax. Same latency-for-footprint tradeoff §4.2 rejected the row-tile
  variant for, so by the lens's own bar it does not ship here. **`nvidia-rtx2070s` may
  still ship it.** Its subsumed sibling — GTE's `upGate` [L,2I], campaign #8 — was not
  attempted once the base transform measured out (same fork/join structure).

---

## Group 3 — the pool / gather is memory- and latency-bound, not compute-bound: no addressable compute lever

### 3.1 · StaticModel word→vector presum (memoization §5) — **wash serial / +30.9% batch**, `apple-m1pro`, 2026-07-31 (`57a93e4`)
- **Tried:** a sharded, arena-backed `word → (f64 presum over subwords, wsum)` cache +
  `eachWord` carve-out mirror + opt-in `EnablePresumCache`, to skip both `wordPiece`
  and the per-subword gather.
- **Mechanism:** 98% of words repeat and `encodeIDs` (the f64 pooling gather) is 49% of
  `Encode` on `apple-m1pro`; caching the pooled sum should collapse the repeated-word
  gather (predicted ~29% pool collapse ≈ 14% of Encode).
- **Number / box:** **serial +0.4% (wash); batch +30.9% SLOWER** under EncodeBatch (8
  workers), on `apple-m1pro`. Bit-exact (0/386 docs).
- **Why:** serial — the per-word cache machinery (FNV hash + sharded `RLock` + map
  probe + arena view) costs as much as the ~0.4 subword-gathers it collapses; the f64
  gather is too cheap to beat with a keyed lookup. Batch — the shared cache's RWMutex
  reader-count atomics ping-pong across cores on the hot words.

### 3.2 · Vectorised f64 gather (the fallback to 3.1) — **memory-bound, no win**, `apple-m1pro`
- **Tried:** SIMD the pooling MAC loop (`sum[j] += f64(row[j])·ww`).
- **Mechanism:** if the pool were compute-throughput bound, a wider MAC would help.
- **Number / box:** **0.7 ns/MAC warm** (4-wide unroll buys only +7%, f32-accumulate
  +5%); **1.42 ns/MAC on the real corpus** (2× warm — cold gathers of the 63 MB table),
  on `apple-m1pro`. NEON f64x2 is only 2-wide → single digits at best.
- **Why:** the warm rate already isn't compute-throughput bound, and the real-corpus
  gap is cold row gathers — memory-bound, which no MAC kernel fixes. The pool is at its
  floor on this box.

---

## Group 4 — the scalar arithmetic is hidden behind the SIMD dot: rank-invariance / selection hoists measure at zero

### 4.1 · Both rank-invariance hoists (lens 2) — **0% at three dimensions each**, `nvidia-rtx2070s`
- **Tried:** hoist the loop-invariant scalar out of the top-K selection in two places —
  `ann/hnsw.go:236` `sim()`'s `qv.scale`, and `linalg/quant.go:225` via
  `ann/flat_i8.go:125` the W8A8 query scale.
- **Mechanism:** the scalar is rank-invariant (monotone), consumed only by a top-K
  comparison, so it can be hoisted past selection.
- **Number / box:** **no win at any dimension** on `nvidia-rtx2070s`. hnsw.sim:
  212533/177480/191400 vs 202560/212670/219227 ns. w8a8: dim 64 = 333,840 vs 341,158;
  dim 384 = 1,095,497 vs 1,110,221; dim 768 = 2,093,873 vs 2,148,820 ns. (An earlier
  count=3 run showed 3.5% — noise; use min-of-N.)
- **Why:** the scan is bound by the int8/SIMD dot, not the scalar around it — one
  multiply per `sim()` is entirely hidden behind a 384-element SIMD dot. Worse, the
  w8a8 hoist is **not** bit-identical: a search over 60 M near-tied tuples found **1,824
  order inversions (3×10⁻⁵)**. Recommendation was: do neither.

### 4.2 · `topk.Push` bounds check — **exactly zero**, `nvidia-rtx2070s`
- **Tried:** eliminate the bounds check in the reject path.
- **Mechanism:** the check runs once per rejected candidate; removing it saves the
  reject-path overhead.
- **Number / box:** **exactly zero** on `nvidia-rtx2070s` — it runs once per rejected
  candidate, i.e. noise.

### 4.3 · Items 10 / 29 bm25 scoring wins — **~0% scoring**, `nvidia-rtx2070s`
- **Tried:** item 10 (precompute per-posting BM25 impact / `1/avgdl` hoist) and item 29
  (16 B → 8 B posting) to speed scoring.
- **Mechanism:** cheaper per-posting scalar arithmetic → predicted 1.5–2× scoring.
- **Number / box:** **~0% scoring** for both on `nvidia-rtx2070s`. (29 still delivered
  −50% index memory, 381.7 → 190.9 MB; 10 delivered nothing on time.)
- **Why — this is measuring-performance §1.18 in the flesh:** item 11 (the touched-set
  accumulator, O(postings touched) instead of O(corpus)) had **already taken the scan
  off the critical path**, so items optimizing the scan optimized something that had
  stopped being the bottleneck. An earlier item spent a later one's win.

### 4.4 · Softmax-scale fusion (lens 3) — **16.0 vs 16.0 ns/elem, worth nothing today**, `nvidia-rtx2070s` (inferred)
- **Tried:** fuse the `scores[i] *= scale` pass into `softmaxRow`.
- **Mechanism:** eliminate a separate O(L²) scaling pass; provably bit-identical.
- **Number / box:** **16.0 vs 16.0 ns/elem at L=80** — `math.Exp` at ~16 ns/elem swamps
  the ~0.3 ns/elem pass. **Revisit after campaign #13 lands SIMD `expF32`** — then it
  becomes ~20% of the softmax instead of 2%. (Conditionally-open, not permanently dead.)

### 4.5 · `scanFlat`'s per-vector `emit` closure (lens 4) — **1.03×**, `nvidia-rtx2070s` (inferred)
- **Tried:** de-closure the per-vector emit callback in `scanFlat`.
- **Number / box:** **1024 → 991 ns/vec, 1.03×** — leave it.

---

## Group 5 — the algorithm needs a shape the package never sees: dynamic-pruning retrieval

### 5.1 · Sparse WAND (item 39, `sparse` half) — **4.93× on 3 terms / 2.4× SLOWER at 30**, `nvidia-rtx2070s` (§7.37)
- **Tried:** port bm25's WAND dynamic pruning to `sparse` (SPLADE) Query.
- **Mechanism:** keep posting cursors ordered by doc, accumulate per-term upper bounds,
  skip any doc whose bound cannot beat the k-th best.
- **Number / box:** **4.93× on a 3-term mixed query**, but a SPLADE expansion emits
  20–40 terms and at 30 it ran **2.4× SLOWER** than the exhaustive scan, on
  `nvidia-rtx2070s`.
- **Why:** behind a length guard the package's only realistic shape (20–40 terms) takes
  the exhaustive path unchanged — the code earns nothing and still must be maintained.
  This is item 37's pattern on the retrieval side: the mechanism was real, the shape it
  needed was not the shape the package sees.

### 5.2 · MaxScore long-query pruning (item 39) — **2.5–5.7× SLOWER**, `apple-m1pro` (2026-07-31, `0269de1`)
- **Tried:** MaxScore (partition terms into essential/non-essential, never advance
  non-essential cursors) for the long-query (>8-term) case WAND cannot serve.
- **Mechanism:** past `maxWandTerms = 8`, MaxScore should beat the exhaustive scan where
  WAND's per-iteration linear cost defeats it.
- **Number / box:** **2.5–5.7× SLOWER** (exact vs exhaustive over 1000 queries,
  mutation-checked), on **`apple-m1pro`**. (The source doc did not name the box; it was
  recovered from two independent signals — see "Recovering an unattributed box" below.
  The bm25-lineage inference to `nvidia-rtx2070s` was wrong.)
- **Why:** the exhaustive TAAT accumulator (postings once into `scores[]`) beats DAAT
  pruning when selectivity is uniform; MaxScore needs skewed impacts (SPLADE weights),
  which nothing here measures.
- *(Sub-finding, recorded:* WAND's first implementation was SLOWER while doing 21× less
  work — 680 documents evaluated against 14,228 scored, still −19% — until caching the
  current document on the cursor and writing out the bisection turned −19% into the
  shipped +3.88×.)*

---

## Group 6 — dead code / unreachable on the measuring box (a flat sweep is itself a signal)

### 6.1 · Item 24 packed-stride pad — **unreachable on `nvidia-rtx2070s`** (`81a8bd6`) — *shipped later on `apple-m1pro`*
- **Tried:** a padded-stride `packedFill` (pad ∈ {0,4,16}) to break the 4096-byte
  power-of-two conflict stride.
- **Mechanism:** the 8 packed b-rows sit exactly 4096 B apart (a power-of-two conflict
  stride); a pad moves them to distinct L1 sets.
- **Number / box:** **the sweep measured nothing** on `nvidia-rtx2070s` — every result
  inside noise *because none of it executed*: `packedFill` is gated on `has2x8Kernel`,
  false off arm64.
- **DEAD-END-THAT-WASN'T:** the same change **shipped positively on `apple-m1pro`**
  (`67da4a2`): −9.8%/−10.7%/−8.8% on large-encoder fc2. File the amd64 result as
  "unreachable here," not "does not work." The lesson (measuring-performance §1.1): a
  uniformly-flat sweep is a signal the code did not run, not that it did nothing.

### 6.2 · Item 31 (QKV transpose) & item 23 (packedFill m-blocking) — re-profiled out, not implemented, `nvidia-rtx2070s` (inferred)
- **Item 31:** QKV split + per-head V transpose, "3.3× on the transpose." Measured, the
  three copies total **80 ms of 126.9 s of samples, 0.06%** — invisible, not worth
  doing.
- **Item 23:** `packedFill` lost `blockedFill`'s m-blocking (a-panel re-read per
  8-column group). **DEFERRED WITH MEASUREMENT, not dead:** serial `packedFill` runs at
  75–81% of the ~42 GMAC/s kernel peak, so the a-re-read is one bounded contributor to a
  ~20% gap it shares with the compute-overlapping b-copy and the reduce. The m-blocking
  fix trades a-reads for redundant b-re-packing — a wash-risk core-GEMM restructure for
  a sub-gap win. *This is an open item of type (b), deferred-with-measurement, not a dead
  end — the row-parallel dispatch in item 22(b) sidesteps it for the Q8 path.*

### 6.3 · Item 26 — `math.Round` not an amd64 intrinsic (premise true, conclusion false) — **wash**, `nvidia-rtx2070s` (`8552e73`)
- **Tried:** replace `math.Round` with `math.Trunc(x + math.Copysign(0.5, x))` (which
  emits `ROUNDSD`).
- **Mechanism:** `math.Round` inlines to an integer-op sequence with no `ROUNDSD`; the
  Trunc form emits the hardware round.
- **Number / box:** **a wash** — `math.Round` 1.56–3.07 ns/elem vs `Trunc+Copysign`
  1.92–2.73 (overlapping, Round's best faster), on `nvidia-rtx2070s`. `Copysign` inlines
  to its own `MOVQ`/`ANDQ`/`ORQ` sequence, which costs about what Round's extra integer
  ops cost. And the target is far smaller than estimated: `math.Round` is **0.61% flat**
  of the W8A8 matmul (`dotI8AVX2` is 84.36%) → ~3% ceiling anyway.

---

## Group 7 — the pass split / overlap wins nothing at repo scale: working-set thrash or goroutine tax exceeds the overlap

### 7.1 · Pass fusion in the index loop (lens N8) — **5.2% SLOWER**, `nvidia-rtx2070s`
- **Tried:** fuse the bm25-tokenize and embed passes in the index loop
  (`examples/rag/main.go`).
- **Mechanism:** one pass over the corpus instead of two should cut redundant traversal.
- **Number / box:** **5.2% slower** (593.2 vs 564.1 ms W1) on `nvidia-rtx2070s`.
- **Why:** the combined working set (bm25's pooled buffers + the WordPiece vocab map +
  the embedding table) evicts more than the split passes cost.

### 7.2 · Query-side retrieval overlap (lens N8) — **21% SLOWER**, `nvidia-rtx2070s`
- **Tried:** overlap `ann.Query` with `bm25.TopK` (independent, 81 µs of 127).
- **Mechanism:** run the two independent retrieval stages concurrently to hide one.
- **Number / box:** **21% slower** (147.5 vs 121.7 µs) on `nvidia-rtx2070s` — goroutine
  spawn/join costs more than the 15 µs it hides. May flip at ~10× corpus; does not flip
  at repo scale.

---

## Group 8 — standalone negatives

### 8.1 · Spin-park worker pool — **built, measured end-to-end by goinfer, pulled** (`902cbec`)
- **Tried:** a spin-park worker pool (warm workers to fix dispatch wake latency).
- **Mechanism:** keep workers spinning/parked so dispatch avoids goroutine wake latency.
- **Number / box:** no aikit-side number — measured end-to-end by goinfer, which said
  no; the pool was built correctly, measured honestly, and pulled. (linalg item 11,
  dynamic work-stealing over column chunks for E-core straggler imbalance, is a
  *different* thing; the bar for re-entering this area stays high.)

### 8.2 · Regex union — **9× SLOWER**, `nvidia-rtx2070s` (inferred)
- **Tried:** compile the N chunker regex patterns into one alternation so a single
  machine acquisition covers all of them.
- **Mechanism:** one regex acquisition/scan instead of N.
- **Number / box:** **9× slower** (Go 2,043 ns/line, TS 5,643 ns/line) — the union
  defeats onepass and literal-prefix optimizations. Prescreen, don't union.

### 8.3 · Pooling HNSW's search heaps *alone* — **originally slightly slower; OVERTURNED to −27.1%** (`8460bbc`)
- **Tried:** pool `searchLayer`'s two heaps.
- **Mechanism:** cut per-call allocations (GC pressure).
- **Original number / box:** **slightly slower** — 519 vs 475 µs at ef=64 (allocs
  19–22/query → 1), on `nvidia-rtx2070s` (inferred, original §6 reading).
- **DEAD-END-THAT-WASN'T:** rebuilt with an **in-process A/B** it became **−27.1% time
  and 19 → 2 allocs/query** (item 15b, Phase D, `nvidia-rtx2070s`). The campaign flags
  this as the cautionary tale — *a negative result needs its numbers written down as
  fully as a positive one.* The original "slightly slower" had no rigorous A/B behind
  it; the measured redo flipped the sign.

### 8.4 · `layerNorm` vectorisation — **no bit-identical variant, ~0.5% of a forward**, `nvidia-rtx2070s` (inferred)
- **Tried:** a fused-moment f32 `layerNorm`.
- **Mechanism:** a SIMD reduction runs at 1.42 ns/elem vs the scalar 2.98 ns/elem (2.1×).
- **Number / box:** **rejected** — a SIMD reduction needs multiple partial accumulators,
  which changes the f64 summation order (not bit-identical), and layerNorm is "the
  single most parity-sensitive op outside the GEMMs" at ~0.5% of a forward. Leave it
  alone. (Also ruled out for caching in the memoization audit.)

### 8.5 · wordPiece arena (memoization §1 item 3) — **populate ~3% WORSE**, 2026-07-30 (`b74ea6e`)
- **Tried:** replace the per-entry `[]int32` wordPiece cache storage with a flat
  `[]int32` arena + `map[string][2]int32` (offset,len).
- **Mechanism:** collapse ~7,427 per-entry allocations to ~one amortized, improving hit
  locality.
- **Number / box:** **populate ~3% worse** (10,204 vs 9,865 mallocs on 8,983 distinct
  words); steady-state hit throughput identical; only ~4.7% retained compactness gained.
- **Why — the premise was a misread:** the ~9k allocs are `wordPieceCompute`'s own `out`
  slices, NOT the cache's storage, so the arena can't remove them — it *copies* `out` in
  and discards it. Left as-is.

### 8.6 · bm25 Tokenize-side string interner — **1.56× SLOWER tokenize** (`2007d45`/`bc4387a`)
- **Tried:** a sharded interner in `bm25.Tokenize` to unpin/dedup source text.
- **Mechanism:** intern token strings so duplicates share a backing array and keys stop
  pinning the source corpus.
- **Number / box:** **1.56× slower tokenize** (172 → 110 MB/s), `nvidia-rtx2070s`
  (inferred) — from adding a lock+map probe to the zero-cost fast-path view.
- **Resolution:** moving it to single-threaded `Build` fixed it (tokenize at baseline,
  Build +8%). *Where* you memoize is the whole result — the hot path was the trap.

### 8.7 · `examples/rag` redundant `cosine` norm — **~10⁻⁶ of a rerank, "not a finding"**, `nvidia-rtx2070s` (inferred)
- `examples/rag/main.go:132-143`'s `cosine` recomputes the query norm per candidate —
  real redundancy (8 × 768 wasted MACs) but ~10⁻⁶ of a rerank forward. Recorded only so
  it is not "optimized."

### 8.8 · MarshalBinary single-encoder wrapper — **2.6× slower** (inside §7.40's WriteTo win)
- The obvious "one `bytes.Buffer` encoder" implementation of the WriteTo path cost
  **2.6× the time** (30.6 → 80.3 ms) because every byte was written twice; the shipped
  version keeps one encoder with two buffer policies instead. A local dead end inside a
  shipped win.

### 8.9 · `dotW4A8Fold4AVX2` — four independent accumulators — **~1% (noise), not the bottleneck**, `nvidia-rtx2070s`, 2026-08-19
- **Tried:** split `dotW4A8FoldAVX2`'s single f32 accumulator (one `VFMADD231PS` into
  `Y10` per group, all 160 groups serialized on it) into four independent accumulators
  (`Y10`-`Y13`, one group per accumulator per unrolled iteration, combined once at the
  end) — `dot_w4a8_fold4_amd64.s`, candidate only, not wired into `dotW4A8`'s dispatch.
- **Mechanism (the premise):** `dotI8AVX2`'s own comment says its four accumulators
  exist to "break the dependency chain so the four interleaved groups issue
  independently" — the marginal-FMA issue-width probe on the cold kernel had shown
  ratio 0.91 (NOT issue-limited, idle ports even while streaming from DRAM), which read
  as "latency-bound on the single-accumulator chain, not port-bound."
- **Number / box:** hot 17.12 → 17.36 GMAC/s (+1.4%), cold 16.31 → 16.15 GMAC/s (−1.0%)
  — both inside noise, `nvidia-rtx2070s` (K=5120, FFN gate/up/down shape). Correctness
  held (`TestW4A8Fold4_dotMatchesScalar` 1e-5 rel-err vs scalar oracle,
  `TestW4A8Fold4_matchesOriginal` vs the production kernel) — this is a clean measured
  negative, not an untested one.
- **Why — the probe answered a narrower question than it was read for:** "not
  issue-limited" from marginal-FMA injection means idle capacity on the ports FMA
  instructions use specifically. It does not rule out contention on a DIFFERENT port —
  and the nibble-unpack prologue (`VPAND`/`VPSRLW`/`VPUNPCKLBW`/`VPUNPCKHBW`/`VPSUBB`×2,
  8 shuffle/logic ops feeding the 3 MAC+fold ops per group) is exactly the kind of work
  that would bottleneck a shuffle port while leaving FMA ports idle — the dead FMAs get
  absorbed for free because they compete for a *different* resource than whatever is
  actually saturated, producing the same "not issue-limited" reading a genuinely
  memory-bound kernel would. The probe distinguishes "busy vs waiting" only for the one
  port class it injects into; it does not localize which resource, if any, is actually
  full. Left open (not re-chased without new evidence): isolating nibble-unpack's cost
  specifically (a pre-unpacked-weights variant, still per-group-scaled) would need
  building before either VNNI (hardware-gated, no VNNI box available) or a format
  change is worth reconsidering — see `docs/internal/cpu-acceleration.md` item 4 and
  goinfer's `docs/measurements/aikit-w4a8-opsperbyte.md`.
- **Companion finding, 2026-08-23, `apple-m1pro` (NOT this entry's box) — the identical fix
  measured real: 1.4-1.75x.** This entry stays a correct, amd64-scoped negative — the AVX2 kernel
  really is port-bound, not latency-bound, and the accumulator-splitting fix really doesn't help
  there. But the same fix on the NEON `dotW4A8FoldSDOT` kernel (a different ISA, a different
  bottleneck) measured a real win once tried — see goinfer's
  `docs/task-w4a8-neon-bandwidth.md`'s item-3 harness results and `priors-microgpt-c.md` §1's
  demotion note. Two ISAs, two different resources saturated; neither result generalizes to the
  other, and both are now measured rather than assumed.

### 8.10 · `q8Span` SIMD int8→f32 widen — **MEASURED NET-ZERO**, `apple-m1pro`, 2026-08-10
- **Tried:** `q8Span` (the `MatmulBTQ8Into` inner span) re-widened each int8 weight row to f32
  with a scalar `for k := range K { deq[k] = float32(bq[k]) }` on every call. Replaced with the
  SIMD-dispatched `dequantRowInt8(deq[:K], bq, 1.0)` (NEON/AVX2 + scalar tail) — `scale=1.0`
  makes it a pure widen, and `float32(int8)*1.0` is exact in IEEE-754, so the result is
  bit-identical, not merely close.
- **Mechanism (the premise):** audit #2 predicted several ms/token at M=1 (the decode LM head,
  N=vocab), where the widen is roughly half the per-column work and is never amortized across
  rows the way it is at larger M.
- **Number / box:** `BenchmarkQ8LMHeadDecode_fused`, M=1 K=1536 N=151936, best-of-3 on
  `apple-m1pro`: scalar widen **6.79-6.85 ms/op**, SIMD widen **6.83-6.87 ms/op** — flat, inside
  noise. Correctness held (`TestMatmulBTQ8Fused_bitIdentical` / `_Into` green), so this is a
  clean measured negative, not an untested one.
- **Why — the widen was never the bottleneck at this shape.** It overlaps the 233 MB int8 head
  read; memory latency, `dotF32`, and parallel overhead dominate. This is *predicted* by
  `cpu-acceleration.md` item 6, which had already measured `dequantRowInt8` itself as **NOT
  issue-limited** (marginal-FMA probe, `apple-m1pro` stacked/alone **0.94-0.97**): a kernel with
  idle issue slots is waiting on memory, and handing it fewer/wider instructions wins nothing.
  The prediction and the measurement agree — worth noting because §8.9's probe reading did *not*
  survive contact, and this one did.
- **amd64/AVX2 is UNMEASURED** — a different memory subsystem, so not formally closed. But temper
  the expectation: item 6's same probe put the Ryzen/AVX2 host at **0.79-0.82**, i.e. *more* idle
  issue slack than arm64, not less. The evidence that explains the arm64 null points the same way
  on amd64, only harder. Measure before building, and do not treat "unmeasured" as "promising".
- **The code is not preserved, on purpose.** The change was ~43 lines (`linalg/quant.go` +7/-3,
  plus a 33-line decode bench) and is fully re-derivable from the description above — one call
  swapped for one loop. It lived on branch `q8span-simd-widen` (`3df4ec6`, never pushed), deleted
  2026-08-24 once this entry existed: a 43-line patch whose approach is written down is cheaper
  to rewrite than to curate, and the sole record of a negative must not be a commit title on an
  unpushed local branch.

### 8.11 · `vit.cu` `attention` full query-tiled rewrite (audit M-14's recommended fix) — **3.2-3.7× SLOWER than the shipped partial fix**, `nvidia-rtx2070s`, 2026-09-11
- **Tried:** the fix M-14 actually asked for, not the narrower one that shipped
  (`c5e2b3f`, staging K for the block's ONE query). One block per (head, tile of
  8 queries — one warp per query), K/V for the tile staged ONCE in shared memory and
  reused by all 8, online (two-level: per-tile local softmax, one rescale-merge into
  the running per-query state) softmax so the shared footprint stays
  `2·ATTN_KTILE·hd·4` bytes regardless of `np` — the standard flash-attention shape,
  matching the finding's own "query-tiled attention (a block per (head, 16-32
  queries)...)" fix text. Three variants measured: `ATTN_QT` ∈ {2, 8, 16} and
  `ATTN_KTILE` ∈ {16, 32}, plus a stripped variant with NO shared-memory K/V staging
  at all (direct global/L2 reads, same query-tiling and online softmax otherwise).
- **Mechanism (the premise):** the finding's own count — "≥4.9 GB/layer at so400m"
  re-streamed from global memory, once per query — assumes that traffic is real DRAM
  cost. Every variant preserved this project's actual bit-identity posture for this
  file (`BIT-IDENTITY-EXEMPT`, cosine-tolerance gate, not bit-for-bit) and passed it:
  `gpu/visioncuda`'s real-checkpoint SigLIP parity held at cosine 1.000000000, worst
  abs Δ 8.34e-07 — this is a measured PERFORMANCE negative, not an abandoned-for-
  correctness one.
- **Number / box:** `BenchmarkCUDAAttention`, `nvidia-rtx2070s`, best variant
  (`ATTN_QT=8`, `ATTN_KTILE=16`, staged) against the shipped baseline (`c5e2b3f`'s
  partial fix, which itself declines its own staging above `np=3072` — see M-14's
  entry — so the baseline numbers below are its UNSTAGED direct-read path at the two
  largest shapes):

      shape          shipped (c5e2b3f)   query-tiled (this attempt)
      so400m np729     101 GMAC/s          21.0 GMAC/s
      so400m np1024    104 GMAC/s          21.1 GMAC/s
      qwen np1024      108 GMAC/s          23.4 GMAC/s
      hires np2048      96 GMAC/s          21.3 GMAC/s
      hires np3072      74 GMAC/s          21.4 GMAC/s
      hires np4096      68 GMAC/s          21.4 GMAC/s

  Every variant — `ATTN_QT` 2/8/16, `ATTN_KTILE` 16/32, and the no-shared-staging
  direct-read stripped-down version — landed in the SAME narrow 15.9-23.4 GMAC/s
  band regardless of what was tuned. `ATTN_KTILE=64` failed to compile at all
  (`CUDA_ERROR_INVALID_PTX`, register pressure from the per-lane `scores[64]` array).
- **Why — the premise did not survive contact.** The direct-read variant is the
  falsification: it removes ALL shared-memory staging, ALL of the barriers that come
  with it, and reads K/V straight from global memory exactly like the retired
  per-query kernel's unstaged path did — the ONE thing query-tiling was supposed to
  buy (fewer bytes crossing to DRAM) is gone, replaced by nothing. It measured
  IDENTICALLY to the staged version (21.0-21.5 vs 21.4-21.5 GMAC/s at every shape) —
  not a partial win eaten by overhead, no difference at all. That rules out staging
  cost, barrier count (also ruled out directly: `ATTN_KTILE=32` halves the tile-loop
  iteration count from `ATTN_KTILE=16` and changed nothing) and shared-memory
  occupancy pressure as the bottleneck. It leaves one live explanation, unconfirmed
  (no `ncu`/`nsys` on the measuring box — a profiler pass would settle this
  directly): K and V for one head at these shapes are ~1.2 MB each, comfortably
  inside this card's 4 MB L2 — so the "redundant" per-query re-reads the finding
  counted as DRAM traffic were most likely already mostly L2 HITS on THIS hardware,
  and the real cost of the new kernel is the warp-cooperative reduction structure
  itself (a `__shfl_xor_sync` butterfly per (query, key) pair, ~5 dependent
  instructions moving ~3 useful floats/lane at hd=72-80) against the retired kernel's
  and this variant's common ancestor: fully independent, uncommunicating per-thread
  serial dot products, which cost more redundant memory traffic in theory but need
  zero cross-thread synchronization to produce it.
- **What would need to be true to revisit this:** either (a) a profiler on the
  measuring box to confirm L2 hit rate directly rather than infer it, or (b) a shape
  where K/V for one head does NOT fit L2 (a much larger `hd` or `np` than any shipped
  tower uses today — so400m tops out at np=4096, hd=72). Neither is close at hand.
  `attention_seg` (Qwen's segmented kernel) was not attempted — it never got even the
  partial K-staging fix `attention` did, so it is a smaller, lower-risk piece of
  M-14's remaining scope that this result does not speak to either way.
- **The code is not preserved, on purpose**, same reasoning as §8.10: fully
  re-derivable from the description and numbers above, and the working tree shipped
  no change from this attempt — `git log` shows nothing between the audit's M-14
  partial fix and the anncuda C-04 completion that follows it.

---

## Reasoned out, not built — recorded so nobody re-chases them

- **Native Q6_K K-quant matmul** — proven below the 1.3× gate by a byte-ratio ceiling:
  Q6_K reads 210 B/superblock vs int8's 256 → ≤ 1.22× if bandwidth-bound, ≤ 1.0× if
  compute-bound. `apple-m1pro` measured the built-and-correct SDOT path at 20–45× slower
  (98% of it weight unpack). See cpu-acceleration.md "Native K-quant matmul — evaluated,
  NOT shipped." (`a5d9030`)
- **PQ / IVF** — not performance items for this codebase: IVF trades recall for what
  HNSW already does; PQ's win is memory, and int8 already took the 4×.
- **AVX-512 for f32** — deprioritized; the blocker is Intel client parts shipping it
  fused-off, and the int8 VNNI subset (`VPDPBUSD`) is the part worth taking, gated for a
  VNNI-capable machine (none available).
- **Double-buffering `gpu/annmetal`'s dispatch** (2026-08-21) — the second of two
  overlap ideas from the July GPU crossover follow-up (the first, CPU∥GPU
  shard-split `QueryBatch`, shipped — see `cpu-acceleration.md`). `annmetal`'s
  `Score`/`ScoreBatch`/`TopKBatch` are documented "Serialized: the reused
  scratch buffers are per-index, and one command queue runs serially" — a
  double-buffered pipeline (quantize batch i+1 / drain batch i-1 while the GPU
  scores batch i) would overlap that, and on unified memory it would be cheap
  to build. Not built: searched every `ann.FlatI8.QueryBatch` call site in
  aikit (only `examples/gpu-ann/main.go` is a real caller, and it calls
  `QueryBatch` twice total, not in a loop) and the accessible parts of
  goinfer's decoder/docs — goinfer does not import `aikit/ann` at all, only
  `aikit/linalg`/`aikit/embed`. No streaming-batch consumer exists to overlap
  against. This is category (a) below (genuinely open, not deferred with a
  measurement) — revisit if a real caller ever issues `QueryBatch` calls
  back-to-back rather than one-shot.

---

## The three kinds of open item (so "open" is actionable)

- **(a) genuinely open** — nothing measured against it yet. (The VNNI
  `MatmulBTW4A8` variant used to be the example here; it SHIPPED on 2026-08-26 —
  `dot_w4a8_avx512vnni_amd64.s`, +1.23-1.28x on a cloud Xeon — so it belongs in
  (c), hardware-gated for VERIFICATION rather than for existence. Corrected per
  audit §4. Note also that `w4a8TileRows` deliberately DECLINES on a VNNI host:
  the VNNI kernel folds through two f32 accumulators where the AVX2 one uses
  one, so letting the M>1 tile run there would make the result depend on M.)
- **(b) deferred WITH measurement** — built or profiled, the number says "not now": item
  23 (packedFill m-blocking, a wash-risk restructure for a sub-gap win); softmax-scale
  fusion §4.4 (revisit after SIMD `expF32`).
- **(c) hardware-gated** — dead on the boxes we have, needs a different one: AVX-512/VNNI
  (Zen 4+ / Cascade Lake+); native I8MM (item 36, M2+ — `apple-m1pro` predates
  FEAT_I8MM); item 24's amd64 side (needs the AVX2 packed path, deferred).

Someone reading "open" should be able to tell which of these it is before spending a day
on it.
