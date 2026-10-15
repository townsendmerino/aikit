# Task: phase-aliased encoder scratch — attention and MLP share one slab (2026-10-14)

> **BLUF.** Every buffer in `encoder/scratch.go` lives in exactly one phase of a layer:
> attention or MLP. The only state that crosses the boundary is the hidden state `h`, which is
> not in the scratch at all. Yet `ensureLayer` gives each buffer its own storage, so the pooled
> per-worker scratch is *attention + MLP* when it only needs to be the *larger of the two*. It
> also sizes buffers that some models never touch: `qkv` on BERT, and `gate` on every GELU and GeGLU model.
>
> Carving both phases out of one slab, starting at offset 0, changes no arithmetic, so it is
> bit-identical by construction. It adds no fork/join barriers, which is what sank lens §4.2's
> column-blocking on arm64. At the base shape (D=768, I=3072, 12×64 heads), computed from the
> buffer formulas (A-0 replaces these with measured numbers):
>
> | model / shape | today | after A-1 (dead buffers) | after A-3 (aliased) |
> |---|--:|--:|--:|
> | nomic/CodeRankEmbed SwiGLU, L=512 | 27.0 MiB | 27.0 MiB | **13.5 MiB** (−50%) |
> | nomic SwiGLU batched, B·Lmax=3584 | 180.0 MiB | 180.0 MiB | **94.5 MiB** (−48%) |
> | BERT GELU, L=512 | 27.0 MiB | 16.5 MiB | **9.0 MiB** (−67%) |
> | GTE GeGLU, L=512 | 39.0 MiB | 33.0 MiB | **19.5 MiB** (−50%) |
>
> At 8 workers on the batched shape, that is 1.41 GiB → 0.74 GiB of pinned scratch. The
> 180 MiB figure matches the lens scan's 188.7 MB arena; that number was in decimal MB.
>
> **Status: IN PROGRESS. A-0, A-1 and A-2 landed 2026-10-15.** The single real risk is a stale read across the
> phase boundary. A-2 builds the gate for that *before* A-3 changes any layout.

---

## 1 · Background

This came out of reading demofox's
[double-headed stack allocator](https://blog.demofox.org/2026/10/03/the-double-headed-stack-allocator/)
(2026-10-03): one fixed buffer shared by two allocators that grow toward each other, so two
systems with different lifetimes share a budget without fragmenting it.

The runtime allocator itself is not what aikit needs, and §6 says why. The idea underneath it
is useful: **storage is assigned by lifetime, not by name.** The encoder scratch assigns by
name today. Each of ~20 named buffers is sized to its own maximum and kept for the life of
the pooled scratch, whether or not anything else could have used those bytes.

Prior art in this repo: the July campaign's lens §4.2 attacked the same footprint (gate is
47% of the batched arena) by column-blocking. It was footprint-free on amd64, but cost
+3.5–6.9% on arm64 from the extra fork/joins, so it was reverted (perf-dead-ends §2.4, the
`swigluMLP` trailing comment). Aliasing reaches a bigger footprint cut than column-blocking
without touching the matmul schedule.

## 2 · The lifetime map (read from the tree at `3a30656`)

Every field use outside `scratch.go`, by phase:

| buffer | size (floats) | phase | users |
|---|---|---|---|
| `qkv` | L·3D | attention | `attention.go`, `attention_batch.go`, `forward_q8.go` (packed-QKV models only; **BERT never reads it**) |
| `Q`, `K`, `V` | L·D each | attention | same + `bert.go:342` |
| `ctx`, `out` | L·D each | attention | `attentionCore`, batched + q8 twins |
| `qH`, `kH`, `vH`, `ctxHead` | P·hd each | attention | per-head serial loop (P = perHeadLen) |
| `scores` | P·P | attention | per-head serial loop |
| `workerScratch[w]` | w × (4·P·hd + P·P) | attention | head-parallel path (`attnHeadWorkers` > 1) |
| `val` | L·I | MLP | swiglu/gelu/moe/gte/bert/q8 |
| `gate` | L·I | MLP | **`swigluMLP` and q8 SwiGLU only.** BERT, GELU-dense, MoE and GTE never read it |
| `mid` | L·D | MLP | every MLP kind |
| `upGate` | L·2I | MLP | GTE only (`ensureFusedMLP`) |
| `moeAllScores`, `moeGathered`, `moeAcc` | small / L·D | MLP | `moeMLP` (sized inside it) |
| `moeExpert`, `moeWeight`, `moeTokens` | int32/f32, L·topK | MLP | `moeMLP` |
| `moeScores`, `moeOut` | — | **dead in the forward** | declared in the struct; the forward never touches them. Only `moe_group_test.go`'s per-token reference helper uses them, and a comment (`mlp.go:181`) still names them |
| `deqW` | D·max(3D, I) | **both** | every q8 matmul in both phases, so it stays separate |
| `h` | L·D | crosses layers | `make`'d fresh per forward, never in scratch |

The boundary is clean in both directions. `attentionCore` ends by adding `out` into `h` in
place (`attention.go:131` onward), and the caller's LayerNorm reads only `h`. Every MLP
variant reads only `h` and its own buffers, then adds into `h`. The next layer's attention
starts from `h`. Nothing in the MLP reads an attention buffer, and nothing in attention reads
an MLP buffer.

Two contracts already in the tree make aliasing safe:

- **Every scratch buffer is fully written before it is read**, because the pool hands each
  forward the previous forward's values. `TestScratchUpGate_fullyOverwritten` pins that for
  `upGate`. Aliasing changes *whose* stale values arrive (the other phase's instead of the
  same buffer's last layer), not whether they arrive.
- **`Backend.MatmulBT` overwrites `dst` and keeps no host slice after it returns.**
  `enccuda` memcpys `a` and `w` into pinned staging before `UploadAsync` and syncs before
  readback. `encmetal` copies out after `Run2D` has waited. The only pointer-keyed cache,
  `q8 map[uintptr]q8w`, is keyed on **weights**, which aliasing never touches. A-3 writes
  this "no retention" rule into the interface doc so a future backend cannot quietly break it.

## 3 · Plan

### A-0 · Measure today (no behavior change)

- Add a test-only `scratchBytes(s *scratch)` that sums every buffer, including
  `workerScratch`, the MoE tables and `deqW`, as both `len` (what the forward asked for) and
  `cap` (what is pinned). They differ: `ensureF32` allocates `n + n/4`, so cap reads about 1.25×
  the formulas above. Report `scratch-MiB` (cap), `scratch-len-MiB`, `scratch-total-MiB`
  (cap summed over distinct scratches, the batched path's real footprint) and `head-workers`
  through a test-only `scratchPutHook` in `putScratch`, from
  `BenchmarkEncode_singleLong`, `BenchmarkEncodeBatch_rerankN50`, `BenchmarkGTEEncode`,
  `BenchmarkBERT_embed`, `BenchmarkMoEEncode` and `BenchmarkQ8Encode`.
- Record latency (benchstat, n ≥ 8) and scratch-MiB on the M1 Pro (arbiter) and the 3700X.
  These replace the computed numbers in the BLUF.
- Record whether `workerScratch` engages at each shape. At a lone L=512 forward,
  `mOut·L·hd` = 16.7M clears `attnHeadParallelThreshold` (1<<20), so on the M1 Pro it can
  add up to 10 × 1.5 = 15 MiB on top of the table. Under `EncodeBatch` (`inflightForwards` > 1)
  it never engages.

### A-1 · Stop sizing buffers the model never reads (independent; ships alone)

- `ensureLayer` takes a small layout descriptor: `packedQKV bool` and the set of MLP kinds
  present (`gelu | swiglu | geglu | moe`; nomic v2-moe mixes dense and MoE layers per
  `applyMLP`). It sizes `qkv` only when packed and `gate` only for SwiGLU, and it absorbs
  `ensureFusedMLP`.
- Seven call sites across six files: `forward.go`, `forward_tokens.go`, `forward_batch.go`,
  `forward_q8.go` ×2, `bert.go`, `gte.go`. The comment at `attention.go:30` quotes the old
  signature and moves with it.
- Delete the dead `moeScores` / `moeOut` fields, give `moe_group_test.go`'s reference helper
  local slices instead, and fix the `mlp.go:181` comment.
- Gain: BERT 27 → 16.5 MiB, GTE 39 → 33 MiB at L=512. No aliasing and no new risk: a
  buffer that was never read cannot be read stale.
- Caveat: the scratch pool is process-global and buffers only grow, so a process that runs
  BERT *and* nomic keeps the larger sizing. That is true today and stays true.

### A-2 · The phase-boundary poison gate (before any layout change)

Modeled on `vision`'s `TestQwenScratch_poisonedArenaIsInert`, but aimed at the boundary
instead of the whole arena:

- **As built:** every forward calls `s.phaseBoundary()` before each attention and before each
  MLP (14 call sites in 7 forwards). In production it is one branch on a nil test-only hook.
  The gate (`scratch_phase_test.go`) installs a hook that overwrites **every** phase-scoped
  buffer to its full capacity, `workerScratch` and the MoE tables included, with NaN and then
  with 1e4. NaN alone would slip past a comparison, such as top-k routing. The int32 tables
  get 1<<30, so a stale index panics. The buffers are found by reflection, so a field added
  later is covered automatically; a field of unknown type panics until someone decides its
  phase. `deqW` is the one exempt buffer.
- **Correction to the scoped design:** the scope said to poison only the phase being
  *entered*. On today's separate-buffer layout that cannot catch the cross-phase read A-3
  creates: at the MLP boundary `s.out` still holds valid attention output, so the doc's own
  mutation would stay green. What aliasing looks like from the inside is that *both* phases'
  buffers are dead at every boundary, so the gate poisons both.
- Every forward runs on synthetic models (no checkpoint, so it runs in CI): nomic mixed
  SwiGLU+MoE+GELU layers, nomic SwiGLU, `forwardTokens`, q8, BERT, BERT `clsOnly`, GTE,
  GTE `clsOnly`. Each runs at a short input (serial) and at L=160, which takes the
  head-parallel path; that path is asserted, not assumed. q8 never fans out:
  `selfAttentionQ8` has its own serial head loop. Nomic and q8 batched run too. Output must
  be **bit-identical to a fresh-scratch reference**, compared on raw bits, each poisoned
  run twice so the lazily sized MoE tables hold stale values the second time. Comparing two
  poisoned runs with each other cannot catch a stale read.
- **Mutation checks (recorded in the A-2 commit):** MLP reads `s.out` → 7 subtests red.
  Attention reads `s.mid` → 14 red. A head worker reads its `ctxHead` before writing it →
  the 7 head-parallel subtests red.
- A-2 lands green on today's layout first. It then becomes A-3's acceptance test.

### A-3 · One slab, both phases from offset 0

- `scratch` gains `slab []float32` (grown with `ensureF32`'s 25% headroom) and keeps every
  existing field as a **view** into it, so no call site in `attention*.go`, `mlp.go`,
  `gte.go`, `bert.go` or `forward_q8.go` changes.
- `ensureLayer` carves the attention views (`qkv`?, `Q`, `K`, `V`, `ctx`, `out`, `qH`, `kH`,
  `vH`, `ctxHead`, `scores`) from offset 0, then carves the MLP views (`val`, `gate`?, `mid`,
  `upGate`?) from offset 0 too. `len(slab)` is the max of the two region sizes.
- **Alignment:** round every view's offset up to 1024 floats (4 KiB). Today each large buffer
  is its own `make`, which Go page-aligns, and some kernel paths may care about cache-line
  splits. Matching today's alignment means A-3's latency comparison measures the aliasing,
  not an alignment change.
- **Full slice expressions** (`slab[o:o+n:o+n]`), so an overrun panics at the bounds check
  instead of writing into a neighbor view. That property separate `make`s gave for free.
- **Re-carve on every `ensureLayer`**, never cache a layout across forwards. The pool is
  shared across models, so the next forward may be a different model.
- Stays out of the slab: `deqW` (crosses phases), the int32 MoE tables (different element
  type, small), and `h` (§6).
- `scratch.go` gets a phase table as a comment block: for each view, its phase and why. A
  future reader who adds an MLP-side read of an attention buffer should see the rule before
  the gate catches them.
- Write the "must not retain `a` / `dst` after return" rule into the `Backend` and
  `Q8Backend` interface docs.

### A-4 · Fold `workerScratch` into the attention region (optional, measure first)

The head-parallel worker sets are attention-only and can reach 15 MiB at a lone L=512 forward
(A-0 confirms). Carved inside the attention region, they would alias the MLP too. The
serial-path `qH`…`scores` set can also double as worker 0's set. Do this only if A-0 shows the
worker sets engaging at shapes people run. Rerank batches never engage them.

### A-5 · Vision towers: same treatment for `newQwenScratch`

`newQwenScratch` is shared by the qwen, qwen3, glm_ocr and pixtral encoders. It is
**allocated per `Forward`, not pooled**, so a footprint cut there shows up directly as B/op.

- Draw the same lifetime map per encoder first. The likely split is attention-only
  `{qkv, q, k, v, headPool}` against MLP-only `{gate, up}`, while `n1`, `n2`, `o`, `att` and
  `mlpOut` sit beside the residual and need checking per encoder.
- Saving ≈ min(6·np·hidden + headPool, 2·np·inter) floats per Forward.
- Gemma 4 (`gemma4_encoder.go`, `attNormed`/`mlpNormed`) and SigLIP (`vision/encoder.go`
  `encScratch`) have their own scratch types, so they are separate maps.
- Extend `TestQwenScratch_poisonedArenaIsInert` with the phase-boundary poisoning from A-2.

## 4 · Acceptance

| gate | bar |
|---|---|
| Correctness | All encoder goldens unchanged at the raw-bit level. A-2 poison gate green on every forward listed; mutation check red. |
| Footprint | `scratch-len-MiB` ratio A-3 / A-0 at or below the BLUF's ratio (±2%) on each benchmarked shape. Compare like with like: len against len, cap against cap, never cap against the computed column. |
| Latency | **Not slower** than A-0 on any listed benchmark beyond benchstat noise, M1 Pro (arbiter) and 3700X. This is lens §4.2's "free" bar, the one column-blocking failed. Any regression over ~2%: check alignment first, then report rather than ship. |
| Release | The usual RELEASING.md ritual, perfgate VERDICT included. Footprint is the claim, so the release notes quote measured MiB, not these computed ones. |

A-1 can ship on its own if A-3 fails its latency bar.

## 5 · Risks

- **Stale read across the boundary.** This is the C-05 class of bug in a new place; aliasing by
  convention bit this repo once already. Mitigations: A-2 lands first, the mutation check
  proves it bites, and the phase table in `scratch.go` makes the rule visible.
- **A future backend that retains host slices**, for example an async device path that reads
  `a` after `MatmulBT` returns. Today's two backends don't. The interface doc change in A-3 makes it a
  stated contract instead of an accident.
- **Locality shifts.** The MLP's first matmul now writes `val` into bytes attention just used.
  That might help (warmer TLB) or might not; A-0/A-3's latency comparison settles it, and the
  4 KiB alignment keeps it from being an alignment effect.

## 6 · Not in scope, and why

- **A runtime double-headed stack (mark/release, two tops).** It earns its keep when region
  sizes vary *during* a run and can't be planned. Here every size is known at `ensureLayer`
  time, so the allocator collapses into a static layout: offsets computed once, no `tos`, no
  release discipline to get wrong. Static is strictly safer.
- **`h` in the slab.** It crosses layers and escapes: `BERT.hiddenStates` returns it to the
  caller. It is one L·D allocation per forward (1.5 MiB at L=512). That is a B/op question for a
  separate item, not a pinned-footprint one.
- **`linalg.Workspace`.** Its buffers are K-wide and tiny, and their *separate* storage is the
  C-05 fix (`deq` split from `f32` so a per-call scratch can't overwrite held activation
  scales). Packing them back together would undo that for no measurable gain.
- **Column-blocking `gate` (lens §4.2).** Already measured out on arm64. Aliasing doesn't need it.
- **GPU device memory.** The place where two budgets really compete for one pool (KV cache
  growing with context against prefill activations growing with batch) is goinfer's memory
  planner, not aikit's encoder backends.
