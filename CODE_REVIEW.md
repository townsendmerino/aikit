# aikit — Code Review: Fixes Needed

**Date:** 2026-07-18
**Reviewed at:** working tree on `main` (fbecab3, clean tree)
**Scope:** both modules — root (`linalg`, `embed`, `encoder`, `vision`, `ann`, `bm25`, `sparse`, `fuse`, `chunk`, `mmap`, `topk`, plus `bench`/`benchmarks`/`examples`) and the `chunk/treesitter` submodule — non-test sources, plus the amd64/arm64 assembly at the Go/asm boundary and the CI config.
**Method:** `go build` + `go vet` on both modules and on `GOOS=darwin,linux GOARCH=arm64` cross-vet of the assembly packages (all clean); `gofmt -l` (clean); then a multi-pass manual review across four parallel per-subsystem passes, with every Critical/Major finding re-verified line-by-line against this tree before inclusion. Line numbers refer to the current working tree.

Severity scale: **Critical** = memory corruption, crash, or silent wrong output on a *supported* path. **Major** = a defect reachable with realistic input (including the crafted/mismatched checkpoints this library exists to ingest), or a documented contract the code does not keep. **Minor** = robustness gaps, latent traps, divergences. **Nit** = polish. Items tagged *(verify)* depend on a spec detail or reference model worth confirming before fixing.

---

## 1. Overall assessment

aikit is a mature, carefully engineered library — visibly further along than the runtime built on top of it. The things that stand out: the bit-exactness engineering in `linalg` (M-invariance, width-invariance, and packed-path identity all *constructed* and pinned by exact-equality differential tests), the GGML quant-decode kernels (Q2_K–Q6_K, IQ2_S/IQ3_S grids verified byte-exact against the ggml reference), the GGUF byte-parser's untrusted-input hygiene (overflow-safe `need()`, allocation clamps, a nest-bomb regression test), the determinism-as-contract discipline across the retrieval stack (descending-score/ascending-id tie-breaks pinned everywhere), and an unusually honest documentation culture that records *measured* rationale (tile sizes, thresholds, even where mmap `madvise` is a no-op). `go vet` is clean on both modules and both architectures, `gofmt` is clean, and CI already runs staticcheck.

Because the happy paths are this solid, the findings concentrate in one place: **aikit is the untrusted-input boundary for the entire ecosystem** — goinfer routes every hostile GGUF/safetensors byte, every crafted persisted index, and every mismatched checkpoint through these parsers — and the hardening that is excellent in the *GGUF byte cursor* is not uniformly applied across its siblings. The safetensors parser, the vision loaders, the HNSW persist path, and the mmap-lifetime story each have a gap where crafted or merely-mismatched input reaches an allocation, an `unsafe` cast, a division, or a `munmap` without the guard the GGUF path would have applied. None of these fire on well-formed input, which is why the fuzz suite (capped at small amd64 inputs) and the parity gates are green — but they are exactly the inputs a foundation library must survive.

The defects cluster as:

1. **The untrusted-input boundary is uneven.** The safetensors header is trusted where the GGUF header is checked (path traversal on shard names, no shape×dtype cross-validation, `unsafe.Slice` with no alignment guard); the HNSW float32 load branch skips the allocation clamp its int8 sibling has; GGUF metadata-array recursion has no depth cap. (H1–H5)
2. **mmap lifetime rests on finalizers without `KeepAlive` backstops.** Three zero-copy accessors (`RowDequantizer`, `TensorF32` widening, `FlatI8.query`) can have the region unmapped mid-read under GC timing. (H6)
3. **The vision loaders skip the validation the text encoder earned.** No tensor-shape checks, no config-field validation → panics (not errors) on mismatched or hostile checkpoints. (H7)
4. **A few exported contracts don't match the code.** The public `Dot4x4`/`Dot8x4` godoc misdescribes the arm64 lane layout; the `Chunker` byte-fidelity invariant is contradicted by the overlapping line fallback; the panic-vs-error boundary is inconsistent and, above the parallel threshold, unrecoverable. (M1–M3)

The remedy is mostly to *propagate patterns aikit already has* — the GGUF cursor's clamp discipline, the text encoder's `TensorF32(name, want...)` shape checks, a `runtime.KeepAlive`/`AddCleanup` convention — into the siblings that lack them. High-leverage fixes (a shared "validate count before allocating" helper, alignment-or-copy in the safetensors accessors, shape-checked vision loads) close whole classes.

---

## 2. The untrusted-input boundary — fix first

These are graded high because this library *is* the parser layer for hostile model files and persisted indexes downstream. Each is reachable from a crafted (or, for H6/H7, merely mismatched) input that passes today's checks.

### H1. safetensors sharded load: shard filenames from untrusted `index.json` escape the model directory (path traversal)
`embed/safetensors.go:194-198`

```go
dir := filepath.Dir(indexPath)
for _, fn := range files {                       // fn from weight_map, attacker-controlled
    data, err := mmap.MapReadOnly(filepath.Join(dir, fn))
```

`files` comes straight from the index JSON's `weight_map` values with no validation. A bundle with `"weight_map": {"w": "../../../etc/somefile"}` (or an absolute path) makes the loader `mmap` and parse a file outside the bundle — a zip-slip-style arbitrary-read primitive, and any file that parses as safetensors is then exposed through the aggregate `Tensor` namespace. The `fs.FS` variant is incidentally safe (`fs.ValidPath` rejects `..`); only the Mmap path is affected. **Fix:** reject `fn != filepath.Base(fn)` (or require `fs.ValidPath(fn)` with no separators) before the `Join`.

### H2. safetensors header never cross-validates shape × dtype against the byte range
`embed/safetensors.go:363-371, 415`

`parseSafetensors` validates only the offset range:

```go
if t.DataOffsets[0] < 0 || t.DataOffsets[1] > len(payload) || t.DataOffsets[0] > t.DataOffsets[1] { … }
… Shape: t.Shape,   // stored untouched — no dim>0, no ∏shape·itemsize == end−start
```

`Shape` is trusted verbatim (negative dims accepted; no product·itemsize consistency check, unlike the reference implementation). So `{"w":{"dtype":"F32","shape":[4096,4096],"data_offsets":[0,4]}}` parses, `TensorF32("w",4096,4096)` passes its `shapeEqual` check (metadata-only, `:415`) and returns a 1-element slice — and `encoder/weights.go` loads exactly this way (`loadF32 → TensorF32(name, want...)`), then indexes by shape and **panics at inference on a hostile file**, violating the never-panic parse contract. Related: `Elements()` (`:599`) and `model.go`'s `len(embData) != vocab*dim` guard use unchecked `int` products, so `shape:[2^32,2^32]` wraps to 0 and "loads". **Fix:** in `parseSafetensors` reject non-positive dims and require overflow-checked `∏shape·dtypeSize(dtype) == end−start` (unknown dtypes may skip, but should then be rejected by the typed accessors as today).

### H3. safetensors `unsafe.Slice` accessors never check alignment
`embed/safetensors.go:468, 482, 496, 511`

```go
return unsafe.Slice((*float32)(unsafe.Pointer(&t.raw[0])), len(t.raw)/4), nil
```

`t.raw` begins at `base+8+headerLen+DataOffsets[0]`; neither `headerLen` nor the offsets are required to be a multiple of the element size (a hostile file trivially isn't, and even honest files aren't guaranteed — tensors pack back-to-back, so an `I64` tensor after a 4-mod-8-sized `F32` lands 4-aligned). A misaligned conversion violates the `unsafe.Pointer` alignment rule: under `-race`/`checkptr` it's an unrecoverable "misaligned pointer conversion" throw, and on strict-alignment ports a SIGBUS — invisible to the fuzz targets on amd64. **Fix:** check `uintptr(unsafe.Pointer(&t.raw[0])) % align == 0` in each accessor and fall back to an allocating byte-wise decode (the BF16/F16 paths already show the copy pattern), or validate `(8+headerLen)%8==0` plus per-dtype offset alignment at parse.

### H4. GGUF metadata: unbounded recursion on nested arrays → fatal stack exhaustion
`embed/gguf.go:199-217`

```go
case ggufArray:
    …
    arr = append(arr, c.value(et))   // recurses, no depth limit
```

The nested-array *allocation* blowup was fixed (`ggufArrayPrealloc` + `gguf_nestbomb_test.go`), but recursion *depth* is still ≈ input/12 (each level costs 12 bytes: a 4-byte element type + 8-byte count). A ~50–150 MB file of repeated `(et=array, n=…)` headers drives millions of frames past Go's 1 GB goroutine-stack limit, which aborts the process with "goroutine stack exceeds" — not a recoverable panic, so `recover()` can't uphold the "error or succeed, never panic" contract. Real metadata nests 1–2 deep. **Fix:** add a depth counter to `gcur` (cap ~128, mirroring `encoding/json`'s nesting cap) and set `c.err` beyond it.

### H5. HNSW load: the float32 vector branch has no `ndocs×dim` allocation guard
`ann/hnsw_persist.go:259-268` (int8 guard at `:251`)

The int8 branch pre-checks the product before allocating; the float32 branch does not:

```go
} else {
    vecs = make([][]float32, ndocs)
    for d := range vecs {
        row := make([]float32, dim)   // no product guard; loop doesn't break on c.err
        for j := range row { row[j] = c.f32() }
```

`count()` clamps `ndocs` and `dim` *individually* (each ≤ remaining/4) but never the product, so a crafted ~1 MB header (dim≈250k, ndocs≈250k) attempts ~250 GB of retained row allocations — and the loop keeps allocating after `c.err` is set (`c.f32()` just returns 0). This directly contradicts the file's own guarantee ("counts are additionally clamped before they drive an allocation … a hostile blob returns an error rather than over-allocating"); the fuzz target caps inputs at 64 KB, so it can't reach it. **Fix:** mirror the int8 guard (`int64(ndocs)*int64(dim)*4 > int64(len(c.b)-c.pos)` → `ErrFormat`) and `break` the row loop on `c.err != nil`.

### H6. mmap lifetime: three zero-copy accessors can be unmapped mid-read (finalizer without `KeepAlive`)
`embed/gguf.go:411-444` (`RowDequantizer`); `embed/safetensors.go:410` (`TensorF32` BF16/F16 widening); `ann/flat_i8_mmap.go:38` (`FlatI8.query`)

`OpenGGUFMmap`/`OpenSafetensorsMmap`/`FlatI8` install a `runtime.SetFinalizer` that `munmap`s the region, and the docs bless "let the finalizer run" over `Close`. But the hot accessors alias the mapping in a way that doesn't keep the owner reachable:

```go
// RowDequantizer's returned closure captures raw (into the mmap), not g:
into = func(start int, dst []float32) error {
    …
    dequantRange(info.typ, raw, start, dst, bs)   // g is unreachable here
```

Per the `SetFinalizer` docs, the owner can become unreachable "as soon as the call begins" — so a streaming loader holding only `into` (or even `g.Tensor(name)`, where `g` is dead while `into(0,data)` still runs) can have the region unmapped under the dequant loop: SIGSEGV or silent garbage. `FlatI8.query` is safe today only by accident (it reads `f.n` after the matmul, keeping `f` live); a routine refactor hoisting that read opens the window. `TensorF32`'s widening loop has the same shape but callers using `defer f.Close()` are incidentally safe. **Fix:** capture the owner in the closure and `defer runtime.KeepAlive(owner)` inside each accessor (covers both GGUF entry points, since `Tensor` goes through `RowDequantizer`); longer-term a `Tensor.owner` back-pointer, or migrate to `runtime.AddCleanup` keyed off the mapping. This is the exact hazard the runtime docs prescribe `KeepAlive` for.

### H7. vision loaders skip all tensor-shape validation → panic on mismatched/hostile checkpoints
`vision/encoder.go:103` (`LoadEncoder`), `vision/qwen_encoder.go:117` (`LoadQwenVisionEncoder`)

Both read every tensor with no expected dims, though the expected shapes are computed right there:

```go
v, err = st.TensorF32(pfx + name)          // no want... dims
… newQMat(v, rows, cols, quant)            // rows/cols known but not passed to the loader
```

Contrast `encoder/weights.go:273` (`TensorF32(name, want...)`), which shape-checks all 112 tensors. Consequences: with `quant=true` a wrong-shaped projection **panics inside `linalg.QuantizeRowsInt8` at load**; with `quant=false` it surfaces as an out-of-range **panic inside `linalg.MatmulBT` at Forward**; and `h[i] += e.posEmb[i]` (`encoder.go:181`) panics if `position_embedding` is smaller than `numPatches*hidden`, or silently uses a prefix if larger — wrong output, no error. **Fix:** pass the known dims to `TensorF32` exactly as the text encoder does (posEmb `[numPatches, hidden]`; qkv/proj/fc/ln shapes are all computed already).

### H8. vision: no config-field validation → divide-by-zero panics
`vision/encoder.go:93, 221`; `vision/qwen_encoder.go:151, 444`

```go
e.grid = cfg.ImageSize / cfg.PatchSize      // ÷0 panic if patch_size absent
hd := hidden / nH                            // ÷0 at Forward if num_heads absent (:221)
```

Same class in the Qwen tower (`headDim := hidden / cfg.NumHeads` `:151`; `vmws := WindowSize / merge / PatchSize` and `llmH % vmws` `:444`). There is no vision equivalent of the text encoder's `ValidateAssumptions`, so unsupported activations load silently and an odd `headDim` mis-partitions heads (leaving output columns zero). **Fix:** a small validate step after config parse (all dims > 0, `hidden%heads==0`, `image_size%patch_size==0`, known activation), mirroring the text side.

---

## 3. Major

### Exported contracts that don't match the code

**M1. Public `Dot4x4`/`Dot8x4` godoc describes a sums layout the arm64 kernel doesn't produce.**
`linalg/linalg.go:11-16`. The exported doc says the kernels write "every row's full sum into the first lane of its 4-lane block in sums (the rest zero)". True on generic/AVX2 (the wrappers reduce into lane 0), **false on arm64** (`dot_arm64.s` leaves the four raw partial-sum lanes). An external caller following the public doc — reading `sums[r*4]` only — gets correct results on amd64 and ~¼-magnitude garbage on arm64, the primary deployment arch. Internal callers are safe because they horizontal-sum all four lanes (`ann/flat.go:175-179`), so this is a *documentation/API-contract* bug, not an internal miscompute — but it's on an exported primitive. The `dot_generic.go:12-17` header compounds it by claiming its full-sum-in-lane-0 layout "mirrors the arm64 asm exactly" (the two layouts differ; they merely agree after a horizontal sum). **Fix:** document the portable contract "horizontal-sum each 4-lane block" (as `Dot2x8`'s doc already does) and correct the `dot_generic.go` header.

**M2. Panic-vs-error boundary is inconsistent, undocumented, and unrecoverable above the parallel threshold.**
`linalg/weightmat.go` + `linalg.go:220`. Three regimes coexist with no documented rule: (a) `WrapInt8`/`WrapInt4`, `QuantizeRowsInt8`/`QuantizeGroupsInt4`, `MatmulBTInto`, `dotF32` panic unconditionally with no "panics if" godoc; (b) `MatmulBTQ4`/`W4A8`/`Dequantize*` shape checks exist only under `-tags aikit_checks` (`checks_off.go` compiles them out), so production violations surface as anonymous bounds/÷0 panics deep in kernels; (c) `WrapF32` validates nothing (`len(w) < rows*cols` blows up later inside `MatmulBT`). Worse, above the MAC threshold the blocked/quant kernels run on spawned goroutines (`parallelSpawnCols`), so a shape-violation panic fires on a *worker* goroutine and is **unrecoverable even by a caller wrapping the call in `recover()`** — a crafted checkpoint that slips past `Wrap*` (e.g. via `rows*cols` int overflow, which the `len(q8) != rows*cols` check doesn't exclude) can hard-kill the goinfer process. This is the seam through which the downstream `H2`-class panics detonate un-catchably. **Fix:** give `Wrap*`/`Quantize*` error-returning forms (or at minimum document the panics), validate `rows/cols/group ≥ 0` and overflow, and do the O(1) shape checks at the public matmul entry *before* the goroutine fan-out so failures are recoverable caller-side.

**M3. `Chunker`'s byte-fidelity invariant is contradicted by the line fallback every chunker falls back to.**
`chunk/registry.go:35` (interface doc), `chunk/chunk.go` (package doc), `chunk/line` instance. The 1.0-committed interface doc states chunks are "contiguous and non-overlapping so concatenating their Text in order reproduces source byte-for-byte", and calls byte-fidelity "a load-bearing invariant every Chunker must satisfy". But the registered `"line"` chunker is 50-line windows with **5 lines of overlap**, and `ChunkFile` routes every unsupported language through it — as do treesitter's timeout/parse-failure fallback and markdown's non-md fallback. `TestChunkFile_UnsupportedLanguageFallsBackToLine` even asserts the concatenation is *longer* than the source. Any consumer implementing "reconstruct file from chunks" (snippet display, resolve-by-line) silently breaks on exactly the fallback files. **Fix:** either carve out the overlap exception in the interface/package docs, or make the fallback instance used by `ChunkFile`/treesitter `Overlap:0`.

### Process / performance

**M4. No arm64 execution in CI for an arm64-first, hand-written-assembly library.**
`.github/workflows/ci.yml`. CI runs only `ubuntu-latest` + `windows-2025` (both amd64). On amd64, `has2x8Kernel` is false, so the `Dot2x8` pair path, `packedFill` (large-K B-panel packing), the NEON f32/i8/SDOT/W4A8 asm, and `detectDotProd` are all **dead code in every CI run** — and the exact-equality gates that protect the M-invariance and packed-path bit-identity contracts only ever execute their trivial amd64 legs. Every tuning note references an M1 Pro, i.e. arm64 is the primary target, yet an arm64-only regression (an asm edit, a reduction-order change) ships green. **Fix:** add an arm64 job (GitHub's `ubuntu-24.04-arm` runners, or a `qemu-user` leg — the auxv-based `detectDotProd` is explicitly qemu-compatible per its comment).

**M5. treesitter/markdown line numbering rescans from byte 0 per span — O(N²/chunkSize), unbounded by the 1s parse timeout.**
`chunk/treesitter/chunker.go:299-300, 345`. `StartLine`/`EndLine` call `bytesBefore('\n', source, sp.start/​end)`, which loops from byte 0 every call:

```go
StartLine: 1 + bytesBefore('\n', source, sp.start),
EndLine:   1 + bytesBefore('\n', source, sp.end) - …,
```

Total work ≈ Σ(sp.start+sp.end) ≈ N²/chunkSize: negligible at 100 KB, ~0.5 s at 1 MB, tens of seconds for a single multi-MB generated `.js`/`.ts` file — dwarfing the 1-second parse timeout the package added specifically to bound per-file cost. Spans come out of the cAST sorted ascending and contiguous, so one incremental pass (carry a cumulative newline count forward) makes it O(N). `chunk/markdown`'s `lineNumber()` has the same from-zero scan with a smaller constant. **Fix:** thread a running offset/line count across spans.

---
## 4. Minor

Grouped by area. Each entry: location — issue → fix.

### `embed` (parser robustness & lifetime)

- **`gguf.go:422` `maxElems` bound** — `2*len(g.data)+qkK` assumes "≥ ~0.5 bytes/element", but Q2_K is 0.328 B/elem, IQ2_S 0.320, Q3_K/IQ3_S 0.43. A legitimate Q2_K/IQ2_S tensor occupying >~65% of the data section (Q3_K ~86%) is falsely rejected with "exceeds data section" even though `tensorBytes` would validate it exactly. → Use `4*len(g.data)+qkK` (safe for ≥0.25 B/elem, still overflow-proof), or derive the exact per-type bound; fix the comment either way.
- **`gguf.go:308` map preallocation hints** — `hintLen` clamps the count by remaining *bytes* (assumes ≥1 byte/entry), but a KV entry needs ≥13 bytes and a tensor ≥24, and each hinted slot costs ~40–64 bytes of eager bucket allocation. A hostile 100 MB file claiming `kvCount≈10^8` forces a multi-GB map prealloc before the parse loop hits EOF — the same amplification class the `ggufArrayPrealloc` fix closed for arrays, left open for the two top-level maps. → Divide the clamp by the minimum encoded entry size (13 / 24), or cap the hints at a constant.
- **`gguf.go:377` `Dims`** — `dims[i] = int(d)` converts untrusted `uint64` with no bound (`d ≥ 2^63` → negative int), unlike `RowDequantizer` which validates the same dims. `Dims` is the documented cheap probe "e.g. deriving vocab size", so a caller feeds the result to `make()` or a shape product on a hostile file. *(verify how downstream uses it)* → Apply the same clamp/validation, or document that `Dims` values are unvalidated.
- **`gguf.go:205,221,324` `ErrFormat` wrapping** — `ErrFormat`'s doc promises loaders wrap it for malformed headers, and most paths use `errFormatf`, but the array-length, unknown-value-type, and tensor-dim-count errors use plain `fmt.Errorf`. A caller sniffing format with `errors.Is(err, ErrFormat)` (the documented pattern) gets inconsistent answers by which byte is corrupt; `parseShardIndex`'s errors are similarly unwrapped. → Switch these sites to `errFormatf`.
- **`safetensors.go:410` `TensorF32` widening** — same finalizer class as H6 but the widened result is documented "safe to keep past Close()"; callers with `defer f.Close()` are incidentally safe. → `defer runtime.KeepAlive(f)` in `TensorF32`/`TensorI32` (folded into the H6 fix).

### `linalg` / `mmap` / `topk`

- **`quant.go:89` `MatmulBTQ8`** — row-outer loop nest re-widens each weight row M times (`deq[k] = float32(bq[k])` runs per (i,j)); the O(K) widen dominates the vectorized dot and repeats M× per row. `MatmulBTQ4`/`w8a8Span` are deliberately column-outer to amortize exactly this (CHANGELOG v1.0.1 concedes "Q8 re-widens per row"), yet `doc.go` still lists `MatmulBTQ8` as a prefill (large-M) kernel. → Swap to j-outer/i-inner (numerically free — each `dst[i,j]` is an independent dot); cuts the dominant scalar work ~M× at prefill.
- **`quant_export.go:10` `DotI8`** — documents "caller must pass `len(a)==len(b)`" but enforces nothing: the SIMD dispatch takes `n:=len(a)` and reads `&b[0]..n`, so `len(b)<len(a)` reads past `b`'s allocation (silent garbage or a page-boundary fault), while the small-n scalar path bounds-*panics* at `b[k]`. An exported slice primitive with an arch- and length-dependent failure mode — the one place this memory-safe library reads OOB from safe code. → `if len(a) != len(b) { panic(...) }` (matching `dotF32`), free relative to the ≥16-element kernel.
- **`quant.go:387` `MatmulBTW4A8`** — documented "fast M=1 (decode) path" yet allocates `aq := make([]int8, M*K)`, `aScales`, and a per-worker `sums := make([]int32, nGroups)` that is entirely dead (all three `dotW4A8` impls document "sums is unused" — a leftover of a removed kernel). Unlike W8A8 there's no zero-alloc `…Into` variant, so an int4 decode loop pays ~K bytes × ~7 projections × layers per token in GC pressure the W8A8 path was re-engineered to remove. → Drop the `sums` parameter; add `MatmulBTW4A8Into(ws, …)`.
- **`topk/topk.go:84` NaN handling** — at capacity, `if score <= s.heap[0].score { return false }` is false for NaN, so a NaN item always evicts the current minimum; once in the heap, every sift comparison against it is false and the ordering silently degrades, so `Result` can return NaN entries while genuinely high-scoring items were dropped. BM25/dot scorers can produce NaN from degenerate vectors, and this is the terminal retrieval stage. → Reject in `Push` (`if score != score { return false }`) or document "NaN is caller error".
- **`mmap/doc.go:23`** — claims the RAM cap is "firm on Linux/BSD", but `madvise_other.go` (`!linux && !darwin`) makes every BSD a no-op, so the cap is firm on Linux *only*; the package's own files contradict each other (`mmap_test.go:63` agrees with the no-op reading). The "no Madvise wrapper" justification is also avoidable — `x/sys/unix` (already a dependency for darwin) provides `Madvise` on the BSDs. → Fix the sentence, or wire the BSDs through `x/sys/unix` and make it true.
- **`mmap/mmap_unix.go:36`** — `MapReadOnly` rejects files `< 8` bytes (duplicated in `mmap_other.go`, pinned by a test), baking a caller's header-size expectation into a general-purpose mapping primitive; the godoc ("whole contents") doesn't mention it, so a legitimate small file fails surprisingly. → Document the minimum, or drop the check and let format parsers own header validation.

### `ann` / `bm25` / `chunk`

- **`ann/hnsw.go:240` `Config{M:1}`** — `NewHNSW` only rejects `m<=0`, so `M=1` sets `mL = 1/ln(1) = +Inf`; `randomLevel`'s `int(-log(r)*mL)` is then implementation-defined (MinInt64 amd64 / MaxInt64 arm64) and `make([][]int32, l+1)` **panics on the first Add** with a message that gives no hint the cause is `M=1`. → Clamp `m` to ≥ 2 (or compute `mL` with `max(m,2)`) in the constructor.
- **`chunk/regex/chunker.go:247` `scanDepth`** — the string-escape handler skips the byte after `\`; when that byte is `\n` (a legal line continuation in JS/TS/Rust string literals), `atLineStart` bookkeeping jumps past that line's start offset, `li` never advances again, and **every subsequent line's depth stays 0** for the rest of the file. Impact is boundary quality, not data loss (the partition stays byte-exact) — e.g. a nested Rust `fn` after the continuation is misread as depth 0 and becomes a mid-function boundary. → `for li < n && lineStart[li] <= pos { depth[li] = cur; li++ }` so jumped-over line starts record the current depth.

### `encoder` / `vision`

- **`encoder/bert.go:184,137` bounds & maxSeq** — `hiddenStates` indexes `wordEmb[int(id)*D:…]`/`posEmb`/`typeEmb` with no range check (unlike the Nomic path, which substitutes OOB ids), so `Embed` panics on any `id ≥ vocab` or `len(ids) > MaxPos`; and `b.maxSeq = c.MaxPos` is overridden by `sentence_bert_config.json`'s `max_seq_length` with no `min(…, c.MaxPos)` clamp, so a checkpoint whose sentence-bert config exceeds `max_position_embeddings` panics on the first long input instead of truncating. → Clamp `maxSeq` to `c.MaxPos`; bounds-check ids (or substitute UNK) in `hiddenStates`.
- **`encoder/forward.go:55` (+3 sites) OOB-token fallback** — `if id<0 || id>=VocabSize { id = 100 }` with the comment "[UNK] id (100) is always in range" — only true for vocabs ≥ 101; the repo's own `vocab_size:4` fixtures would index `WordEmb[400:404]` and **panic — the exact crash it defends against** — and it silently assumes 100 is UNK for any drop-in checkpoint. → Substitute id 0 (clamped), or thread the tokenizer's `UnkID`.
- **`encoder/weights.go:76` `ValidateAssumptions`** — parses `scale_attn_weights` (`:57`) but never enforces it, while `selfAttention` unconditionally applies `1/sqrt(headDim)`: a checkpoint with `scale_attn_weights:false` loads clean and silently produces wrong activations — the exact failure this function exists to prevent. An odd `HeadDim` (e.g. 770/10) also only fails at first Encode (`panic("rope headDim must be even")`), not at load. → Add both cases to the validation switch.
- **`vision/qwen_encoder.go:203` per-image divisibility** — `forwardViT` validates the *global* `nPatches%mergeUnit` but not per-image `h/w` divisibility by `spatial_merge_size`; `gridTHW={{1,3,4}}` (merge 2) passes the global check but `rotaryFreqs`/`windowIndex` emit fewer entries than `groups`, so `winIdx[g]` **panics OOB**. → Validate `g[1]%merge==0 && g[2]%merge==0` (and non-negative t/h/w) per grid entry, return an error.
- **`encoder/forward_q8.go:67,138`** — the Q8 forward hardcodes CLS pooling (`copy(cls, h[:D])`) and ignores `Cfg.pooling`, while the f32 mirror routes through `poolOne(…, w.Cfg.pooling)`; `pooling` *is* copied into `WeightsQ8.Cfg`, so when the planned mean-pooling loader lands, `ModelQ8` will silently return CLS vectors. → Route both q8 poolings through `poolOne`.
- **`encoder/crossencoder.go:94` pair truncation** — `pairIDs` gives the query the whole budget first (`if len(q)>avail { q=q[:avail] }` then `d=d[:avail-len(q)]`), so a long query starves the document to zero tokens and the score becomes document-independent; HF/sentence-transformers `CrossEncoder` (the cited reference) defaults to `longest_first`. *(verify)* → Implement `longest_first`, or document the deviation.
- **`encoder/weights.go:149` no exported `Close`** — `Weights.st`/`BERT.st` retain the mmapped file for life; the only deterministic unmap is the unexported `st.Close()` the tests reach into, so long-running servers that swap models keep ~547 MB mappings alive until a GC finalizer runs. `LoadWeightsQ8` and the vision loaders close/copy-out correctly — the f32 text models are the odd ones out. → Add `(*Model) Close()` (and BERT/CrossEncoder/SPLADE) forwarding to `st.Close`.
- **`vision/qwen_encoder.go:523` `applyRotaryVision`** — `tmp := make([]float32, hd)` allocates inside `for patch { for head { …(q); …(k) } }` per block: ~8M small allocs per realistic image (~8k patches × 16 heads × 32 blocks) — the same per-head-alloc hotspot the text encoder already fixed. → Hoist one `tmp` (or a fixed stack array).
- **`encoder/bert.go:207` per-head allocation** — BERT/SPLADE/cross-encoder allocate `qH/kH/vHT` and an L² `scores` per (layer, head) and route through the allocating `matmulBT` (which never consults `wantParallelMatmul`), so a lone `SPLADE.Expand` runs single-threaded while `Model.Encode` fans out. Material for SPLADE indexing. → Reuse the scratch arena + `matmulBTInto`.

---

## 5. Nits & polish

- **`linalg/dot.go:14`** — `panic("encoder: dotF32 length mismatch")` carries a stale package prefix (every other panic says `linalg:`), and the comment "For ken's matmul shapes" leaks a private-repo name. → Fix prefix and scrub "ken".
- **`topk/topk.go:20`** — package doc cites "ADR-025/026", "internal/ann.Flat.Query", "internal/bm25", "ken is on Go 1.26" — none exist in this module (real consumers are `aikit/ann`/`aikit/bm25`, no `internal/` tree, no shipped ADRs). Public godoc pointing at unreachable paths confuses external users. → Scrub the private-monorepo references.
- **`linalg/doc.go:42`** — "the parallel and blocked paths are bit-identical to the serial scalar reference … exact equality" overstates: the exact-equality tests pin parallel==serial (same kernel), width- and M-invariance; `MatmulBT` vs the naive scalar triple-loop is *tolerance*-checked (1e-3), and CHANGELOG 1.6.0 notes results moved ~1e-5. → Reword to "bit-identical to a serial run of the same kernel".
- **`linalg/weightmat.go:114`** — `MatmulBT` and `MatmulBTInto` test their dispatch switch cases in different order (`q4` first vs `w8a8` first); harmless with constructor-built values (mutually exclusive), but a future both-set/hand-built `WeightMat` routes the two entries to different kernels. → Align the case order, ideally factor into one switch.
- **`encoder/model.go:59` `SetMaxSeqLength`** — mutates `m.maxSeqLength` unsynchronized while the type documents "all internal state is immutable after Load / goroutine-safe"; a data race if called during in-flight Encodes. → Document "call before sharing", or make it atomic.
- **`encoder/crossencoder.go:89`** — discards the `ok` from `SpecialID("[CLS]")`/`("[SEP]")` (a tokenizer without them silently yields id 0 = `[PAD]`), and doesn't check `ce.labels = ct.Shape[0] > 0` (degenerate classifier → `all[0]` panic). → One-line error returns at load/score.
- **`vision/preprocess.go:68`** — `ic.Width*ic.Height > cfg.MaxPixels` multiplies two attacker-controlled header ints; on 32-bit (386/arm) a `65535×65535` header wraps negative and bypasses the decompression-bomb guard the doc advertises. → `int64(ic.Width)*int64(ic.Height)`.
- **`encoder/quant.go:36`** — `quantizeRowsInt8`/`dequantizeRowsInt8`/allocating `matmulBTQ8` are dead outside tests (production moved to `linalg.QuantizeInt8` + `matmulBTQ8Into`); two "bit-identical" quantizers is a drift hazard. → Delete and re-point tests, or note the redundancy.
- **`vision/encoder.go:164`** — `Forward`'s CPU path duplicates `GridPatches` (gpu_export.go) verbatim; the resident path already calls it. → Call `GridPatches` from `Forward`. Also worth one doc line: concurrent `Forward` is safe only if `EnableResident`/`Close` aren't called concurrently (`e.resident` written unsynchronized).
- **`ann/flat_i8_mmap.go:38`** — folded into H6, but explicitly: add `defer runtime.KeepAlive(f)` in `query()` so a future refactor can't open a use-after-unmap.

---

## 6. Architecture notes

**What's working well.** The two-module split (root + `chunk/treesitter` isolating the `gotreesitter` dep) keeps the core dependency-light. `linalg`'s dispatch is clean and cheap — GOARCH build tags select the file set, one-time init probes (hand-rolled CPUID/XGETBV on amd64; `/proc/self/auxv` HWCAP on linux-arm64; constant-true on darwin-arm64) set package bools the hot paths branch on, and every SIMD kernel has a scalar reference pinned by exact-equality (integer) or tolerance (f32-fold) tests. The bit-identity engineering (M-invariance, width-invariance, packed-path identity) is genuinely first-rate and *constructed*, not hoped-for. The GGUF byte cursor, the GGML quant kernels, the HNSW implementation (correct heap orientation, Alg-4 diversity heuristic, tombstone deletes that sidestep the entry-point-on-delete problem), and the determinism contract across retrieval are all strong. Documentation density — recording measured rationale and honest platform caveats — is well above average.

**Recommendations, highest leverage first:**

1. **Make the untrusted-input boundary uniform.** The GGUF cursor's discipline (overflow-safe `need()`, clamp-before-allocate, `ErrFormat` wrapping) is the standard; port it to safetensors (H1/H2), the HNSW f32 branch (H5), and GGUF recursion depth (H4). A shared `validateCount(n, elemSize, remaining)` helper consulted by every parser would single-source the clamp and prevent the next sibling from drifting.
2. **Fix the mmap-lifetime class structurally, not per-accessor.** A `Tensor.owner`/`FlatI8` back-pointer plus `runtime.KeepAlive` (or `runtime.AddCleanup`) in the zero-copy accessors (H6) eliminates the whole use-after-unmap family instead of documenting the traps. This is the highest-severity latent class because it's a segfault in otherwise-correct downstream code.
3. **Alignment-or-copy in the safetensors typed accessors (H3).** Either validate alignment at parse and reject misaligned tensors, or fall back to the allocating byte-wise decode the BF16/F16 paths already use. This is the one place the "pure-Go, memory-safe" promise currently has a hole.
4. **Bring the parallel/quant matmul entries into the panic-vs-error discipline (M2).** Do the O(1) shape checks *before* the goroutine fan-out so a bad checkpoint yields a recoverable panic (or, better, an error) at the API boundary rather than an uncatchable worker-goroutine crash — this is what makes the downstream H2/H7 panics survivable.
5. **Bring the vision loaders up to the text encoder's bar (H7/H8).** Pass known dims to `TensorF32`, add a `validate*` step. The transformer block now exists ~6 times (attention/batched, Q8 twins, BERT, two vision towers) and validation/pooling drift has already crept in (Q8 ignoring `Cfg.pooling`, vision missing shape checks) — a shared block or at least a shared validation helper would arrest it.
6. **Run arm64 in CI (M4).** The flagship kernels and every bit-identity gate are dead on the amd64-only runners; an `ubuntu-24.04-arm` (or qemu) leg turns the arm64 half of the matrix from manual-run to gated.
7. **Reconcile the byte-fidelity contract (M3).** Decide whether "concatenate chunks == source" is a guarantee or a best-effort, and make the interface doc, the package doc, and the fallback instance agree.
8. **Seed the fuzzers past their current reach.** The GGUF nest-bomb fuzzer stops at allocation (not depth); the HNSW fuzzer caps at 64 KB (below the f32 over-allocation trigger); the safetensors targets run on amd64 (blind to the alignment throw and to `checkptr`). Widening these — and running the typed accessors under `-race` — would catch H3/H4/H5 mechanically. The harness contracts are already exactly right.

---

## 7. Suggested order of attack

| Order | Items | Rationale |
|---|---|---|
| 1 | H1, H5, H4 | Small, self-contained parser guards; crash/traversal class |
| 2 | H2 + M2 together | Shape cross-validation is only safe once the matmul entries validate pre-fan-out |
| 3 | H3, H6 | The two `unsafe`/mmap memory-safety holes; do as one lifetime+alignment pass |
| 4 | H7, H8 | Vision loaders → text-encoder parity; mechanical, closes the panic-on-load gap |
| 5 | M1, M3 | Exported-contract corrections (arm64 doc, chunk fidelity) — doc + one instance |
| 6 | M4, M5 | CI arm64 leg; O(N) line numbering |
| 7 | Minors by area | Batch by file: `embed` parser gaps, `linalg` perf/`DotI8`, `encoder`/`vision` robustness |
| 8 | Nits (§5) | Doc scrubbing, dead code, race annotations |

*Review produced by an automated multi-pass code review (4 parallel per-subsystem deep reviews + independent line-by-line verification of every High/Major finding against this tree). Items tagged (verify) depend on a reference model or downstream-usage detail — confirm before fixing. Everything else was re-verified in the source.*
