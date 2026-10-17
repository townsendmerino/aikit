# Task: the aikit half of the memory campaign (2026-10)

> **Status 2026-10-17: all ten items committed on `memory-campaign-aikit` (item 5 stopped and reported, by its own rule); not yet released.** Scope: the aikit items of goinfer's
> `docs/tasks/task-memory-campaign-2026-10.md` (branch `memory-campaign-2026-10`, MEM3 and MEM14–MEM16),
> plus D6 (the safetensors mmap finalizer) and annmetal, as Francis decided them on 2026-10-17. MEM3 shipped
> first on this branch (`db44648`). Still on hold: MEM17 (chunk text by reference, waits on D4) and MEM4(a)
> (BF16 towers via `WrapBF16`, waits on goinfer's MEM0 measurements).
>
> One commit per item. Each item's gate is registered in this file, in a commit before the item's code.
> Quick checks only by day; perfgate, vulncheck and the full release gate go to nobara or the night queue.
> Release: v1.71.0 (new API) plus a gpu-module tag for annmetal, through `RELEASING.md`, when Francis says go.

## How "identical" is checked

There are no on-disk BM25 or HNSW goldens; their suites build seeded synthetic corpora. So every "identical"
gate below is checked against the tree before the change: a capture of the item's outputs is hashed on the
unchanged code (the hash and the capture's inputs are committed in the item's test), and the changed code must
reproduce the hash exactly, bit for bit. Model-backed captures (SPLADE, Q8, Model2Vec, BERT/GTE) use the local
fixtures under `testdata/` that are gitignored, so those tests skip in CI; each such item also gets a
model-free test that runs everywhere.

## Group 1 — exact results or plain fixes

### 1. SPLADE row tiling (`encoder/splade.go:SPLADE.expandIDs`)

Max-reduce the vocabulary logits over row tiles of 16 instead of materializing L × vocab.

- **Why it can be exact:** `linalg.MatmulBT` is documented M-invariant (`TestMatmulBT_MConsistent`): an
  output element is bit-identical whatever the row count, so a tile computes the same logits as the whole
  matrix; the bias add is elementwise; max is order-free for non-NaN values and both forms skip NaN.
- **Gate (a), identical:** on `testdata/splade-model`, `expandIDs` output (terms and weight bits) hashed before
  the change at L ∈ {1, 15, 16, 17, 64, 512} equals the hash after. Model-free: a synthetic head (random t,
  decoder, bias, including negative-only columns and a NaN logit) gives identical pooled vectors tiled vs whole,
  at the same L set.
- **Gate (b), bytes:** `runtime.MemStats.TotalAlloc` around one 512-token `Expand`, before and after. Bar: after
  ≤ before − 55 MiB (the L × V × 4 B logits are 59.6 MiB at V = 30,522; one 16-row tile is 1.9 MiB).

### 2. `WeightsQ8.WordEmb` read from the mapping (`encoder/weights_q8.go`)

Stop cloning the f32 word-embedding table onto the heap; alias it from the checkpoint mapping, which the Q8
weights now keep open (new `ModelQ8.Close` / `WeightsQ8` holds the file).

- **Gate (a), identical:** on `testdata/encoder-model` (CodeRankEmbed), `ModelQ8.Encode` output for a fixed text
  set (queries and documents) hashed before equals after.
- **Gate (b), heap:** `HeapAlloc` after `LoadQ8` and two GCs, before and after. Bar: drops by ≥ 80 MiB (the
  table is 30,522 × 768 × 4 B = 89.4 MiB).
- Linux note: `TestQ8LoadFootprint`'s peak/after RSS ratio (≤ 1.5) is Linux-only; it is re-run on nobara in the
  release gate, not by day.

### 3. BM25 postings arena (`bm25/index.go`)

One flat `[]posting` with per-term offsets in place of one slice (and 24 B header, plus append slack) per term;
`df` becomes the list length.

- **Gate (a), identical:** over the seeded corpora of `TestScoring_bitIdenticalToReference` plus a 20k-document
  seeded corpus, every `Search`/`TopK`/WAND result list (ids and score bits) hashed before equals after; the
  existing reference-formula test stays green.
- **Gate (b), bytes:** bytes held by a built index (`HeapAlloc` delta after GC, built over the 20k-document
  corpus), before and after. Reported, no bar beyond "smaller".

### 4. HNSW build pre-sizing (`ann/hnsw.go`)

`BuildHNSW` pre-sizes `vecs`/`bq`/`scales`/`nodes` from n; `Add` stops allocating the throwaway int8 row and
sizes each new neighbor list to its cap (mmax + 1) so later back-links do not regrow it.

- **Gate, identical:** for f32 and int8 builds (n = 5,000, dim 64, two seeds, default and SimpleNeighbors),
  `Query` and `QueryEf` results (ids and score bits) for 200 seeded queries, plus the `MarshalBinary` bytes,
  hashed before equal after. Allocations per `Add` reported (`testing.AllocsPerRun`), no bar.

## Group 2 — decided by Francis

### 5. D6: the safetensors mmap finalizer (`embed/safetensors.go`)

Today the finalizer never runs: each `Tensor` holds `owner` → its file, and the runtime does not collect a
finalized object in a cycle, so a dropped mapping leaks (measured 2026-10-17, `db44648`).

- **Audit first:** every caller of `OpenSafetensorsMmap` / `OpenSafetensorsShardedMmap` and their wrappers, in
  aikit, goinfer (all packages) and ken: does anything drop the file while still reading its views? Any unsafe
  caller is fixed before the cleanup change; if one cannot be fixed, stop and report.
- **Then:** make the release real with `runtime.AddCleanup` (its argument is the regions, which do not reach the
  file, so the cycle no longer blocks it), cancelled by `Close`; correct the comments.
- **Gate:** (a) a test that a dropped mmap file is unmapped: a weak pointer to it goes nil and the cleanup's unmap
  runs (counted by a test hook), within 10 GCs; a held file is not unmapped. (b) The audio lifetime test now goes
  red on its own: re-run its three mutations with the cycle intact; all three must fail. (c) `go test ./...` in
  aikit, `go test -race` on embed, encoder, audio, vision; goinfer's `decoder` package tests against this aikit
  through a temporary `go.work` (not committed).

### 6. MEM14: BM25 `Builder.Add` and a saved index format

- `Builder` (N6): `NewBuilder()`, `Add(tokens []string)`, `Build() *Index`. Peak is the index under
  construction plus one document, instead of the whole tokenized corpus. `Build(docs)` becomes a loop over it.
- Format (N4), version 1: header, then aligned sections (docLen, postings arena, term dictionary). `WriteTo`,
  `ReadIndex` (heap) and `LoadIndexMmap` (the postings arena aliases the mapping, `Close` releases it).
  Version-skew policy: the loader rejects an unknown version with an error; a later version keeps reading v1.
- **Gate (a):** `Builder` over the 20k-document corpus gives scores identical (bits) to `Build`, and to the item-3
  hash. (b) Bytes allocated by a `Builder` build are O(index), not O(corpus): `TotalAlloc` across
  the `Add` calls and `Build` (inputs excluded: each document's tokens are made just before its `Add`) ≤ 5 × the
  built index's bytes. (`TotalAlloc` counts every grown-and-discarded slice, which for geometric growth is about
  2 × the final size, so 5 × leaves room for that and for the final compaction, and nothing for a tokenized
  corpus.) Peak `HeapInuse` during the build is reported beside `Build`'s over the held corpus. (c) A mapped load holds no heap copy of the postings:
  `HeapAlloc` delta after `LoadIndexMmap` ≤ the dictionary + docLen bytes + 1 MiB, and the arena's address lies in
  the mapping. (d) Round trip: `ReadIndex` and `LoadIndexMmap` give results identical to the built index; a
  corrupt or truncated file is an error, never a panic (fuzz seed corpus).

### 7. MEM15: the HNSW graph as CSR (`ann/hnsw*.go`)

Blob version 5: the graph is three aligned sections, per-node list starts, per-list id offsets and one flat id
array, aliased under `LoadHNSWMmap` and copied (as one arena each) under `Load`. Same format bump as BM25's.
v4 blobs still load (ken has indexes on disk). An index built in memory keeps per-node lists; a loaded index is
CSR and turns back into lists on its first `Add`.

- **Gate (a):** search results identical (ids and score bits) to the item-4 hash, for a built index, its v5 round
  trip (`Load` and `LoadHNSWMmap`), and a v4 blob written by the current code (committed under `ann/testdata`)
  loaded by the new code. (b) Heap after a mapped load: for n = 50,000, dim 64, int8, `HeapAlloc` delta after
  `LoadHNSWMmap` and GC ≤ 1 MiB (today: the graph, ~n × 160 B ≈ 7.6 MiB). (c) Add-after-load still works and
  matches Add on the original. (d) Corrupt and truncated v5 blobs are errors, not panics.

### 8. Model2Vec mapped by default (`embed/model.go`)

`LoadFromFS` maps `model.safetensors` when the `fs.FS` hands back a real `*os.File` (as `os.DirFS` does), and
heap-reads otherwise (`embed.FS`, `fstest.MapFS`); `Load` follows. New `LoadFromFSHeap` keeps the heap copy.

- **Gate (a), identical:** `Encode`/`EncodeBatch` output on the embedded-corpus example's model is bit-identical
  between the mapped and heap loads.
- **Gate (b), measurement (no bar):** time to first result, in process, by day: `bench` W3 shape, heap vs mapped,
  interleaved, 15 rounds each, warm page cache, M1 Pro; report median and range. The number goes in the
  CHANGELOG (the README's figure, +17%, was nobara's).

### 9. f16 or int8 word embeddings for BERT and GTE (`encoder/bert.go`, `encoder/gte.go`)

New `LoadBERTWithOptions` / `LoadGTEWithOptions` with a word-embedding precision (f32 default, f16, int8 per
row). The table is only gathered, so the rows are dequantized at gather time. SPLADE (tied decoder) stays f32.

- **Gate (quality), registered before the run:** for each fixture (`minilm-model`, `crossencoder-model` and
  `splade-model` trunks as BERT; `arctic2-m` as GTE) and every text in that fixture's golden plus a fixed
  20-text set, cosine of the output embedding against the f32 load. **f16: min cosine ≥ 0.9999. int8: min
  cosine ≥ 0.999.** Also reported: mean cosine and the table's bytes.
- The default stays f32 unless the gate clearly passes and Francis agrees.

### 9b. The word-embedding default: native precision, mapped (decided 2026-10-17, after item 9)

Francis first chose f16 as the default; measured, that adds half the table to the heap for an F32 checkpoint, whose
f32 table was a mapped view (arctic2-m: +366 MiB, load 0.33 → 0.84 s), because every local BERT/GTE fixture is
F32. Decided instead: **the default keeps the table in the checkpoint's own precision, read from the mapping.** An
F32 table stays a mapped view (as before v1.71.0); an F16 or BF16 table becomes a mapped 16-bit view, widened one row
at a time in the gather — where today it is widened whole onto the heap (twice the stored bytes). When the 16-bit
data is not 2-byte aligned in the mapping, the load falls back to today's widened copy. `WordEmbF16` and
`WordEmbInt8` stay opt-ins (heap tables). No f16 sidecar.

- **Gate (a), identical:** on F16 and BF16 checkpoints — a synthetic tiny BERT and GTE written by the test, and
  minilm-model converted to each — `Embed`/`Encode` output from the mapped 16-bit view is bit-identical to the
  widened path. The row decode is checked exhaustively: all 65,536 bit patterns of each 16-bit format decode to
  the same f32 bits as embed's widening accessors (`Float16sToF32`, `BFloat16sToF32`).
- **Gate (b), no heap:** an F16/BF16 load adds no heap for the table beyond row scratch: HeapAlloc after a native
  load ≤ HeapAlloc after the widened load − the table's f32 bytes + 1 MiB (the rest of the model is widened in
  both).
- **Gate (c), misaligned fixture:** a checkpoint whose data starts at an odd offset (header padded to make it so)
  loads through the widened-copy fallback, with output identical to an aligned copy of the same weights.
- F32 checkpoints: default output bit-identical to v1.70.2 (it is the same path); the encoder suite stays green.

## Group 3 — annmetal (gpu module, its own tag series)

### 10. annmetal: no-copy codes, reused score buffers, opt-in on Apple

- New optional `ann.MappedBackend` seam: `FlatI8.EnableGPU` on a `LoadFlatI8Mmap` index passes the mapping, and
  annmetal wraps the code block in a no-copy buffer (page-aligned span of the mapping, `Buffer.At` to the codes)
  instead of copying it; otherwise it copies as today.
- `ScoreBatch` and `TopKBatch` reuse per-index device scratch, grown on demand, instead of allocating M × N per call.
- Opt-in: nothing in aikit, goinfer or ken imports annmetal or calls `EnableGPU` on its own (checked 2026-10-17);
  importing the package and calling `EnableGPU` is the opt-in, and that is documented on `EnableGPU`.
- **Gate (a):** results identical to the CPU `FlatI8` (scores bit-equal, as the existing parity tests require),
  aliased and copied, `Score`, `ScoreBatch`, `TopKBatch`. (b) `Device.CurrentAllocatedSize` delta after
  `EnableGPU` on a mapped index (n = 100,000, dim 256): ≤ n × 8 B (scales + output) + 1 MiB; the copied path is
  reported for comparison (≥ n × dim). (c) Repeated `ScoreBatch` calls at a fixed M allocate no new device buffers
  after the first.

## Results

| item | commit | gate |
|---|---|---|
| 1 SPLADE tiling | `6e9609d` | PASS: hash identical; 512-token expand 89.4 → 7.0 MiB allocated |
| 2 Q8 WordEmb from the mapping | `7bfd399` | PASS: hash identical; heap after load 200.0 → 110.5 MiB |
| 3 BM25 postings arena | `74a9365` | PASS: hash identical; index 18.66 → 14.89 MiB; Build +5.6% (flagged for perfgate) |
| 4 HNSW build pre-sizing | `e9f1c41` | PASS: query and blob hashes identical; allocations per Add 69 → 2 |
| 5 D6 | `a25abff` | STOPPED before the change — see below |
| 6 MEM14 Builder + saved index | `06c5619` | PASS: Builder hash identical; allocates 4.38× the index (bar 5×); mapped load +0.66 MiB heap (bound 1.18); round trips identical; damage rejected (one gap found and closed: docLen now tied to the tf sums) |
| 7 MEM15 HNSW CSR, blob v5 | `50f9780` | PASS: query hash identical built, v5 Load/LoadHNSWMmap, and v4 fixtures; graph byte-identical (v4 re-encoding hash); mapped load +0.001 MiB heap (bound 1 MiB; v4 6.9 MiB); Add-after-load matches; damage rejected |
| 8 Model2Vec mapped by default | `6284cb0` | PASS: 1,753 texts bit-identical mapped/heap/MapFS; TTFR (M1 Pro, warm, 15 rounds) heap 40.2 → mapped 29.7 ms, 0.74× (nobara's July +17% is the other sign; re-measure there) |
| 9 BERT/GTE f16/int8 word embeddings | `cb0c08b` | PASS, clearly: min cosine f16 1.0000000 on all 4 fixtures (bar 0.9999), int8 0.99993–0.999997 (bar 0.999); tables 2×/4× smaller. Default stays f32 until Francis agrees |
| 10 annmetal no-copy + scratch reuse | `0f20f32` | PASS: (a) mapped ≡ copied bit for bit, both rank as the CPU (Δ 0); (b) mapped EnableGPU +0.44–0.47 MiB footprint (bar 1.76), copied +49.3 MiB — instrument changed, see below; (c) 60 batch calls add no device memory |

### Item 5 (D6): stopped before making the release real

The audit (every caller of the two mmap opens and their wrappers, in aikit, goinfer and ken; 2026-10-17)
found one caller that would crash and one hazard that is not a caller:

1. **goinfer `internal/whisper/decoder.go` `Load`** keeps F32 views (`st.TensorF32`, no copy) and drops the
   file; its fixture `testdata/whisper-tiny-rand` is 89/89 F32, and serve reaches it through
   `LoadTranscriber`. Fix (not made here, goinfer's to land): hold `st` on `Decoder` with a `Close`, close it on
   the error return, and give `Transcriber` a `Close` over both halves. Today it only leaks a second mapping.
2. **Early collection of a holder.** The views point into the mapping, not the Go heap, so only the holder
   struct's field keeps the file alive, and the runtime may collect the holder as soon as a method has loaded its
   last field, while that method is still reading the views. With a real cleanup that is an unmap under a running
   forward. Closing it needs `runtime.KeepAlive` in every public forward of every view-holding type, in aikit
   (`StaticModel`, encoder `Model`/`ModelQ8`/`BERT`/`GTE`/`CrossEncoder`/`SPLADE`, the audio encoders) and in goinfer
   (`decoder.Model` and its drafters, LoRA, the Metal/CUDA uploaders that keep host views), and keeping it there for
   every future one.

Every other site is safe: it holds the file for the views' lifetime, or copies out before `Close` (table in the
audit report). Item 2 is a project-wide discipline in another repo rather than a caller to fix, so per the
instruction ("if it isn't possible, stop and report") the cleanup was not made real. Done instead: the comments
now say what the code does (`finalizeMmaps` explains the cycle and why it stays), and
`TestSafetensorsMmap_droppedFileWithTensorsIsNotFinalized` pins it, so a change that breaks the cycle fails until
the holders are revisited. GGUF's mmap file has no tensor back-pointer, so this cycle does not apply to it (not
probed).

### Item 10: two corrections to its registered gate

- **(a) said "scores bit-equal, as the existing parity tests require".** The existing parity tests require identical
  indices and scores within 1e-3: the GPU's float rescale is not bit-identical to the CPU's. Run as two checks:
  the mapped device index against the copied one, bit for bit (it is: same kernel, same bytes), and against the CPU
  at the existing bar (indices identical; the score Δ measured 0).
- **(b) named `Device.CurrentAllocatedSize` as the instrument.** Metal counts a no-copy buffer there at its full
  length although it allocates nothing (measured: 25.97 MiB mapped vs 25.20 copied), so it cannot tell aliasing
  from copying. Replaced, after seeing that, by the process's physical footprint (`footprint -f bytes`) right after
  `EnableGPU`, with the registered bar unchanged (n × 8 B + 1 MiB). The test also requires that the copied index's
  footprint delta exceed 90% of the codes, so the instrument is shown able to see a copy. A first measurement
  taken after a query as well read 1.62–2.03 MiB against the 1.76 bar: the query's own host allocations, which is
  why the bar sits at the registered point (after `EnableGPU`) and the after-query figure is only reported.
