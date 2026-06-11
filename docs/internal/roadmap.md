# aikit roadmap v2 — post-1.3.0

> Rewritten 2026-06-09 at v1.2.0; **refreshed 2026-06-10 after v1.2.1 + v1.3.0**
> (which executed §1.1 and all of §3's unblocked items within a day). Per-item
> annotations and measurements live in git history
> (`git log -- docs/internal/roadmap.md`) and the CHANGELOG. Sections ordered
> by importance; items within likewise; **[impact / effort]** tags.
>
> **The binding constraint has moved — and v1.3.0 confirmed it.** Everything
> unblocked-and-engineering gets done within a release cycle; what remains is
> gated on an audience, a Python parity toolchain, and x86 hardware. The open
> engineering list is now: one showcase example, one CI gate, and §4's API
> health — everything else waits on an external unlock.

## Shipped in v1.3.0 (2026-06-10)

- **§1.1 comparative benchmarks — done** (isolated `benchmarks/` module;
  README perf table + capability matrix; the methodology finding: synthetic
  high-dim vectors can't measure recall — real embeddings required).
- **§3.1–3.3 — done**: FlatI8 `MarshalBinary`/`LoadFlatI8` (+ fuzz),
  `LoadFlatI8Mmap` zero-copy + `Close`, int8 HNSW (`Config.Int8`, recall@10
  Δ0.0000 vs f32, format v3). The ¼-memory `//go:embed` story is now uniform
  across both index types.
- **Off-roadmap**: `linalg.MatmulBTAcc64` (f64-accumulating matmul,
  bit-identical to scalar f64 — for goinfer's MoE router near-tie problem).
- New deferred follow-ups recorded below: HNSW zero-copy mmap (needs a
  format-v4 alignment bump — §3.2), BEIR/VectorDBBench slice + inference-row
  vs hugot (§1.1 remainder).

## Shipped in v1.2.0 (v1 roadmap scorecard)

- §1 perf: amd64 AVX2 W4A8 (closing the marquee cliff), scores·V attention
  vectorization (~2.85× long-seq Encode; QKᵀ premise inverted by profiling),
  ann SIMD scoring (~7× Flat scan), K-crossover (non-issue, doc fix).
- §2 features: `sparse` (index half), HNSW persistence, `bench` harness,
  `FlatI8`, `TokenizePlain`, `QueryFilter`+base/delta/fuse (immutability made
  design rule 4), `embed.Truncate`, pooling seam (internal).
- §3 robustness: complete — parser+dequant fuzzing (5 crashes found/fixed),
  chunk fuzzing, build-tagged kernel contract checks, mmap guardrail, nightly
  long-fuzz CI.
- §4 quality: complete — HNSW Alg-4 default (recall@10 0.68 → 1.00), recall
  regression gate, RSF.
- Measured-and-closed: int8 M-tiling (no win), `GOEXPERIMENT=simd` (deferred,
  see Watch list), AVX-512/VNNI (hardware-gated, see Watch list).
- §6 docs: architecture.md, linalg doc.go, ken-free quickstart.

---

## 1. Adoption & proof — now the top of the roadmap

The code is ahead of its audience (the road-to-1.0 critique's point, still
true two releases later). Everything here is unblocked today.

1. **`examples/embedded-corpus` showcase** — ✅ **DONE.** A separate module
   (`GOWORK=off`, so its `//go:embed`-model can't touch the root build) with a
   `gen/` builder (Go stdlib + aikit package docs via `go doc -all` + aikit markdown
   → embed → `FlatI8` → committed `index.bin` + `corpus.json`) and a `main.go`
   runtime that `//go:embed`s the model + index + corpus and answers Go/aikit
   questions over hybrid dense (int8 ANN) + lexical (BM25) → RRF. Measured: one
   ~70 MB self-contained binary (64 MB is the model; the aikit surface + 443 KB int8
   index + corpus is ~5 MB), **~50 ms startup, zero external files**. Model-gated
   smoke test; results are spot-on (file I/O → `bufio.ReadLine`, int8 quant →
   `NewFlatI8`/`QuantizeRowInt8`). The demo that recruits §1.3's adopter — the
   `//go:embed`-a-corpus lane no Python/ONNX stack reaches.
2. **Release-process gate in CI** — ✅ **DONE.** `scripts/release-gate.sh` (a
   testable script, not trapped in YAML) + `.github/workflows/release.yml`,
   tag-triggered (`v*.*.*`) with a `workflow_dispatch` test path. Checks: the
   `## [x.y.z]` CHANGELOG section + compare link exist; `apidiff` shows no Hard-tier
   incompatible change vs the previous tag (the release bar); and the core module
   pulls no external dep beyond `golang.org/x/text` (the dependency-light invariant,
   a robust stand-in for the docs-vs-`go list` check). On a tag it then creates the
   GitHub Release from the CHANGELOG section if absent — closing every gap the
   v1.1.1/1.2.0/1.2.1 cycles leaked.
3. **One named external adopter** — [high / not-engineering]. Unchanged and
   still the highest-leverage non-code item: ship the next minor *with* a
   consumer (an MCP server embedding a docs corpus is the natural shape).
   The benchmark table + §1.1 example are the recruiting assets; the
   road-to-1.0 critique's warning ("1.0 to an audience of nearly nobody")
   now applies to 1.3.
4. **Comparative benchmark table** — ✅ **DONE in v1.3.0** (isolated
   `benchmarks/` module; README perf table — aikit HNSW/FlatI8 ~0.995 recall
   vs coder/hnsw 0.22 (construction-limited) and chromem-go ~45× p50 — plus
   capability matrix incl. Bleve/hugot; real-embedding methodology).
   **Remainder, [medium / medium]:** a BEIR/VectorDBBench slice for
   cross-referenceable absolute numbers; an inference-side `embed`/`encoder`
   vs hugot pure-Go throughput row (pairs with §2.2's MiniLM work — same
   models would then run on both, making it apples-to-apples).

## 2. Model-blocked track — one toolchain unblocks four items

Every remaining model-side feature is parity-blocked the same way: it needs
torch + sentence-transformers + a checkpoint to generate golden fixtures.
Items 2–5 are sequenced behind item 1.

1. **Stand up the Python parity toolchain** — ✅ **DONE.** A gitignored `.venv`
   (torch 2.12 / sentence-transformers 5.5.1 / transformers 5.11.0, CPU) pinned in
   `scripts/requirements.txt` with a `scripts/README.md` setup + regeneration doc.
   Validated by loading all-MiniLM-L6-v2 and embedding (dim 384). Unblocks the rest
   of the model-blocked track (§2.2–2.4).
2. **MiniLM-class encoder support** — ✅ **DONE.** A second encoder architecture
   (`encoder.BERT`, `bert.go`) alongside CodeRankEmbed: learned absolute positions,
   GELU FFN, no-RoPE attention, mean pooling. `LoadBERT(dir)` + `Encode(text)` is
   the cgo-free `.encode()`. Parity-pinned to all-MiniLM-L6-v2 (golden via
   `pin_minilm.py`): hidden states Δ~1e-6, sentence embedding cosine 1.000000,
   tokenizer ids identical to HF. Separate file, CodeRankEmbed untouched (additive).
   This is the "BERT family you already use" answer to hugot's CrossEncoders.
3. **SPLADE expansion head** — ✅ **DONE** (the full in-process head, not the interim
   pin-script fallback). §2.2's BERT forward made it tractable: a SPLADE model is a
   BERT + a masked-LM head, so `LoadSPLADE` reuses `LoadBERT` (now prefix-aware for
   raw `BertForMaskedLM`) and adds the head; `Expand(text)` → log(1+ReLU) → max-pool
   → `sparse.SparseVec`. Parity 1.000000 (identical term sets) vs
   naver/splade-cocondenser-ensembledistil. Learned-sparse retrieval now runs
   end-to-end in-process (`Expand` → `sparse.New`/`Query`) — the `sparse` package
   (index half from 1.2.0) is complete.
4. **potion-retrieval-32M parity pin** — ✅ **DONE** (and bigger than scoped). The
   "loads already, same format" premise was wrong: potion-retrieval-32M uses the
   *standard* Model2Vec layout (only an `embeddings` tensor), while embed required
   the vocabulary-quantized layout (`mapping` + `weights`). Made both optional —
   absent ⇒ direct token-id indexing + mean pooling. Parity cosine 1.000000 vs
   StaticModel.encode (golden via `pin_retrieval.py`); potion-code-16M unregressed.
   Docs point general-retrieval users to potion-retrieval-32M.
5. **forward_q8 scores·V vectorization** — ✅ **DONE.** It *was* oracled after all
   — `TestModelQ8_cosineMatchesF32` (cosine ≥ 0.97 vs the f32 model) + the testdata
   model cover the Q8 forward. So `selfAttentionQ8`'s scalar scores·V triple-loop
   now mirrors the f32 path: build `vHT` (V transposed) and compute the context as
   `scores·(vHT)ᵀ` through the A·Bᵀ matmul (the same §1.3 win, ≈⅓ of attention).
   Mathematically equivalent — Q8-vs-f32 cosine unchanged at 0.997 (the parity
   gate); full encoder + -race green. Pure perf, no API change.

## 3. Embedded-index coherence — core complete in v1.3.0

§3.1–3.3 shipped (see scorecard): FlatI8 persistence + fuzz, FlatI8 zero-copy
mmap + `Close`, int8 HNSW (`Config.Int8`, recall Δ0.0000, format v3). The
¼-memory `//go:embed` story is uniform. What remains is deliberately gated:

1. **HNSW zero-copy mmap** — [low-medium / medium]. *New (deferred out of
   §3.2):* the f32 vector block needs a format-level alignment bump (v4) and
   the nested graph is parsed regardless, so only the vectors can alias.
   Worth doing only bundled with the next format bump, not as its own —
   format churn (v1→v2→v3 in two days) has a cost once blobs circulate.
2. **Binary/Hamming pre-filter + f32 rescore** — [low-medium / medium]. The
   third compression tier (32× candidate filter). Only worth it at corpus
   sizes aikit doesn't see yet — keep behind §1.3's adopter signal.
3. **Windows real mmap** (`CreateFileMapping`) — [low / low-medium]. Got
   slightly more relevant now that mmap'd indexes exist (`LoadFlatI8Mmap`
   heap-falls-back on Windows, like `embed`'s loaders); still gated on a
   sizable Windows consumer.
4. **Format-stability note** — [low / low]. *New:* the persisted-blob formats
   burned two versions in 48 hours (v1→v2 Alg-4 byte, v2→v3 int8 mode) —
   fine while Experimental and pre-circulation, but before any adopter embeds
   blobs in *their* releases, decide and document a format-compat policy
   (e.g. "Load always reads N−1" or "rebuild per minor"), and consider
   reserving header bytes so the next axis doesn't need v4.

## 4. API & code health — the untouched v1 section

Carried over intact; all three remain open and got *more* relevant as the
Experimental tier grows toward graduation.

1. **Typed sentinel errors** — ✅ **DONE.** `ann.ErrFormat` is wrapped by all three
   blob loaders (`Load`, `LoadFlatI8`, `LoadFlatI8Mmap` — the last via the shared
   `flatI8Layout`, so its I/O errors stay un-tagged), and `embed.ErrFormat` by
   `OpenSafetensors*` / `OpenGGUF*` for bad-magic / unsupported-version / truncated
   blobs. Callers `errors.Is(err, ann.ErrFormat)` instead of string-matching. Both
   are additive (apidiff: `ErrFormat: added`); per-tensor lookups and mmap I/O are
   deliberately *not* wrapped. (Chose one `ErrFormat` per package over a magic-only
   `ErrBadMagic` since it also covers version + truncation.)
2. **Scope the global knobs** (`linalg.SetParallelThreshold/Width`) — ✅ **DONE.**
   The threshold is now scoped into `Workspace` (`SetThreshold`) alongside the
   pre-existing width scoping (`SetWorkers`), with the globals as process-wide
   defaults a zero-value `Workspace` inherits. The W8A8 hot path (`MatmulBTW8A8Into`/
   `Batch`) reads the scoped threshold, and the f32 matmuls gained `Workspace`
   methods (`(*Workspace).MatmulBT` / `MatmulBTAcc64`) — so independent decode
   streams tune parallelism without racing on a global. Bit-identical (parallelism
   is numerically inert), race-clean across concurrent Workspaces, additive (apidiff:
   3 methods added). Remaining matmul variants (`MatmulBTQ8/W4A8/Q4`) can gain
   `Workspace` methods additively if needed — the shape is settled, so no longer a
   pre-graduation break risk.
3. **Decide the worker pool's fate** — [low / low]. Still built, still
   measured-neutral, still shipped-but-unused. Delete it (the measurement
   note in git history is the record) or mark deprecated. Pick one this
   cycle.
4. **`linalg` surface audit before graduation** — [low-medium / low]. *New:*
   the Experimental surface keeps growing opportunistically (`DotI8` exported
   for int8 HNSW, `MatmulBTAcc64` for goinfer's MoE router). Each is justified,
   but graduation-to-Hard gets harder with every export — do a deliberate
   keep/unexport pass (with goinfer's actual usage as the consumer evidence)
   before promising semver on any of it.

## 5. Watch list — externally gated, with explicit triggers

| Item | Gate | Trigger to act |
|---|---|---|
| VNNI `VPDPBUSD` W4A8 variant | no Zen 4+/Cascade Lake+ box | hardware access (a cloud c7i/c7a spot instance is a cheap unblock — consider renting instead of waiting) |
| AVX-512 dispatch tier | same hardware gap + downclocking caveats | same box; bundle with VNNI work |
| `GOEXPERIMENT=simd` portable kernels | archsimd is amd64-only + experiment-gated (confirmed still true, June 2026; arm64 planned via SVE on dev.simd for 1.27) | BOTH: arm64 support lands AND experiment graduates. Then re-run the spike; if within ~10% of hand asm, migrate and delete `.s` files. Note: when this graduates, hugot's pure-Go backend gets its predicted 3–10× — aikit's kernel lead narrows, which is *also* why §1.1's benchmark table should publish now |
| In-place index mutation | design rule 4 (immutability cornerstone) | a real consumer demonstrating rebuild/delta/swap insufficiency — not before |
| AMX | out of scope | — |

---

## Competitive context (refreshed 2026-06-09)

- **hugot** (knights-analytics) — *moved the most since v1.* Now ships
  CrossEncoders (reranking — direct overlap with `encoder`), text-generation
  and image-classification pipelines; pure-Go backend targets exactly the
  MiniLM-class models §2.2 would add; publicly expecting 3–10× from Go SIMD.
  Still cgo (ONNX Runtime) for full speed. aikit's answers: no-cgo + hand
  kernels today (§1.1: publish the comparison), MiniLM parity (§2.2), and
  the embedded-binary pattern hugot's ONNX dependency can't follow (§1.2).
- **Antfly/Termite** — core rewritten in **Zig** (May 2026); the Go engine
  story is now historical. Still the feature bar for hybrid retrieval
  (sparse + dense + lexical + published recall numbers) — aikit now matches
  the retrieval-feature list in library form except SPLADE inference (§2.3).
  Their exit from Go arguably *widens* aikit's lane: one fewer serious
  pure-Go retrieval engine.
- **coder/hnsw** — measured, not just claimed, as of v1.3.0: the benchmarks
  module puts it at recall@10 ~0.22 on real embeddings (construction-limited;
  plain selection vs Alg-4) vs aikit's ~0.995. The README table makes the
  comparison legible; keep it honest if upstream improves.
- **Bleve / chromem-go / sqlite-vec / LanceDB-go** — index-only; no embedded
  inference. Unchanged moat.
- **Ollama** — unchanged: server dependency, awkward rerankers. aikit's
  pitch stays in-process, zero-deploy.
- **Model2Vec upstream (MinishLab)** — potion-retrieval-32M remains the best
  static retrieval model; §2.4 pins it. Parity discipline keeps inheriting
  upstream gains cheaply.
- **Go 1.26 archsimd** — amd64-only, GOEXPERIMENT-gated; arm64 (SVE) and a
  portable high-level API are explicitly future work on dev.simd. The
  watch-list trigger stands.
