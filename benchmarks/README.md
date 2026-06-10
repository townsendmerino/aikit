# aikit benchmarks

A reproducible, apples-to-apples comparison of aikit's dense indexes against
pure-Go ANN libraries, plus a capability matrix for the wider field.

This is a **separate module** (its own `go.mod`, `replace github.com/townsendmerino/aikit => ../`)
so the competitors' dependency trees never touch aikit's pure-Go core graph — the
root module's `CGO_ENABLED=0` build and "no cgo deps" CI check stop at this module
boundary.

## Running

```bash
cd benchmarks
GOWORK=off go run .          # the comparison table
GOWORK=off go test -run Sweep -v   # the coder/hnsw recall-vs-tuning sweeps
```

It needs the Model2Vec model in `../testdata/model` (the repo README's
`huggingface-cli download minishlab/potion-code-16M …` step) — see "Why real
embeddings" below.

## Methodology

Every library is wrapped in one `idx` interface (`build` + `query`) and driven
through the **same** measurement code: the same corpus, the same queries, and the
same independently-computed exact top-k ground truth. So:

- **Recall@10** is measured against brute-force exact cosine — even aikit Flat and
  chromem-go are *measured* (they score 1.0000, validating the harness), not
  assumed.
- **Latency** percentiles are wall-clock per query, single-threaded query loop.
- **Memory** is the heap-allocation delta around `build` (GC'd before and after) —
  a proxy for index footprint. aikit Flat/FlatI8 index the input vectors in place
  (no copy), so their delta reflects only what they add; coder/hnsw and chromem-go
  copy the vectors, so theirs includes a copy. That difference is real (aikit's
  zero-copy indexing is a genuine memory advantage), not an artifact.
- **Config**: `M=16`, `EfSearch=64`, `k=10`, N=8000 docs, dim 256. aikit HNSW uses
  `EfConstruction=200`; coder/hnsw has no separate construction-ef knob (see below).

### Why real embeddings (not synthetic vectors)

The first cut used synthetic vectors and produced nonsense — *both* HNSW libraries
scored ~0.55 recall. The cause is fundamental: synthetic data can't measure
recall@k. Random high-dim vectors concentrate distances (the k-th and (k+1)-th
neighbors are near-tied), and Gaussian clusters make the exact top-k an arbitrary
choice among near-duplicates. Either way the ground-truth top-k is unstable, so no
ANN can score well and the number says nothing about implementation quality. This
is why ann-benchmarks uses real datasets. The harness embeds deterministically
generated code-ish phrases with Model2Vec (potion-code-16M) — real embeddings have
the stable, well-separated neighbor structure recall@k needs.

## Results

Real Model2Vec embeddings, N=8000, dim=256, M=16, EfSearch=64, k=10:

| index | recall@10 | build | p50 | p95 | mem | notes |
|---|---|---|---|---|---|---|
| aikit Flat | 1.0000 | 0 ms | 0.28 ms | 0.32 ms | ~0 MB | exact, zero-copy |
| **aikit HNSW** | **0.9950** | 4.5 s | 0.085 ms | 0.12 ms | ~2 MB | Alg-4 diversity heuristic |
| **aikit FlatI8** | **0.9952** | 12 ms | 0.13 ms | 0.14 ms | ~2 MB | int8, ¼ memory |
| coder/hnsw | 0.2198 | 1.9 s | 0.058 ms | 0.077 ms | ~8 MB | see below |
| chromem-go | 1.0000 | 13 ms | 3.77 ms | 4.08 ms | ~4 MB | exact brute-force |

(Numbers vary run to run — coder/hnsw uses its default non-seeded RNG, and timings
are machine-dependent; recall and the relative ordering are stable. Measured on an
Apple M-series.)

### Reading the table

- **aikit FlatI8 is the headline**: 0.995 recall at exact-search latency and ¼ the
  float32 memory. For a repo-scale embedded corpus it's hard to beat.
- **chromem-go** is exact (recall 1.0) but ~45× the p50 of aikit Flat — it's a
  brute-force scan without aikit's SIMD-blocked kernel.
- **coder/hnsw recall (0.22) is real and fair, not a misuse.** Verified: canonical
  API (matches coder's own tests), correct cosine distance, returns a full k=10,
  self-query returns itself, finds the right neighborhood (the top result is
  correct). The gap is **structural**: coder's recall is flat across search-ef
  (64→800) and only crawls up with M (0.22→0.39 from M=16→64, at multiplying memory
  and latency), never approaching aikit. coder uses plain greedy neighbor selection;
  aikit defaults to the **Algorithm-4 diversity heuristic**, which exists precisely
  because plain selection under-explores clustered real embeddings — aikit's own
  measurements show the same effect (plain selection capped recall@10 at ~0.68
  internally; the heuristic took it to ~1.00). `go test -run Sweep -v` reproduces
  the ef and M sweeps. coder/hnsw remains a lean, fast graph with a simple API and
  import/export — recall on clustered embeddings is just not where it wins.

## Not benchmarked here (different category or cgo)

A head-to-head latency row would be unfair for these, so they're in the capability
matrix instead:

- **Bleve** — a full-text search engine; its dense vector search requires a cgo
  faiss backend (build-tagged), so it's not comparable in the pure-Go lane and
  pulls heavy native deps.
- **hugot** — an *inference* library (Hugging Face pipelines: embedders, rerankers,
  classifiers), not an index. Real speed needs ONNX Runtime (cgo); its pure-Go
  backend is limited. The apt aikit comparison is `embed`/`encoder`, not the index.

See the capability matrix in the root [README](../README.md#how-aikit-compares).
