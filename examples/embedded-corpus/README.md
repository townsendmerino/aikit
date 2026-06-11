# embedded-corpus — a zero-deploy hybrid-search binary

A single self-contained Go binary that does **hybrid semantic search** — dense
(int8 ANN) + lexical (BM25) fused with RRF — over a corpus of **Go standard-library
and aikit documentation**. The Model2Vec model, a prebuilt int8 index, and the
corpus are **all `//go:embed`-ed**, so the binary needs *zero external files* at
runtime and starts in ~50 ms.

This is the `//go:embed`-a-corpus, zero-deploy pattern no Python or ONNX-runtime
stack reaches: copy one binary to a box (or a scratch container, or a Lambda) and it
answers queries — no model server, no vector DB, no sidecar files.

```
$ ./embedded-corpus "how do i read a file line by line in go"
ready in 50ms — 1747 chunks, int8 index 443 KB, model+index+corpus all embedded, zero external files

"how do i read a file line by line in go"
  1. [stdlib      bufio   ] func (b *Reader) ReadLine() (line []byte, isPrefix bool, err error)
  2. [stdlib      bufio   ] func (b *Reader) ReadSlice(delim byte) (line []byte, err error)
  3. [stdlib      os      ] func (f *File) Read(b []byte) (n int, err error)
```

## Run it

The model (~61 MB) is fetched on demand and embedded at build — it is not committed.
You need `huggingface-cli` (`pip install -U huggingface_hub`) once:

```sh
cd examples/embedded-corpus
go generate            # fetch potion-code-16M + rebuild the int8 index from the corpus
go build               # → one ~70 MB self-contained binary
./embedded-corpus "quantize float32 vectors to int8"      # one-shot
./embedded-corpus                                         # interactive
```

Without `go generate` the binary still compiles (a placeholder keeps `//go:embed`
happy) but reports that the model needs fetching.

## What's embedded, and how big

| asset | size | notes |
|---|---|---|
| Model2Vec model (potion-code-16M) | ~61 MB | dominates the binary; embeds queries at runtime |
| `assets/index.bin` (FlatI8) | ~443 KB | 1747 chunks × 256-d, int8 (¼ the float32 size) |
| `assets/corpus.json` | ~690 KB | the chunk texts (also rebuilds BM25 at startup) |
| aikit retrieval code | ~few MB | `embed` + `ann` + `bm25` + `fuse`, cgo-free |

The model is the only large part; the *aikit* surface that makes it searchable —
embedding, the int8 index, lexical BM25, and fusion — is a few MB of pure Go.

## How it works

Two phases:

- **Build (`gen/`, run by `go generate`)** — extract a doc-oriented corpus: Go stdlib
  package docs via `go doc -all`, aikit package docs, and aikit's markdown; embed each
  chunk with the Model2Vec model; quantize the matrix to a `FlatI8` index and write
  `assets/index.bin` + `assets/corpus.json`.
- **Run (`main.go`)** — `//go:embed` the model, index, and corpus; load the model
  (`embed.LoadFromFS`), the int8 index (`ann.LoadFlatI8`), and rebuild BM25 from the
  corpus (it's cheap, so it has no on-disk form); then for each query: embed it,
  `FlatI8.Query` (dense) + `bm25.TopK` (lexical), and `fuse.RRF` the two rankings.

Swap-ins for other shapes: a very large corpus → int8 **HNSW** (`ann.Config{Int8:true}`)
instead of the exact FlatI8 scan; not wanting the index *in* the binary →
`ann.LoadFlatI8Mmap` over a sidecar file (zero-copy, OS-page-cached).
