// Command rag is the end-to-end aikit retrieval pipeline in one file:
//
//	chunk → embed → (ANN + BM25) → RRF fuse → encoder rerank → top-K
//
// It indexes a tiny in-memory corpus, runs a hybrid (lexical + dense) search,
// fuses the two rankings with reciprocal-rank fusion, and reranks the fused
// shortlist with the CodeRankEmbed encoder for the final order.
//
// It needs two local models (skipped-clean if absent, so `go build ./...`
// always compiles and `go run` without flags just prints guidance):
//
//	go run ./examples/rag \
//	    --embed-model  testdata/model \
//	    --rerank-model testdata/encoder-model \
//	    --q "read a file line by line"
//
// See the repo README for how to fetch the models.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/bm25"
	"github.com/townsendmerino/aikit/chunk"
	_ "github.com/townsendmerino/aikit/chunk/regex" // registers the "regex" chunker via init()
	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/encoder"
	"github.com/townsendmerino/aikit/fuse"
	"github.com/townsendmerino/aikit/topk"
)

// A small, deliberately varied corpus. Each entry is a "file"; the chunker
// splits it into indexable units. The query below should surface the
// file-reading snippet over the unrelated ones.
var corpus = []struct{ name, src string }{
	{"readlines.go", "func readLines(path string) ([]string, error) {\n\tf, err := os.Open(path)\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\tdefer f.Close()\n\tvar lines []string\n\ts := bufio.NewScanner(f)\n\tfor s.Scan() {\n\t\tlines = append(lines, s.Text())\n\t}\n\treturn lines, s.Err()\n}"},
	{"json.go", "func parseConfig(b []byte) (*Config, error) {\n\tvar c Config\n\tif err := json.Unmarshal(b, &c); err != nil {\n\t\treturn nil, fmt.Errorf(\"parse config: %w\", err)\n\t}\n\treturn &c, nil\n}"},
	{"server.go", "func handler(w http.ResponseWriter, r *http.Request) {\n\tw.Header().Set(\"Content-Type\", \"application/json\")\n\tjson.NewEncoder(w).Encode(map[string]string{\"ok\": \"true\"})\n}"},
	{"math.go", "func fib(n int) int {\n\tif n < 2 {\n\t\treturn n\n\t}\n\treturn fib(n-1) + fib(n-2)\n}"},
	{"hash.go", "func sha256File(path string) (string, error) {\n\tf, err := os.Open(path)\n\tif err != nil {\n\t\treturn \"\", err\n\t}\n\tdefer f.Close()\n\th := sha256.New()\n\tif _, err := io.Copy(h, f); err != nil {\n\t\treturn \"\", err\n\t}\n\treturn hex.EncodeToString(h.Sum(nil)), nil\n}"},
}

func main() {
	embedDir := flag.String("embed-model", "", "dir with a Model2Vec checkpoint for embed.Load")
	rerankDir := flag.String("rerank-model", "", "dir with a CodeRankEmbed checkpoint for encoder.Load")
	query := flag.String("q", "read a file line by line", "search query")
	shortlist := flag.Int("shortlist", 20, "candidates each retriever contributes to the fuse")
	rerankN := flag.Int("rerank", 8, "fused candidates to rerank with the encoder")
	flag.Parse()

	if *embedDir == "" || *rerankDir == "" {
		fmt.Println(`rag — end-to-end aikit retrieval example.

Needs two local models:
  --embed-model  <dir>   Model2Vec       (e.g. testdata/model)
  --rerank-model <dir>   CodeRankEmbed    (e.g. testdata/encoder-model)

Without them this just prints guidance; the pipeline code below is the point.`)
		return
	}

	em, err := embed.Load(*embedDir)
	check(err, "load embed model")
	enc, err := encoder.Load(*rerankDir)
	check(err, "load rerank model")

	// 1) CHUNK — split each file into indexable units. One flat slice; its
	//    index is the shared id space for BM25 (Result.Doc) and ANN (Hit.Index).
	var chunks []chunk.Chunk
	for _, d := range corpus {
		cs, err := chunk.ChunkFile("regex", d.name, []byte(d.src), 60)
		check(err, "chunk "+d.name)
		chunks = append(chunks, cs...)
	}

	// 2) EMBED + index for dense (semantic) search. embed.Encode returns an
	//    L2-normalized vector, which is ann's input contract.
	vecs := make([][]float32, len(chunks))
	for i, c := range chunks {
		vecs[i] = em.Encode(c.Text)
	}
	dense := ann.New(vecs)

	// 2b) Build the BM25 lexical index over the same chunks (same order).
	docs := make([][]string, len(chunks))
	for i, c := range chunks {
		docs[i] = bm25.Tokenize(c.Text)
	}
	lexical := bm25.Build(docs)

	// 3) RETRIEVE both ways, then FUSE the rankings. RRF is rank-based, so it
	//    blends BM25's unbounded scores with cosine [-1,1] without normalizing.
	lexHits := lexical.TopK(bm25.Tokenize(*query), *shortlist)
	denHits := dense.Query(em.Encode(*query), *shortlist)
	fused := fuse.RRF(fuse.DefaultK,
		fuse.Keys(lexHits, func(r bm25.Result) int { return r.Doc }),
		fuse.Keys(denHits, func(h ann.Hit) int { return h.Index }),
	)

	// 4) RERANK the fused shortlist with the encoder: encode the query and the
	//    candidates, score by cosine, and keep the top-K via topk.
	n := min(*rerankN, len(fused))
	cand := fused[:n]
	texts := make([]string, n)
	isQuery := make([]bool, n) // all docs
	for i, r := range cand {
		texts[i] = chunks[r.Key].Text
	}
	qEmb, err := enc.Encode(*query, true)
	check(err, "encode query")
	dEmbs, err := enc.EncodeBatch(texts, isQuery, 0)
	check(err, "encode candidates")

	sel := topk.New[int](n)
	for i, r := range cand {
		sel.Push(r.Key, cosine(qEmb, dEmbs[i]))
	}

	// 5) Final ranked output.
	fmt.Printf("query: %q\n\n", *query)
	for rank, hit := range sel.Result() {
		c := chunks[hit.Item]
		fmt.Printf("%d. %.4f  %s:%d-%d\n     %s\n", rank+1, hit.Score, c.File, c.StartLine, c.EndLine, firstLine(c.Text))
	}
}

// cosine similarity in float64 (the encoder returns raw, un-normalized CLS).
func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func firstLine(s string) string {
	for i, r := range s {
		if r == '\n' {
			return s[:i]
		}
	}
	return s
}

func check(err error, what string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "rag: %s: %v\n", what, err)
		os.Exit(1)
	}
}
