// Command embedded-corpus is a self-contained hybrid-search showcase. The Model2Vec
// model, a prebuilt int8 index, and the corpus are all //go:embed-ed, so a single
// static binary answers Go / aikit questions over dense (int8 ANN) + lexical (BM25)
// retrieval fused with RRF — with no external files and effectively instant startup.
// This is the //go:embed-a-corpus, zero-deploy pattern no Python or ONNX stack reaches.
//
//go:generate sh -c "test -f assets/model/model.safetensors || huggingface-cli download minishlab/potion-code-16M tokenizer.json config.json model.safetensors --local-dir assets/model"
//go:generate go run ./gen
package main

import (
	"bufio"
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/bm25"
	m2v "github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/fuse"
)

//go:embed assets/model
var modelFS embed.FS

//go:embed assets/index.bin
var indexBlob []byte

//go:embed assets/corpus.json
var corpusJSON []byte

// Chunk mirrors gen.Chunk; corpus.json is parallel to the index rows.
type Chunk struct {
	Source string `json:"source"`
	Ref    string `json:"ref"`
	Text   string `json:"text"`
}

func main() {
	topN := flag.Int("n", 5, "number of results to show")
	flag.Parse()

	start := time.Now()
	model, err := m2v.LoadFromFS(modelFS, "assets/model")
	if err != nil {
		die("load embedded model: %v", err)
	}
	index, err := ann.LoadFlatI8(indexBlob)
	if err != nil {
		die("load embedded index: %v", err)
	}
	var corpus []Chunk
	if err := json.Unmarshal(corpusJSON, &corpus); err != nil {
		die("parse embedded corpus: %v", err)
	}
	// BM25 has no on-disk form (it's cheap to build) — rebuild it from the embedded
	// corpus at startup so the lexical half needs nothing external either.
	docs := make([][]string, len(corpus))
	for i, c := range corpus {
		docs[i] = bm25.TokenizePlain(c.Text)
	}
	lexIndex := bm25.Build(docs)

	fmt.Fprintf(os.Stderr,
		"ready in %s — %d chunks, int8 index %d KB, model+index+corpus all embedded, zero external files\n",
		time.Since(start).Round(time.Microsecond), len(corpus), len(indexBlob)/1024)

	search := func(query string) {
		qv := model.Encode(query)
		dense := index.Query(qv, 50)                        // []ann.Hit  (dense, int8 ANN)
		lex := lexIndex.TopK(bm25.TokenizePlain(query), 50) // []bm25.Result (lexical)
		fused := fuse.RRF(fuse.DefaultK,                    // rank-fuse the two
			fuse.Keys(dense, func(h ann.Hit) int { return h.Index }),
			fuse.Keys(lex, func(r bm25.Result) int { return r.Doc }))
		fmt.Printf("\n%q\n", query)
		for i, f := range fused {
			if i >= *topN {
				break
			}
			c := corpus[f.Key]
			fmt.Printf("  %d. [%-11s %-18s] %s\n", i+1, c.Source, c.Ref, firstLine(c.Text))
		}
	}

	if args := flag.Args(); len(args) > 0 {
		search(strings.Join(args, " "))
		return
	}
	fmt.Println("\nAsk a Go / aikit question (Ctrl-D to quit):")
	sc := bufio.NewScanner(os.Stdin)
	fmt.Print("\n> ")
	for sc.Scan() {
		if q := strings.TrimSpace(sc.Text()); q != "" {
			search(q)
		}
		fmt.Print("\n> ")
	}
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if len(s) > 96 {
		s = s[:96] + "…"
	}
	return s
}

func die(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "embedded-corpus: "+format+"\n", a...)
	os.Exit(1)
}
