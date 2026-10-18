// gen writes job 5's two BM25 corpora as bm25.MarshalTokens blobs, so both arms of the A/B build from identical input:
// aikit's tree (every .go and .md file under -repo, 40-line chunks, bm25.Tokenize) and a seeded synthetic corpus.
package main

import (
	"flag"
	"fmt"
	"io/fs"
	"math/rand"
	"os"
	"path/filepath"
	"strings"

	"github.com/townsendmerino/aikit/bm25"
)

func main() {
	repo := flag.String("repo", "", "tree to chunk")
	outTree := flag.String("out-tree", "", "aikit-tree tokens blob")
	outSynth := flag.String("out-synth", "", "synthetic tokens blob")
	nSynth := flag.Int("synth-docs", 100000, "synthetic documents")
	flag.Parse()

	var tree [][]string
	err := filepath.WalkDir(*repo, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && (d.Name() == ".git" || d.Name() == "testdata" || d.Name() == "assets") {
			return filepath.SkipDir
		}
		if d.IsDir() || !(strings.HasSuffix(p, ".go") || strings.HasSuffix(p, ".md")) {
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		lines := strings.Split(string(b), "\n")
		for i := 0; i < len(lines); i += 40 {
			tree = append(tree, bm25.Tokenize(strings.Join(lines[i:min(i+40, len(lines))], "\n")))
		}
		return nil
	})
	check(err)
	write(*outTree, tree)

	vocab := make([]string, 50000)
	for i := range vocab {
		vocab[i] = fmt.Sprintf("w%05x", (i*2654435761)%1048573)
	}
	rng := rand.New(rand.NewSource(100000))
	z := rand.NewZipf(rng, 1.1, 2, uint64(len(vocab)-1))
	synth := make([][]string, *nSynth)
	for d := range synth {
		toks := make([]string, 64+rng.Intn(385))
		for i := range toks {
			toks[i] = vocab[z.Uint64()]
		}
		synth[d] = toks
	}
	write(*outSynth, synth)
}

func write(path string, docs [][]string) {
	b, err := bm25.MarshalTokens(docs)
	check(err)
	check(os.WriteFile(path, b, 0o644))
	n := 0
	for _, d := range docs {
		n += len(d)
	}
	fmt.Printf("wrote %s: %d documents, %d tokens, %d bytes\n", path, len(docs), n, len(b))
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
