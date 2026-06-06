package bm25_test

import (
	"fmt"

	"github.com/townsendmerino/aikit/bm25"
)

// Identifier-aware lexical search: Tokenize splits camelCase / snake_case so a
// query for "read file" matches the chunk containing readFile.
func Example() {
	docs := [][]string{
		bm25.Tokenize("func readFile reads a file from disk"),
		bm25.Tokenize("func writeFile writes a file to disk"),
		bm25.Tokenize("type Config holds settings"),
	}
	ix := bm25.Build(docs)

	hits := ix.TopK(bm25.Tokenize("read file"), 1)
	fmt.Println("top doc:", hits[0].Doc)
	// Output:
	// top doc: 0
}
