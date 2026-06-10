package bm25_test

import (
	"fmt"

	"github.com/townsendmerino/aikit/bm25"
)

// A general-text BM25 index: tokenize prose with TokenizePlain (no identifier
// splitting), then Build / Query as usual. Use Tokenize instead for code corpora.
func ExampleTokenizePlain() {
	docs := [][]string{
		bm25.TokenizePlain("The cat sat on the mat."),
		bm25.TokenizePlain("Dogs are loyal companions."),
		bm25.TokenizePlain("A cat and a dog played together."),
	}
	ix := bm25.Build(docs)

	for _, r := range ix.TopK(bm25.TokenizePlain("cat"), 2) {
		fmt.Println("doc", r.Doc)
	}
	// Output:
	// doc 0
	// doc 2
}
