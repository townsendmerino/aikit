package sparse_test

import (
	"fmt"

	"github.com/townsendmerino/aikit/sparse"
)

// Learned-sparse (SPLADE-style) retrieval over pre-computed sparse vectors: an
// inverted index scored by sparse dot product. The vectors here are hand-built;
// in practice they come from a SPLADE-family model's expansion of each document.
func ExampleIndex_Query() {
	docs := []sparse.SparseVec{
		{Terms: []uint32{1, 5, 9}, Weights: []float32{0.8, 1.2, 0.3}}, // doc 0
		{Terms: []uint32{5, 7}, Weights: []float32{2.0, 0.5}},         // doc 1
		{Terms: []uint32{1, 2}, Weights: []float32{0.4, 1.0}},         // doc 2
	}
	ix := sparse.New(docs)

	// A query expansion weighting terms 5 and 1.
	q := sparse.SparseVec{Terms: []uint32{5, 1}, Weights: []float32{1.0, 0.5}}
	for _, h := range ix.Query(q, 2) {
		fmt.Printf("doc %d  score %.2f\n", h.Index, h.Score)
	}
	// Output:
	// doc 1  score 2.00
	// doc 0  score 1.60
}
