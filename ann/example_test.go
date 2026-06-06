package ann_test

import (
	"fmt"

	"github.com/townsendmerino/aikit/ann"
)

// Exact cosine search over a set of L2-normalized vectors (the normalization
// contract lives at the embed boundary; ann assumes unit vectors and scores by
// dot product, higher = better).
func Example() {
	vecs := [][]float32{
		{1, 0},     // 0
		{0, 1},     // 1
		{0.6, 0.8}, // 2 — unit length
	}
	idx := ann.New(vecs)

	for _, h := range idx.Query([]float32{1, 0}, 2) {
		fmt.Printf("%d %.2f\n", h.Index, h.Score)
	}
	// Output:
	// 0 1.00
	// 2 0.60
}
