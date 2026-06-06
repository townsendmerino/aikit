package fuse_test

import (
	"fmt"

	"github.com/townsendmerino/aikit/fuse"
)

// Reciprocal-rank fusion blends a lexical (BM25) and a dense (ANN) ranking
// without reconciling their incomparable score scales — only rank order
// matters. An item ranked well by both is pulled to the top.
func Example() {
	lexical := []int{3, 1, 2} // doc ids, best first
	dense := []int{1, 3, 5}

	fused := fuse.RRF(fuse.DefaultK, lexical, dense)

	ids := make([]int, len(fused))
	for i, r := range fused {
		ids[i] = r.Key
	}
	fmt.Println(ids) // 3 and 1 score highest (ranked well by both)
	// Output:
	// [3 1 2 5]
}
