package topk_test

import (
	"fmt"

	"github.com/townsendmerino/aikit/topk"
)

// Keep the K highest-scoring items from a stream without sorting everything —
// an O(N log K) bounded min-heap.
func Example() {
	sel := topk.New[string](2) // retain the top 2
	sel.Push("alpha", 0.5)
	sel.Push("beta", 0.9)
	sel.Push("gamma", 0.1)

	for _, r := range sel.Result() { // descending by score
		fmt.Printf("%s %.1f\n", r.Item, r.Score)
	}
	// Output:
	// beta 0.9
	// alpha 0.5
}
