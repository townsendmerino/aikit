package ann_test

import (
	"fmt"

	"github.com/townsendmerino/aikit/ann"
)

// FlatI8 is a drop-in for Flat that stores vectors as int8 (¼ the memory) and
// scores via the int8 W8A8 kernel — for embedded / RAM-constrained retrieval,
// at a small recall cost. Same Hit / Query(q, k) shape, so it also feeds fuse.RRF.
func ExampleFlatI8() {
	vecs := [][]float32{
		{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1},
	}
	ix := ann.NewFlatI8(vecs) // stored as int8 + per-vector scales

	hits := ix.Query([]float32{0.1, 0.2, 0.95, 0.1}, 1)
	fmt.Println("nearest doc:", hits[0].Index)
	// Output:
	// nearest doc: 2
}
