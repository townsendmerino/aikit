package ann_test

import (
	"fmt"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/fuse"
)

// Serving a changing corpus WITHOUT mutating an index (design rule 4): keep a big
// immutable base index plus a small, frequently-rebuilt delta of recent docs,
// query both, and fuse the rankings. Each index's local Hit.Index is mapped to a
// global doc id; fuse.RRF merges them into one ranking. Periodically fold the
// delta into a fresh base and swap.
func Example_baseDeltaFusion() {
	base := ann.New([][]float32{ // global ids 0..2 (the established corpus)
		{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0},
	})
	const baseN = 3
	delta := ann.New([][]float32{ // global ids 3..4 (just added)
		{0, 0, 0, 1}, {0.7071, 0.7071, 0, 0},
	})

	q := []float32{0, 1, 0, 0}
	fused := fuse.RRF(60,
		fuse.Keys(base.Query(q, 10), func(h ann.Hit) int { return h.Index }),
		fuse.Keys(delta.Query(q, 10), func(h ann.Hit) int { return baseN + h.Index }),
	)
	fmt.Printf("fused %d docs from base+delta; top global id %d\n", len(fused), fused[0].Key)
	// Output:
	// fused 5 docs from base+delta; top global id 1
}
