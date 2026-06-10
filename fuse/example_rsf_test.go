package fuse_test

import (
	"fmt"

	"github.com/townsendmerino/aikit/fuse"
)

// Relative Score Fusion blends a dense (cosine) and a lexical (BM25) ranking by
// normalized score rather than rank — use it when the per-list scores are
// calibrated. Pair it with fuse.Scores to project typed result lists.
func ExampleRSF() {
	dense := []fuse.Scored[int]{{Key: 1, Score: 0.92}, {Key: 2, Score: 0.55}, {Key: 3, Score: 0.10}}
	lexical := []fuse.Scored[int]{{Key: 2, Score: 8.0}, {Key: 3, Score: 5.0}, {Key: 4, Score: 1.0}}

	for _, r := range fuse.RSF(dense, lexical) {
		fmt.Printf("doc %d  %.2f\n", r.Key, r.Score)
	}
	// Output:
	// doc 2  1.55
	// doc 1  1.00
	// doc 3  0.57
	// doc 4  0.00
}
