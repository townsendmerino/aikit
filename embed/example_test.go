package embed_test

import (
	"fmt"

	"github.com/townsendmerino/aikit/embed"
)

// Load a Model2Vec checkpoint and embed text. The returned vector is
// L2-normalized — ready to hand straight to ann.New / Flat.Query.
//
// This example needs a model on disk, so it is illustrative (compiled, not
// run). See the repo README for fetching a checkpoint.
func Example() {
	m, err := embed.Load("path/to/model2vec-snapshot")
	if err != nil {
		return
	}
	vec := m.Encode("recursive directory walk that respects gitignore")
	fmt.Println(len(vec)) // == m.Dim()
}
