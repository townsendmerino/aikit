package ann_test

import (
	"fmt"

	"github.com/townsendmerino/aikit/ann"
)

// Persist a built HNSW index and reload it — the //go:embed-an-index pattern:
// build the graph once offline, write MarshalBinary's bytes to a file (or embed
// them in the binary), and Load them at startup instead of rebuilding per process.
func ExampleHNSW_MarshalBinary() {
	vecs := [][]float32{
		{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1},
	}
	ix := ann.BuildHNSW(vecs, ann.Config{Seed: 1})

	blob, err := ix.MarshalBinary() // os.WriteFile("index.hnsw", blob, 0o644), or //go:embed
	if err != nil {
		panic(err)
	}

	loaded, err := ann.Load(blob)
	if err != nil {
		panic(err)
	}
	hits := loaded.Query([]float32{0.1, 0.1, 0.9, 0.1}, 1)
	fmt.Println("nearest doc:", hits[0].Index)
	// Output:
	// nearest doc: 2
}
