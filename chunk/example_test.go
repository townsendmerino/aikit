package chunk_test

import (
	"fmt"

	"github.com/townsendmerino/aikit/chunk"
	_ "github.com/townsendmerino/aikit/chunk/regex" // registers the "regex" chunker via init()
)

// Chunk a source file by name. The byte-fidelity invariant holds for every
// chunker: concatenating the chunks' Text reproduces the source exactly.
func Example() {
	src := []byte("package main\n\nfunc main() {\n\tprintln(\"hi\")\n}\n")

	chunks, err := chunk.ChunkFile("regex", "main.go", src, 60)
	if err != nil {
		panic(err)
	}

	var joined []byte
	for _, c := range chunks {
		joined = append(joined, c.Text...)
	}
	fmt.Println("got chunks:", len(chunks) >= 1)
	fmt.Println("byte-faithful:", string(joined) == string(src))
	// Output:
	// got chunks: true
	// byte-faithful: true
}
