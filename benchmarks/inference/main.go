// Command inference measures aikit's pure-Go all-MiniLM-L6-v2 throughput (the BERT
// bi-encoder), as the apples-to-apples baseline for an inference-side comparison vs
// hugot (both run the same checkpoint). Pure Go, no ONNX Runtime / GPU / native lib.
//
//	cd benchmarks && GOWORK=off go run ./inference
package main

import (
	"fmt"
	"os"
	"time"

	"github.com/townsendmerino/aikit/encoder"
)

var sentences = []string{
	"how do i read a file line by line in go",
	"the quick brown fox jumps over the lazy dog",
	"machine learning models learn representations from data",
	"what is the capital of france",
	"recursive directory walk that respects gitignore",
	"compute the sha256 hash of a file in python",
	"neural networks are trained with gradient descent",
	"a relational database stores rows in tables with a schema",
}

func main() {
	m, err := encoder.LoadBERT("../testdata/minilm-model")
	if err != nil {
		fmt.Fprintln(os.Stderr, "inference: load all-MiniLM-L6-v2:", err)
		os.Exit(1)
	}
	// warm up
	for _, s := range sentences {
		if _, err := m.Encode(s); err != nil {
			fmt.Fprintln(os.Stderr, "inference:", err)
			os.Exit(1)
		}
	}
	const reps = 400
	start := time.Now()
	n := 0
	for i := 0; i < reps; i++ {
		for _, s := range sentences {
			m.Encode(s)
			n++
		}
	}
	el := time.Since(start)
	fmt.Printf("aikit all-MiniLM-L6-v2 (pure-Go, no runtime): %d encodes in %s — %.0f texts/sec, %.2f ms/text\n",
		n, el.Round(time.Millisecond), float64(n)/el.Seconds(), float64(el.Microseconds())/float64(n)/1000)
}
