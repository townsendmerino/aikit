package encoder_test

import (
	"fmt"

	"github.com/townsendmerino/aikit/encoder"
)

// Rerank candidates with the CodeRankEmbed transformer: encode the query (with
// the mandatory query prefix) and each candidate, then score by cosine of the
// returned hidden states. Higher-fidelity than the Model2Vec first stage —
// the typical two-stage retrieve-then-rerank setup.
//
// Needs a checkpoint on disk, so this example is illustrative (compiled, not
// run). See the repo README for fetching CodeRankEmbed.
func Example() {
	m, err := encoder.Load("path/to/coderankembed-snapshot")
	if err != nil {
		return
	}
	query, _ := m.Encode("how do I parse json", true) // isQuery = true
	doc, _ := m.Encode("func parseJSON(b []byte) (*Config, error) { ... }", false)
	_, _ = query, doc

	fmt.Println(m.HiddenDim())
}
