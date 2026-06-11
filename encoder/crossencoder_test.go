package encoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestCrossEncoder_parity pins the reranker (§1 v3) against the Python golden
// (scripts/pin_crossencoder.py): feeding the golden's input_ids + token_type_ids,
// scoreIDs must reproduce the classification logit, and the live Score(query, doc)
// pipeline (aikit's own pair tokenization) must match too. Model-gated.
func TestCrossEncoder_parity(t *testing.T) {
	const dir = "../testdata/crossencoder-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skip("testdata/crossencoder-model/ not present; see scripts/README.md")
	}
	ce, err := LoadCrossEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../testdata/crossencoder_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		Labels int `json:"labels"`
		Cases  []struct {
			Query    string    `json:"query"`
			Doc      string    `json:"doc"`
			InputIDs []int32   `json:"input_ids"`
			TypeIDs  []int32   `json:"token_type_ids"`
			Score    []float32 `json:"score"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if ce.labels != g.Labels {
		t.Fatalf("labels %d != golden %d", ce.labels, g.Labels)
	}

	var worstFwd, worstLive float64
	for _, c := range g.Cases {
		// (1) forward parity: feed the golden's exact ids + segments.
		got := ce.scoreIDs(c.InputIDs, c.TypeIDs)
		for l := range got {
			if d := math.Abs(float64(got[l]) - float64(c.Score[l])); d > worstFwd {
				worstFwd = d
			}
		}
		// (2) end-to-end: aikit's own pair tokenization → Score.
		live, err := ce.Score(c.Query, c.Doc)
		if err != nil {
			t.Fatal(err)
		}
		if d := math.Abs(float64(live) - float64(c.Score[0])); d > worstLive {
			worstLive = d
		}
		t.Logf("%-30q | %-30q score go %.4f / live %.4f / py %.4f", trunc(c.Query), trunc(c.Doc), got[0], live, c.Score[0])
	}
	t.Logf("cross-encoder parity: worst forward Δ %.2e, worst end-to-end Δ %.2e", worstFwd, worstLive)
	if worstFwd > 5e-3 {
		t.Errorf("forward score max Δ %.2e vs golden", worstFwd)
	}
	if worstLive > 5e-3 {
		t.Errorf("end-to-end score max Δ %.2e (tokenization mismatch?)", worstLive)
	}
}

func trunc(s string) string {
	if len(s) > 28 {
		return s[:28]
	}
	return s
}

// TestCrossEncoder_reranks shows the headline use: scoring a query against a
// candidate list ranks the relevant document first.
func TestCrossEncoder_reranks(t *testing.T) {
	const dir = "../testdata/crossencoder-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skip("no cross-encoder model")
	}
	ce, err := LoadCrossEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	query := "how do i read a file in go"
	cands := []string{
		"the eiffel tower is in paris",
		"os.Open opens a file; bufio.Scanner reads it line by line",
		"a recipe for chocolate chip cookies",
	}
	best, bestScore := -1, float32(math.Inf(-1))
	for i, d := range cands {
		s, err := ce.Score(query, d)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("  %.3f  %s", s, d)
		if s > bestScore {
			best, bestScore = i, s
		}
	}
	if best != 1 {
		t.Errorf("expected the os.Open doc (1) to rank first, got %d", best)
	}
}
