package encoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/sparse"
)

// TestSPLADE_endToEnd shows the closed loop: SPLADE expansions feed the sparse
// inverted index directly, and a learned-sparse query ranks the relevant doc first
// — in-process, no Python. Model-gated.
func TestSPLADE_endToEnd(t *testing.T) {
	const dir = "../testdata/splade-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skip("testdata/splade-model/ not present; see scripts/README.md")
	}
	s, err := LoadSPLADE(dir)
	if err != nil {
		t.Fatal(err)
	}
	docs := []string{
		"def parse_json(s):\n    return json.loads(s)",
		"train a deep neural network with gradient descent",
		"the cat sat quietly on the warm mat",
	}
	dvecs := make([]sparse.SparseVec, len(docs))
	for i, d := range docs {
		v, err := s.Expand(d)
		if err != nil {
			t.Fatal(err)
		}
		dvecs[i] = v
	}
	ix := sparse.New(dvecs)
	q, err := s.Expand("how to read and parse a json file in python")
	if err != nil {
		t.Fatal(err)
	}
	hits := ix.Query(q, len(docs))
	if len(hits) == 0 {
		t.Fatal("no hits")
	}
	t.Logf("SPLADE→sparse end-to-end: query ranks doc %d first (score %.3f)", hits[0].Index, hits[0].Score)
	if hits[0].Index != 0 {
		t.Errorf("expected the json doc (0) to rank first for a json query, got doc %d", hits[0].Index)
	}
}

// TestSPLADE_parity pins the SPLADE expansion (§2.3) against the Python reference
// (scripts/pin_splade.py): feeding the golden's input_ids, expandIDs must produce
// the same sparse term-weight vector (compared by cosine; term-set agreement logged).
// Model-gated.
func TestSPLADE_parity(t *testing.T) {
	const dir = "../testdata/splade-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skip("testdata/splade-model/ not present; see scripts/README.md")
	}
	s, err := LoadSPLADE(dir)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../testdata/splade_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		Vocab int `json:"vocab"`
		Cases []struct {
			Text     string    `json:"text"`
			InputIDs []int32   `json:"input_ids"`
			Terms    []uint32  `json:"terms"`
			Weights  []float32 `json:"weights"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	worstCos, worstJac := 1.0, 1.0
	for _, c := range g.Cases {
		v := s.expandIDs(c.InputIDs)
		got := map[uint32]float32{}
		for i, tm := range v.Terms {
			got[tm] = v.Weights[i]
		}
		want := map[uint32]float32{}
		for i, tm := range c.Terms {
			want[tm] = c.Weights[i]
		}
		union := map[uint32]bool{}
		for tm := range got {
			union[tm] = true
		}
		for tm := range want {
			union[tm] = true
		}
		var dot, ng, nw float64
		inter := 0
		for tm := range union {
			a, b := float64(got[tm]), float64(want[tm])
			dot += a * b
			ng += a * a
			nw += b * b
			if got[tm] > 0 && want[tm] > 0 {
				inter++
			}
		}
		cos := dot / (math.Sqrt(ng) * math.Sqrt(nw))
		jac := float64(inter) / float64(len(union))
		if cos < worstCos {
			worstCos = cos
		}
		if jac < worstJac {
			worstJac = jac
		}
		t.Logf("%-40q nnz go=%d py=%d  cosine %.6f  term-jaccard %.4f", c.Text, len(v.Terms), len(c.Terms), cos, jac)
		if cos < 0.999 {
			t.Errorf("%q: SPLADE cosine %.6f < 0.999", c.Text, cos)
		}
	}
	t.Logf("SPLADE parity: worst cosine %.6f, worst term-jaccard %.4f", worstCos, worstJac)
}
