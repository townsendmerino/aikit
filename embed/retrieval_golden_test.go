package embed

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestStaticModel_potionRetrieval pins embed's standard (no-mapping) Model2Vec path
// against minishlab/potion-retrieval-32M, whose safetensors holds only `embeddings`
// (no `mapping`/`weights`): token ids index rows directly, pooling is a plain mean.
// Golden from scripts/pin_retrieval.py; model-gated.
func TestStaticModel_potionRetrieval(t *testing.T) {
	const dir = "../testdata/retrieval-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skip("testdata/retrieval-model/ not present; see scripts/README.md")
	}
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load potion-retrieval-32M (no-mapping format): %v", err)
	}
	raw, err := os.ReadFile("../testdata/retrieval_model_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		Dim   int `json:"dim"`
		Cases []struct {
			Text string    `json:"text"`
			Emb  []float64 `json:"embedding"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if m.Dim() != g.Dim {
		t.Fatalf("dim %d != golden %d", m.Dim(), g.Dim)
	}

	norm64 := func(v []float64) float64 {
		var s float64
		for _, x := range v {
			s += x * x
		}
		return math.Sqrt(s)
	}
	worst := 1.0
	for _, c := range g.Cases {
		v := m.Encode(c.Text)
		if norm64(c.Emb) < 1e-6 { // degenerate empty input — both should be ~zero
			var s float64
			for _, x := range v {
				s += float64(x) * float64(x)
			}
			if math.Sqrt(s) > 1e-6 {
				t.Errorf("%q: expected ~zero embedding, got norm %g", c.Text, math.Sqrt(s))
			}
			continue
		}
		cos := cosine(v, c.Emb)
		if cos < worst {
			worst = cos
		}
		if cos < 0.9999 {
			t.Errorf("%q: cosine %.6f < 0.9999 vs StaticModel.encode", c.Text, cos)
		}
	}
	t.Logf("potion-retrieval-32M (no-mapping format) parity over %d cases: worst cosine %.6f", len(g.Cases), worst)
}
