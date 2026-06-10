package encoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestBERT_parity pins the MiniLM forward (§2.2) against the sentence-transformers
// golden (scripts/pin_minilm.py): feeding the golden's input_ids, the Go forward's
// last_hidden_state must match per-element and its mean-pooled L2-normalized
// embedding must match by cosine. Model-gated (skips without testdata/minilm-model).
func TestBERT_parity(t *testing.T) {
	const dir = "../testdata/minilm-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no MiniLM model at %s — fetch via scripts/README.md", dir)
	}
	b, err := LoadBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../testdata/minilm_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		Hidden int `json:"hidden"`
		Cases  []struct {
			Text     string    `json:"text"`
			InputIDs []int32   `json:"input_ids"`
			L        int       `json:"L"`
			HiddenSt []float32 `json:"hidden"`
			Emb      []float32 `json:"embedding"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}

	var worstHidden, worstCos float64 = 0, 1
	for _, c := range g.Cases {
		h := b.hiddenStates(c.InputIDs)
		if len(h) != len(c.HiddenSt) {
			t.Fatalf("%q: hidden len %d != golden %d", c.Text, len(h), len(c.HiddenSt))
		}
		var maxd float64
		for i := range h {
			if d := math.Abs(float64(h[i]) - float64(c.HiddenSt[i])); d > maxd {
				maxd = d
			}
		}
		cos := cos32(b.Embed(c.InputIDs), c.Emb)
		if maxd > worstHidden {
			worstHidden = maxd
		}
		if cos < worstCos {
			worstCos = cos
		}
		t.Logf("%-34q L=%2d  hidden maxΔ %.2e  emb cosine %.6f", c.Text, c.L, maxd, cos)

		if maxd > 5e-3 {
			t.Errorf("%q: hidden-state max |Δ| %.2e vs golden (forward bug?)", c.Text, maxd)
		}
		if cos < 0.9999 {
			t.Errorf("%q: embedding cosine %.6f < 0.9999", c.Text, cos)
		}
	}
	t.Logf("MiniLM parity over %d cases: worst hidden maxΔ %.2e, worst emb cosine %.6f", len(g.Cases), worstHidden, worstCos)
}
