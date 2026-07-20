package encoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestNomicEmbed_parity certifies the MEAN-pooled nomic-bert (RoPE) embedder
// against a real reference (nomic-ai/nomic-embed-text-v1.5, scripts/
// pin_nomic_embed.py). It's the same architecture as CodeRankEmbed (RoPE, SwiGLU)
// but pools MEAN instead of CLS — so it's the end-to-end gate for the Nomic
// loader's declared-pooling change on the RoPE path, which the BERT-based
// all-MiniLM/bge fixtures don't exercise.
func TestNomicEmbed_parity(t *testing.T) {
	const dir = "../testdata/nomic-embed"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no nomic-embed model at %s — fetch + run scripts/pin_nomic_embed.py", dir)
	}
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	w := m.weights
	// The declared reduction must be mean (read from config, not the CLS default).
	if w.Cfg.pooling != poolMean {
		t.Fatalf("nomic-embed pooling = %q, want mean (from 1_Pooling/config.json)", w.Cfg.pooling)
	}

	raw, err := os.ReadFile("../testdata/nomic_embed_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		Cases []struct {
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
	D := w.Cfg.HiddenDim

	var worstHidden, worstCos float64 = 0, 1
	var brokeAtLeastOnce bool
	for _, c := range g.Cases {
		h := w.forwardTokens(c.InputIDs) // full [L, D] hidden — no pooling
		if len(h) != len(c.HiddenSt) {
			t.Fatalf("%q: hidden len %d != golden %d", c.Text, len(h), len(c.HiddenSt))
		}
		var maxd float64
		for i := range h {
			if d := math.Abs(float64(h[i]) - float64(c.HiddenSt[i])); d > maxd {
				maxd = d
			}
		}
		cos := cos32(l2norm(w.forward(c.InputIDs)), c.Emb) // pooled (mean) + normalized
		if maxd > worstHidden {
			worstHidden = maxd
		}
		if cos < worstCos {
			worstCos = cos
		}
		t.Logf("%-46q L=%2d  hidden maxΔ %.2e  emb(mean) cosine %.6f", c.Text, c.L, maxd, cos)
		if maxd > 5e-3 {
			t.Errorf("%q: hidden-state max |Δ| %.2e vs golden (RoPE forward bug?)", c.Text, maxd)
		}
		if cos < 0.9999 {
			t.Errorf("%q: mean embedding cosine %.6f < 0.9999", c.Text, cos)
		}

		// Break-it-first: pooling CLS instead of mean must diverge from the (mean)
		// golden — proving the gate can fail, i.e. the declared mode is load-bearing.
		if c.L > 2 {
			clsCos := cos32(l2norm(poolOne(h, len(h)/D, D, poolCLS)), c.Emb)
			if clsCos < 0.99 {
				brokeAtLeastOnce = true
			}
			if clsCos >= cos {
				t.Errorf("%q: CLS pooling cosine %.6f >= mean %.6f — the gate can't tell them apart", c.Text, clsCos, cos)
			}
		}
	}
	if !brokeAtLeastOnce {
		t.Error("break-it-first vacuous: CLS pooling never diverged materially from the mean golden")
	}
	t.Logf("nomic-embed mean parity over %d cases: worst hidden maxΔ %.2e, worst emb cosine %.6f", len(g.Cases), worstHidden, worstCos)
}
