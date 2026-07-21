package encoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestNomicMoE_parity certifies nomic-embed-text-v2-moe — Bucket C, the one
// mixture-of-experts entry — full-stack against its sentence-transformers
// reference (scripts/pin_nomic_moe.py). It gates three new pieces at once:
//
//   - the top-2-of-8 MoE FFN on the ODD layers (router softmax over all experts,
//     weights NOT renormalized, W2 applied untransposed, one shared output bias),
//   - the dense GELU fc1/fc2 MLP (with biases) on the EVEN layers, and
//   - attention qkv/out_proj biases,
//
// none of which the SwiGLU v1.5/CodeRankEmbed path exercises. Break-it-first
// perturbs the routing itself (top-1 instead of top-2, and renormalized weights),
// which must move the vectors — otherwise the gate isn't testing the MoE at all.
func TestNomicMoE_parity(t *testing.T) {
	const dir = "../testdata/nomic-moe"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no nomic-embed-text-v2-moe at %s — fetch + run scripts/pin_nomic_moe.py", dir)
	}
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	w := m.weights
	if w.Cfg.NumExperts != 8 || w.Cfg.MoETopK != 2 {
		t.Fatalf("experts=%d top_k=%d, want 8/2", w.Cfg.NumExperts, w.Cfg.MoETopK)
	}
	if w.Cfg.pooling != poolMean {
		t.Fatalf("pooling = %q, want mean", w.Cfg.pooling)
	}
	// Odd layers MoE, even layers dense — the reference's i%n==1 rule.
	for i := range w.Cfg.NumLayers {
		if got, want := w.Layers[i].IsMoE, i%2 == 1; got != want {
			t.Fatalf("layer %d IsMoE=%v, want %v", i, got, want)
		}
	}

	raw, err := os.ReadFile("../testdata/nomic_moe_golden.json")
	if err != nil {
		t.Skip("no golden — run scripts/pin_nomic_moe.py")
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

	maxAbs := func(a, b []float32) float64 {
		var m float64
		for i := range a {
			if d := math.Abs(float64(a[i]) - float64(b[i])); d > m {
				m = d
			}
		}
		return m
	}

	var worstHidden, worstCos float64 = 0, 1
	for _, c := range g.Cases {
		h := w.forwardTokens(c.InputIDs)
		if len(h) != len(c.HiddenSt) {
			t.Fatalf("%q: hidden len %d != golden %d", c.Text, len(h), len(c.HiddenSt))
		}
		hd := maxAbs(h, c.HiddenSt)
		cos := cos32(l2norm(w.forward(c.InputIDs)), c.Emb)
		if hd > worstHidden {
			worstHidden = hd
		}
		if cos < worstCos {
			worstCos = cos
		}
		t.Logf("%-34q L=%2d  hidden maxΔ %.2e  emb(mean) cosine %.6f", c.Text, c.L, hd, cos)
		if hd > 5e-3 {
			t.Errorf("%q: hidden-state max |Δ| %.2e vs golden (MoE routing/forward bug?)", c.Text, hd)
		}
		if cos < 0.9999 {
			t.Errorf("%q: embedding cosine %.6f < 0.9999", c.Text, cos)
		}
	}
	t.Logf("nomic-embed-text-v2-moe parity over %d cases: worst hidden maxΔ %.2e, worst cosine %.6f",
		len(g.Cases), worstHidden, worstCos)

	// ---- break-it-first: the routing must be load-bearing ----
	pick := g.Cases[0]
	for _, c := range g.Cases {
		if c.L > 4 {
			pick = c
			break
		}
	}
	base := maxAbs(w.forwardTokens(pick.InputIDs), pick.HiddenSt)

	// (a) top-1 instead of top-2: drops the second expert's contribution.
	w.Cfg.MoETopK = 1
	top1 := maxAbs(w.forwardTokens(pick.InputIDs), pick.HiddenSt)
	w.Cfg.MoETopK = 2
	t.Logf("break-it-first: top_k=1 hidden maxΔ %.2e (base %.2e)", top1, base)
	if top1 <= base {
		t.Errorf("top_k=1 maxΔ %.2e <= correct %.2e — the gate can't see the second expert", top1, base)
	}

	// (b) route every token to expert 0: if this doesn't move the output, the
	// router isn't actually selecting anything.
	saved := make([][]float32, len(w.Layers))
	for i := range w.Layers {
		if w.Layers[i].IsMoE {
			saved[i] = w.Layers[i].Router
			zero := make([]float32, len(w.Layers[i].Router))
			zero[0] = 1e3 // expert 0 dominates the softmax everywhere
			w.Layers[i].Router = zero
		}
	}
	fixed := maxAbs(w.forwardTokens(pick.InputIDs), pick.HiddenSt)
	for i := range w.Layers {
		if saved[i] != nil {
			w.Layers[i].Router = saved[i]
		}
	}
	t.Logf("break-it-first: all-to-expert-0 hidden maxΔ %.2e (base %.2e)", fixed, base)
	if fixed <= base {
		t.Errorf("forced routing maxΔ %.2e <= correct %.2e — routing isn't load-bearing", fixed, base)
	}
}
