package encoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestMultilingualE5_parity is the CAPSTONE multilingual gate: it certifies the
// whole stack at once against a real sentence-transformers reference
// (intfloat/multilingual-e5-base, scripts/pin_e5.py) — the SentencePiece/Unigram
// tokenizer, the XLM-R position-id offset (posOff=2), mean pooling, and the
// forward — where the earlier xlm-roberta-base gate could only check the
// forward+offset (bare LM, no pooling head). Three layers:
//
//   - hidden-state parity on the golden input_ids (forward + offset),
//   - CLS-vs-mean and offset break-it-first (the gate must be able to fail),
//   - Encode(text) cosine parity (tokenizer + forward + pool, end to end).
func TestMultilingualE5_parity(t *testing.T) {
	const dir = "../testdata/multilingual-e5-base"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no multilingual-e5-base at %s — fetch + run scripts/pin_e5.py", dir)
	}
	b, err := LoadBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	if b.pool != poolMean {
		t.Fatalf("e5 pool = %q, want mean (from 1_Pooling/config.json)", b.pool)
	}
	if b.posOff != 2 {
		t.Fatalf("e5 posOff = %d, want 2 (xlm-roberta, pad+1)", b.posOff)
	}
	if b.tok == nil {
		t.Fatal("e5 has no tokenizer — SentencePiece/Unigram should now load")
	}

	raw, err := os.ReadFile("../testdata/e5_golden.json")
	if err != nil {
		t.Skip("no golden — run scripts/pin_e5.py")
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
	D := b.cfg.Hidden

	var worstHidden, worstCos, worstEnc float64 = 0, 1, 1
	var brokePool, brokeOffset bool
	for _, c := range g.Cases {
		h := b.hiddenStates(c.InputIDs, nil)
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
		t.Logf("%-30q L=%2d  hidden maxΔ %.2e  emb(mean) cosine %.6f", c.Text, c.L, maxd, cos)
		if maxd > 5e-3 {
			t.Errorf("%q: hidden-state max |Δ| %.2e vs golden", c.Text, maxd)
		}
		if cos < 0.9999 {
			t.Errorf("%q: mean embedding cosine %.6f < 0.9999", c.Text, cos)
		}

		// End-to-end: aikit's own tokenizer + forward + pool vs the golden vector.
		enc, err := b.Encode(c.Text)
		if err != nil {
			t.Fatalf("%q: Encode: %v", c.Text, err)
		}
		encCos := cos32(enc, c.Emb)
		if encCos < worstEnc {
			worstEnc = encCos
		}
		if encCos < 0.9999 {
			t.Errorf("%q: Encode(text) cosine %.6f < 0.9999 (tokenizer/forward/pool)", c.Text, encCos)
		}

		if c.L > 2 {
			// Break-it-first (a): CLS pooling must diverge from the mean golden.
			clsCos := cos32(l2norm(poolOne(h, len(h)/D, D, poolCLS)), c.Emb)
			if clsCos < 0.99 {
				brokePool = true
			}
			if clsCos >= cos {
				t.Errorf("%q: CLS cosine %.6f >= mean %.6f — gate can't tell pooling apart", c.Text, clsCos, cos)
			}
			// Break-it-first (b): zeroing the offset must diverge the hidden states.
			b.posOff = 0
			hBroken := b.hiddenStates(c.InputIDs, nil)
			b.posOff = 2
			var bd float64
			for i := range hBroken {
				if d := math.Abs(float64(hBroken[i]) - float64(c.HiddenSt[i])); d > bd {
					bd = d
				}
			}
			if bd > 0.1 {
				brokeOffset = true
			}
			if bd <= maxd {
				t.Errorf("%q: zeroed-offset maxΔ %.2e <= correct %.2e", c.Text, bd, maxd)
			}
		}
	}
	if !brokePool {
		t.Error("break-it-first vacuous: CLS pooling never diverged from the mean golden")
	}
	if !brokeOffset {
		t.Error("break-it-first vacuous: zeroing posOff never diverged the hidden states")
	}
	t.Logf("multilingual-e5-base full-stack parity over %d cases: worst hidden maxΔ %.2e, worst emb cosine %.6f, worst Encode cosine %.6f",
		len(g.Cases), worstHidden, worstCos, worstEnc)
}
