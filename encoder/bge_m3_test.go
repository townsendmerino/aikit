package encoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestBGEM3_parity certifies BAAI/bge-m3 — the flagship multilingual retriever —
// full-stack against its sentence-transformers reference (scripts/pin_bge_m3.py).
// It reuses the exact path multilingual-e5-base certified (XLM-R posOff=2 +
// SentencePiece/Unigram tokenizer + forward), but CLS pooling instead of mean and
// at 24 layers / 1024 dim — so it independently gates the CLS reduction on the
// multilingual stack. Same three layers: hidden parity, mean-vs-CLS + offset
// break-it-first, and Encode(text) end-to-end cosine.
func TestBGEM3_parity(t *testing.T) {
	const dir = "../testdata/bge-m3"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no bge-m3 at %s — fetch, convert bin→safetensors, run scripts/pin_bge_m3.py", dir)
	}
	b, err := LoadBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	if b.pool != poolCLS {
		t.Fatalf("bge-m3 pool = %q, want cls (from 1_Pooling/config.json)", b.pool)
	}
	if b.posOff != 2 {
		t.Fatalf("bge-m3 posOff = %d, want 2 (xlm-roberta, pad+1)", b.posOff)
	}
	if b.tok == nil {
		t.Fatal("bge-m3 has no tokenizer — SentencePiece/Unigram should load")
	}

	raw, err := os.ReadFile("../testdata/bge_m3_golden.json")
	if err != nil {
		t.Skip("no golden — run scripts/pin_bge_m3.py")
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
		t.Logf("%-30q L=%2d  hidden maxΔ %.2e  emb(cls) cosine %.6f", c.Text, c.L, maxd, cos)
		if maxd > 5e-3 {
			t.Errorf("%q: hidden-state max |Δ| %.2e vs golden", c.Text, maxd)
		}
		if cos < 0.9999 {
			t.Errorf("%q: CLS embedding cosine %.6f < 0.9999", c.Text, cos)
		}

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
			// Break-it-first (a): mean pooling must diverge from the CLS golden.
			meanCos := cos32(l2norm(poolOne(h, len(h)/D, D, poolMean)), c.Emb)
			if meanCos < 0.99 {
				brokePool = true
			}
			if meanCos >= cos {
				t.Errorf("%q: mean cosine %.6f >= CLS %.6f — gate can't tell pooling apart", c.Text, meanCos, cos)
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
		t.Error("break-it-first vacuous: mean pooling never diverged from the CLS golden")
	}
	if !brokeOffset {
		t.Error("break-it-first vacuous: zeroing posOff never diverged the hidden states")
	}
	t.Logf("bge-m3 full-stack CLS parity over %d cases: worst hidden maxΔ %.2e, worst emb cosine %.6f, worst Encode cosine %.6f",
		len(g.Cases), worstHidden, worstCos, worstEnc)
}
