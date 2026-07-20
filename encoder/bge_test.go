package encoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestBGE_parity certifies the CLS-pooled BERT embedder against a real reference
// (BAAI/bge-small-en-v1.5, scripts/pin_bge.py). bge-small is the same
// architecture as all-MiniLM (learned-absolute positions, GELU FFN, WordPiece)
// but pools CLS instead of mean, so this is the end-to-end gate for the
// declared-pooling work: LoadBERT must read pooling_mode_cls_token from
// 1_Pooling/config.json and Embed must reproduce the CLS-pooled vector.
func TestBGE_parity(t *testing.T) {
	const dir = "../testdata/bge-small"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no bge-small model at %s — fetch + run scripts/pin_bge.py", dir)
	}
	b, err := LoadBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	// The declared reduction must be CLS (read from config, not the mean default).
	if b.pool != poolCLS {
		t.Fatalf("bge-small pool = %q, want cls (from 1_Pooling/config.json)", b.pool)
	}
	if b.posOff != 0 {
		t.Errorf("bge-small posOff = %d, want 0 (BERT)", b.posOff)
	}

	raw, err := os.ReadFile("../testdata/bge_golden.json")
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
	D := b.cfg.Hidden

	var worstHidden, worstCos float64 = 0, 1
	var brokeAtLeastOnce bool
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
		t.Logf("%-52q L=%2d  hidden maxΔ %.2e  emb(cls) cosine %.6f", c.Text, c.L, maxd, cos)
		if maxd > 5e-3 {
			t.Errorf("%q: hidden-state max |Δ| %.2e vs golden (forward bug?)", c.Text, maxd)
		}
		if cos < 0.9999 {
			t.Errorf("%q: CLS embedding cosine %.6f < 0.9999", c.Text, cos)
		}

		// Break-it-first: pooling MEAN instead of CLS must materially diverge from
		// the (CLS) golden on a non-degenerate case — proving the gate can fail,
		// i.e. that reading the right pooling mode is load-bearing, not incidental.
		if c.L > 2 {
			meanCos := cos32(l2norm(poolOne(h, len(h)/D, D, poolMean)), c.Emb)
			if meanCos < 0.99 {
				brokeAtLeastOnce = true
			}
			t.Logf("    break-it-first: mean-pooled cosine %.6f (must be < CLS)", meanCos)
			if meanCos >= cos {
				t.Errorf("%q: mean pooling cosine %.6f >= CLS %.6f — the gate can't tell them apart", c.Text, meanCos, cos)
			}
		}
	}
	if !brokeAtLeastOnce {
		t.Error("break-it-first vacuous: mean pooling never diverged materially from the CLS golden")
	}
	t.Logf("bge-small CLS parity over %d cases: worst hidden maxΔ %.2e, worst emb cosine %.6f", len(g.Cases), worstHidden, worstCos)
}

// TestBGE_encodeEndToEnd certifies the full text→embedding pipeline for a
// do_lower_case=true WordPiece model (the MiniLM fixture is do_lower_case=false,
// so this exercises the lowercasing path): aikit's tokenizer must produce the
// same input_ids as HF, and Encode(text) must match the CLS-pooled golden.
func TestBGE_encodeEndToEnd(t *testing.T) {
	const dir = "../testdata/bge-small"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no bge-small model at %s", dir)
	}
	b, err := LoadBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../testdata/bge_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		Cases []struct {
			Text     string    `json:"text"`
			InputIDs []int32   `json:"input_ids"`
			Emb      []float32 `json:"embedding"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	tokMismatch := 0
	for _, c := range g.Cases {
		ids, err := b.tok.EncodeWithSpecials(c.Text, b.maxSeq)
		if err != nil {
			t.Fatal(err)
		}
		same := len(ids) == len(c.InputIDs)
		for i := range ids {
			if same && ids[i] != c.InputIDs[i] {
				same = false
			}
		}
		if !same {
			tokMismatch++
			t.Logf("tokenizer mismatch %q: got %v want %v", c.Text, ids, c.InputIDs)
		}
		emb, err := b.Encode(c.Text)
		if err != nil {
			t.Fatal(err)
		}
		if cos := cos32(emb, c.Emb); cos < 0.9999 {
			t.Errorf("%q: Encode cosine %.6f < 0.9999", c.Text, cos)
		}
	}
	if tokMismatch > 0 {
		t.Errorf("%d/%d cases had WordPiece id mismatches (do_lower_case handling?)", tokMismatch, len(g.Cases))
	}
}
