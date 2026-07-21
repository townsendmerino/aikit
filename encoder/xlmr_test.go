package encoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestXLMR_forwardParity certifies aikit's XLM-R position-id-OFFSET path against
// a real reference (FacebookAI/xlm-roberta-base, scripts/pin_xlmr.py). XLM-R
// numbers learned positions from padding_idx+1 (posOff = pad_token_id+1 = 2),
// where BERT starts at 0 — a silent-wrong if mishandled, since it shifts every
// position embedding by two slots.
//
// This is a FORWARD-ONLY gate: xlm-roberta-base ships no sentence-transformers
// pooling head and its SentencePiece/Unigram tokenizer isn't one aikit can parse
// yet (see TestXLMR_tokenizerGap), so we feed the golden's HF-produced input_ids
// straight to the forward and compare last_hidden_state. The break-it-first
// zeroes posOff and requires the hidden states to diverge — proving the offset is
// load-bearing, not incidental.
func TestXLMR_forwardParity(t *testing.T) {
	const dir = "../testdata/xlm-roberta-base"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no xlm-roberta-base model at %s — fetch + run scripts/pin_xlmr.py", dir)
	}
	b, err := LoadBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	if b.posOff != 2 {
		t.Fatalf("xlm-roberta-base posOff = %d, want 2 (pad_token_id+1)", b.posOff)
	}

	raw, err := os.ReadFile("../testdata/xlmr_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		PosOffset int `json:"pos_offset"`
		Cases     []struct {
			Text     string    `json:"text"`
			InputIDs []int32   `json:"input_ids"`
			L        int       `json:"L"`
			HiddenSt []float32 `json:"hidden"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if g.PosOffset != b.posOff {
		t.Fatalf("golden pos_offset %d != loaded posOff %d", g.PosOffset, b.posOff)
	}

	maxd := func(h, want []float32) float64 {
		var m float64
		for i := range h {
			if d := math.Abs(float64(h[i]) - float64(want[i])); d > m {
				m = d
			}
		}
		return m
	}

	var worst float64
	var brokeAtLeastOnce bool
	for _, c := range g.Cases {
		h := b.hiddenStates(c.InputIDs, nil)
		if len(h) != len(c.HiddenSt) {
			t.Fatalf("%q: hidden len %d != golden %d", c.Text, len(h), len(c.HiddenSt))
		}
		d := maxd(h, c.HiddenSt)
		if d > worst {
			worst = d
		}
		t.Logf("%-24q L=%2d  hidden maxΔ %.2e (posOff=2)", c.Text, c.L, d)
		if d > 5e-3 {
			t.Errorf("%q: hidden-state max |Δ| %.2e vs golden (offset/forward bug?)", c.Text, d)
		}

		// Break-it-first: with the offset zeroed (BERT-style), the same forward on
		// the same ids must diverge materially — otherwise the gate can't tell a
		// correct offset from a wrong one.
		if c.L > 1 {
			b.posOff = 0
			hBroken := b.hiddenStates(c.InputIDs, nil)
			b.posOff = 2
			broken := maxd(hBroken, c.HiddenSt)
			t.Logf("    break-it-first: posOff=0 hidden maxΔ %.2e (must be large)", broken)
			if broken > 0.1 {
				brokeAtLeastOnce = true
			}
			if broken <= d {
				t.Errorf("%q: zeroed-offset maxΔ %.2e <= correct %.2e — the gate can't tell them apart", c.Text, broken, d)
			}
		}
	}
	if !brokeAtLeastOnce {
		t.Error("break-it-first vacuous: zeroing posOff never made the hidden states diverge")
	}
	t.Logf("xlm-roberta-base forward parity over %d cases (posOff=2): worst hidden maxΔ %.2e", len(g.Cases), worst)
}

// TestXLMR_encodeTokenizer closes the loop now that aikit has a Unigram
// tokenizer: LoadBERT wires it (b.tok non-nil for XLM-R), and the encoder's
// tokenizer produces HF-exact input_ids for the golden texts. Combined with
// TestXLMR_forwardParity (same ids → matching hidden states), the full
// text→hidden pipeline is certified end-to-end.
func TestXLMR_encodeTokenizer(t *testing.T) {
	const dir = "../testdata/xlm-roberta-base"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no xlm-roberta-base model at %s", dir)
	}
	b, err := LoadBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	if b.tok == nil {
		t.Fatal("expected a usable Unigram tokenizer for XLM-R now that aikit supports SentencePiece")
	}

	raw, err := os.ReadFile("../testdata/xlmr_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		Cases []struct {
			Text     string  `json:"text"`
			InputIDs []int32 `json:"input_ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	for _, c := range g.Cases {
		ids, err := b.tok.EncodeWithSpecials(c.Text, b.maxSeq)
		if err != nil {
			t.Fatalf("%q: %v", c.Text, err)
		}
		if len(ids) != len(c.InputIDs) {
			t.Errorf("%q: %d ids, want %d\n got  %v\n want %v", c.Text, len(ids), len(c.InputIDs), ids, c.InputIDs)
			continue
		}
		for i := range ids {
			if ids[i] != c.InputIDs[i] {
				t.Errorf("%q: id[%d]=%d, want %d\n got  %v\n want %v", c.Text, i, ids[i], c.InputIDs[i], ids, c.InputIDs)
				break
			}
		}
	}
}
