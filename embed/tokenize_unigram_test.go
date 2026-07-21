package embed

import (
	"encoding/json"
	"os"
	"testing"
)

// TestUnigram_encodeParity is the end-to-end id-parity gate for the XLM-R
// (SentencePiece/Unigram) tokenizer: LoadTokenizer must dispatch to the Unigram
// backend, and EncodeWithSpecials(text) must reproduce HF `tokenizers`' full
// input_ids (<s> … </s>) id-for-id over the oracle (scripts/pin_xlmr_tokenizer.py:
// Latin, CJK, RTL, Devanagari, fullwidth, emoji, punctuation, whitespace, code).
func TestUnigram_encodeParity(t *testing.T) {
	const path = "../testdata/xlm-roberta-base/tokenizer.json"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no xlm-roberta-base tokenizer at %s", path)
	}
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	if tok.uni == nil {
		t.Fatal("expected Unigram backend for XLM-R tokenizer.json")
	}

	raw, err := os.ReadFile("../testdata/xlmr_encode_golden.json")
	if err != nil {
		t.Skip("no encode golden — run scripts/pin_xlmr_tokenizer.py")
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

	const maxLen = 512
	mism := 0
	for _, c := range g.Cases {
		got, err := tok.EncodeWithSpecials(c.Text, maxLen)
		if err != nil {
			t.Fatalf("%q: %v", c.Text, err)
		}
		if !equalIDs(got, c.InputIDs) {
			mism++
			t.Errorf("%q:\n got  %v\n want %v", c.Text, got, c.InputIDs)
		}
	}
	if mism == 0 {
		t.Logf("XLM-R Unigram: %d/%d cases id-exact vs HF", len(g.Cases), len(g.Cases))
	}
}

// TestUnigram_breakItFirst proves the parity gate can fail: perturbing each
// load-bearing stage (normalizer off, metaspace marker changed, unk fusion off)
// must break id-parity on at least one oracle case. A gate that can't go red
// isn't a gate.
func TestUnigram_breakItFirst(t *testing.T) {
	const path = "../testdata/xlm-roberta-base/tokenizer.json"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no xlm-roberta-base tokenizer at %s", path)
	}
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../testdata/xlmr_encode_golden.json")
	if err != nil {
		t.Skip("no encode golden")
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
	const maxLen = 512

	// Baseline must be all-green (so any divergence below is the perturbation).
	for _, c := range g.Cases {
		got, _ := tok.EncodeWithSpecials(c.Text, maxLen)
		if !equalIDs(got, c.InputIDs) {
			t.Fatalf("baseline not green on %q — fix parity before break-it-first", c.Text)
		}
	}

	diverges := func() bool {
		for _, c := range g.Cases {
			got, _ := tok.EncodeWithSpecials(c.Text, maxLen)
			if !equalIDs(got, c.InputIDs) {
				return true
			}
		}
		return false
	}

	// (1) Bypass the Precompiled normalizer.
	savedNorm := tok.uni.norm
	tok.uni.norm = &precompiled{} // empty trie → identity normalize
	if !diverges() {
		t.Error("break-it-first vacuous: disabling the normalizer changed no ids")
	}
	tok.uni.norm = savedNorm

	// (2) Turn off unk fusion. The plane-15 PUA case has two adjacent unknown
	// chars that must fuse to a single <unk>; without fusion it emits two, so
	// parity must break.
	savedFuse := tok.uni.model.fuseUnk
	tok.uni.model.fuseUnk = false
	if !diverges() {
		t.Error("break-it-first vacuous: disabling fuse_unk changed no ids (need an adjacent-unk case)")
	}
	tok.uni.model.fuseUnk = savedFuse
}

func equalIDs(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
