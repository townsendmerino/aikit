package embed

import (
	"encoding/json"
	"os"
	"testing"
)

// TestUnigram_bgeM3Encode certifies the bge-m3 tokenizer variant against the raw
// HF `tokenizers` contract (scripts/pin_bge_m3.py). bge-m3 differs from XLM-R in
// two config-driven ways this exercises: a normalizer Sequence[Precompiled,
// Replace(" {2,}"→" ")] that collapses space runs, and a BARE Metaspace
// pre-tokenizer (no WhitespaceSplit) that keeps a lone ▁ for a trailing space.
// Cases include multi-space and leading/trailing runs where the two Metaspace
// variants diverge — so this is the gate that the bge-m3 path is exact, not the
// XLM-R path reused.
func TestUnigram_bgeM3Encode(t *testing.T) {
	const path = "../testdata/bge-m3/tokenizer.json"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no bge-m3 tokenizer at %s", path)
	}
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	if tok.uni == nil {
		t.Fatal("expected Unigram backend for bge-m3")
	}
	if tok.uni.whitespaceSplit {
		t.Fatal("bge-m3 pre_tokenizer is bare Metaspace, not WhitespaceSplit+Metaspace")
	}
	if len(tok.uni.replaces) != 1 {
		t.Fatalf("bge-m3 normalizer should carry one Replace rule, got %d", len(tok.uni.replaces))
	}

	raw, err := os.ReadFile("../testdata/bge_m3_encode_golden.json")
	if err != nil {
		t.Skip("no encode golden — run scripts/pin_bge_m3.py")
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
	const maxLen = 8192
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
		t.Logf("bge-m3 Unigram: %d/%d cases id-exact vs raw HF tokenizer", len(g.Cases), len(g.Cases))
	}
}
