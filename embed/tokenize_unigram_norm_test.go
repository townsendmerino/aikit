package embed

import (
	"encoding/json"
	"os"
	"testing"
)

// loadCharsmap pulls the base64 precompiled_charsmap out of a tokenizer.json.
func loadCharsmap(t *testing.T, path string) *precompiled {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("no tokenizer.json at %s", path)
	}
	var tj struct {
		Normalizer struct {
			Type                string `json:"type"`
			PrecompiledCharsmap string `json:"precompiled_charsmap"`
		} `json:"normalizer"`
	}
	if err := json.Unmarshal(raw, &tj); err != nil {
		t.Fatal(err)
	}
	if tj.Normalizer.Type != "Precompiled" {
		t.Fatalf("normalizer.type = %q, want Precompiled", tj.Normalizer.Type)
	}
	p, err := newPrecompiled(tj.Normalizer.PrecompiledCharsmap)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// TestPrecompiledNormalizer_oracle is the byte-exact gate for the SentencePiece
// Precompiled normalizer: for every case in the HF-generated oracle
// (per-codepoint sweep over U+0000..U+2FFFF plus combining sequences and real
// multilingual lines, scripts/pin_xlmr_tokenizer.py) aikit's normalize must equal
// HF `tokenizers`' normalizer.normalize_str output character-for-character.
func TestPrecompiledNormalizer_oracle(t *testing.T) {
	p := loadCharsmap(t, "../testdata/xlm-roberta-base/tokenizer.json")

	raw, err := os.ReadFile("../testdata/xlmr_norm_golden.json")
	if err != nil {
		t.Skip("no norm golden — run scripts/pin_xlmr_tokenizer.py")
	}
	var g struct {
		Cases []struct {
			In  string `json:"in"`
			Out string `json:"out"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}

	var mism, checked int
	for _, c := range g.Cases {
		got := p.normalize(c.In)
		checked++
		if got != c.Out {
			mism++
			if mism <= 20 {
				t.Errorf("normalize(%q) = %q, want %q", c.In, got, c.Out)
			}
		}
	}
	if mism > 0 {
		t.Errorf("%d/%d normalization mismatches vs HF oracle", mism, checked)
	}
	t.Logf("Precompiled normalizer: %d cases byte-exact vs HF", checked)
}
