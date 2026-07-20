package encoder

import (
	"os"
	"testing"
)

// TestPositionOffset (embedding-coverage §Phase1 item 1): the RoBERTa/XLM-R
// padding_idx+1 position offset is gated on model_type — applying it to BERT
// (which also carries pad_token_id) would be a silent-wrong that reads the wrong
// posEmb row for every token.
func TestPositionOffset(t *testing.T) {
	cases := []struct {
		modelType string
		pad       int
		want      int
	}{
		{"bert", 0, 0},        // BERT: no offset even with pad_token_id set
		{"bert", 1, 0},        //
		{"roberta", 1, 2},     // RoBERTa: padding_idx+1
		{"xlm-roberta", 1, 2}, // XLM-R:   padding_idx+1
		{"xlm-roberta", 0, 1}, //
		{"", 5, 0},            // unknown/absent model_type → no offset
		{"roberta", -1, 0},    // guard a bogus negative pad
	}
	for _, c := range cases {
		if got := positionOffset(c.modelType, c.pad); got != c.want {
			t.Errorf("positionOffset(%q, %d) = %d, want %d", c.modelType, c.pad, got, c.want)
		}
	}
}

// TestBERT_loaderVariantsFromFixture asserts the real all-MiniLM fixture derives
// BERT-family defaults: no position offset, token_type present. (The parity test
// covers that these produce bit-identical output.)
func TestBERT_loaderVariantsFromFixture(t *testing.T) {
	const dir = "../testdata/minilm-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no MiniLM model at %s", dir)
	}
	b, err := LoadBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	if b.posOff != 0 {
		t.Errorf("all-MiniLM posOff = %d, want 0 (BERT numbers positions from 0)", b.posOff)
	}
	if len(b.typeEmb) == 0 {
		t.Error("all-MiniLM has type_vocab_size 2 but token_type embeddings weren't loaded")
	}
}
