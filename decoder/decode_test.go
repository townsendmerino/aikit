package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/tokenizer"
)

// M4 multi-token decode parity. Prefills a prompt, greedy-decodes N tokens off
// the KV cache, and asserts the continuation matches HF greedy id-for-id — the
// check that catches position-advance, K/V-append, and causal-mask bugs that a
// single forward (M3) can't see. Also decodes the ids back through the M2
// tokenizer and compares the string, exercising the whole stack end to end.
//
// Regenerate the oracle:
//
//	.venv/bin/python scripts/pin_gemma_generate.py
const gemmaGenerateGoldenPath = "../testdata/gemma_generate_golden.json"

type generateGolden struct {
	PromptIDs        []int  `json:"prompt_ids"`
	NNew             int    `json:"n_new"`
	ContinuationIDs  []int  `json:"continuation_ids"`
	ContinuationText string `json:"continuation_text"`
}

func TestDecode_greedyContinuationParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: 48-step greedy decode on the naive backend (M7 perf pending)")
	}
	raw, err := os.ReadFile(gemmaGenerateGoldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no generate golden at %s — regenerate with scripts/pin_gemma_generate.py", gemmaGenerateGoldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g generateGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(gemmaModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — huggingface-cli download google/gemma-3-270m --local-dir %s",
			gemmaModelDir, gemmaModelDir)
	}

	m, err := Load(gemmaModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Prefill the prompt; the last prompt token's forward seeds step 0.
	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	for _, id := range g.PromptIDs[:len(g.PromptIDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
	}
	logits, err := m.forward(g.PromptIDs[len(g.PromptIDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	// Greedy decode N tokens (EOS suppressed to match the forced-length golden).
	got := make([]int, 0, g.NNew)
	for i := 0; i < g.NNew; i++ {
		next := argmax(logits)
		got = append(got, next)
		if i < g.NNew-1 {
			if logits, err = m.forward(next, cache); err != nil {
				t.Fatalf("forward step %d: %v", i, err)
			}
		}
	}

	// 1. Continuation ids identical (the decode-loop correctness gate).
	if len(got) != len(g.ContinuationIDs) {
		t.Fatalf("got %d continuation ids, want %d", len(got), len(g.ContinuationIDs))
	}
	for i := range got {
		if got[i] != g.ContinuationIDs[i] {
			t.Fatalf("continuation diverges at step %d: got %d, want %d\n  got  %v\n  want %v",
				i, got[i], g.ContinuationIDs[i], got, g.ContinuationIDs)
		}
	}

	// 2. Decoded string matches HF (end-to-end: M3/M4 ids → M2 Decode).
	tk, err := tokenizer.Load(gemmaModelDir)
	if err != nil {
		t.Fatalf("tokenizer.Load: %v", err)
	}
	text, err := tk.Decode(got)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if text != g.ContinuationText {
		t.Errorf("continuation text mismatch:\n  got  %q\n  want %q", text, g.ContinuationText)
	}
	t.Logf("greedy continuation (%d tok) matches HF: %q", g.NNew, text)
}
