package tokenizer

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// GGUF tokenizer parity (G7 follow-up). The bare-GGUF loader must reproduce the
// HF ids id-for-id, and agree with the tokenizer.json loader on the same model.
// Both assets are TinyLlama (Llama-2 SentencePiece, tokenizer.ggml.model ==
// "llama"); skips cleanly when they're absent so a fresh checkout stays green.
//
// Regenerate the golden:
//
//	hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 tokenizer.json tokenizer_config.json --local-dir testdata/tinyllama-1.1b
//	.venv/bin/python scripts/pin_tinyllama_tokenizer.py

const (
	tinyllamaGGUF   = "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
	tinyllamaGolden = "../testdata/tinyllama_tokenizer_golden.json"
)

// TestLoadGGUF_tinyllamaParity: the tokenizer built from GGUF metadata alone
// (vocab + merges + special ids) must match the HF oracle id-for-id, with the
// ▁ dummy prefix on encode and the single-leading-space strip on decode.
func TestLoadGGUF_tinyllamaParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: reads a ~1 GB GGUF")
	}
	g := loadGoldenAt(t, tinyllamaGolden)
	if _, err := os.Stat(tinyllamaGGUF); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no GGUF at %s", tinyllamaGGUF)
	}

	tk, err := LoadGGUF(tinyllamaGGUF)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}
	if tk.mode != modeGemma {
		t.Fatalf("resolved mode %d, want modeGemma (byte-fallback)", tk.mode)
	}
	if sp := tk.Special(); sp.BOS != g.SpecialTokens.BOS || sp.EOS != g.SpecialTokens.EOS {
		t.Errorf("special tokens: got BOS=%d EOS=%d, want BOS=%d EOS=%d",
			sp.BOS, sp.EOS, g.SpecialTokens.BOS, g.SpecialTokens.EOS)
	}

	checkGoldenCases(t, tk, g)
}

// checkGoldenCases asserts a loaded tokenizer matches every golden case:
// Encode without/with BOS, and Decode back to HF's rendering. (Mirrors the
// byte-level/Gemma parity runners; lives here to validate either loader.)
func checkGoldenCases(t *testing.T, tk *Tokenizer, g tokGolden) {
	t.Helper()
	for _, c := range g.Cases {
		t.Run(caseName(c.Text), func(t *testing.T) {
			got, err := tk.Encode(c.Text, false)
			if err != nil {
				t.Fatalf("Encode(addBOS=false): %v", err)
			}
			if !equalInts(got, c.IDs) {
				t.Errorf("Encode(%q, false)\n  got  %v\n  want %v\n  toks %v", c.Text, got, c.IDs, c.Tokens)
			}
			gotBOS, err := tk.Encode(c.Text, true)
			if err != nil {
				t.Fatalf("Encode(addBOS=true): %v", err)
			}
			if !equalInts(gotBOS, c.IDsBOS) {
				t.Errorf("Encode(%q, true)\n  got  %v\n  want %v", c.Text, gotBOS, c.IDsBOS)
			}
			dec, err := tk.Decode(c.IDs)
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			if dec != c.Decode {
				t.Errorf("Decode(%v) = %q, want %q", c.IDs, dec, c.Decode)
			}
		})
	}
}

// loadGoldenAt reads a tokGolden from path, skipping when absent.
func loadGoldenAt(t *testing.T, path string) tokGolden {
	t.Helper()
	raw, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden at %s — regenerate with scripts/pin_tinyllama_tokenizer.py", path)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g tokGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	return g
}
