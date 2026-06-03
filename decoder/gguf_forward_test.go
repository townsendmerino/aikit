package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// G7 GGUF parity. Loads quantized TinyLlama GGUFs (Q8_0, Q4_0) — the same model
// as testdata/tinyllama-1.1b — through the generic forward and compares to the
// f32 oracle (llama_forward_golden.json + llama_forward_full.json). Quantization
// is lossy, so the bar is the right one for each type: argmax must still land on
// ' Paris', and the cosine vs the f32 logits must clear a per-type floor (Q8_0
// near-lossless; Q4_0 coarser). This validates the GGUF container parse, block
// dequant, llama.cpp→config metadata mapping, and the q/k RoPE un-permutation.
//
// Models: TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF (ungated).
func testGGUFParity(t *testing.T, ggufPath string, minCosine float64) {
	if testing.Short() {
		t.Skip("slow: loads + runs a TinyLlama GGUF")
	}
	raw, err := os.ReadFile(llamaForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no llama golden at %s — regenerate with scripts/pin_llama_forward.py", llamaForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(ggufPath); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no GGUF at %s", ggufPath)
	}

	m, err := Load(ggufPath, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "llama" {
		t.Fatalf("resolved arch %q, want llama", m.w.arch.Name)
	}

	cache := m.NewCache(len(g.IDs))
	for _, id := range g.IDs[:len(g.IDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
	}
	logits, err := m.forward(g.IDs[len(g.IDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	// Quantized argmax must still match the f32 oracle (a confident token).
	if got := argmax(logits); got != g.Argmax {
		t.Errorf("argmax = %d, want %d (logit[got]=%.4f logit[want]=%.4f)",
			got, g.Argmax, logits[got], logits[g.Argmax])
	}
	cos := cosineToFull(t, logits, llamaForwardFullPath)
	if !math.IsNaN(cos) && cos < minCosine {
		t.Errorf("cosine vs f32 oracle = %v, want ≥ %v", cos, minCosine)
	}
	t.Logf("gguf %s: argmax=%d (want %d) | cosine vs f32 = %v", ggufPath, argmax(logits), g.Argmax, cos)
}

func TestGGUF_Q8_0_parity(t *testing.T) {
	testGGUFParity(t, "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q8_0.gguf", 0.999)
}

func TestGGUF_Q4_0_parity(t *testing.T) {
	testGGUFParity(t, "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q4_0.gguf", 0.99)
}

// Q4_K_M — the most-downloaded laptop quant — mixes Q4_K (most weights) and
// Q6_K (output + some attention/ffn), exercising both K-quant dequants.
func TestGGUF_Q4_K_M_parity(t *testing.T) {
	testGGUFParity(t, "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", 0.99)
}
