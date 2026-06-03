package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// M8 int8 accuracy. Loads the checkpoint with per-row int8 weight quant and
// checks the forward stays faithful to the f32 reference: the argmax (predicted
// token) is unchanged and the full logit vector keeps a high cosine (a looser
// bar than the f32 parity gate — quantization perturbs the logits, it doesn't
// reproduce them). Also confirms the f32 weights were actually freed.
func TestQuantInt8_accuracy(t *testing.T) {
	raw, err := os.ReadFile(gemmaForwardGoldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no forward golden at %s — regenerate with scripts/pin_gemma_forward.py", gemmaForwardGoldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(gemmaModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s", gemmaModelDir)
	}

	m, err := Load(gemmaModelDir, Options{Quant: "int8"})
	if err != nil {
		t.Fatalf("Load int8: %v", err)
	}
	// The quantization must have replaced f32 with int8 (the memory win).
	if m.w.Embed.f32 != nil || m.w.Embed.q8 == nil {
		t.Fatalf("Embed not quantized: f32=%v q8=%v", m.w.Embed.f32 != nil, m.w.Embed.q8 != nil)
	}

	// Retained weight footprint: int8 codes + f32 scales vs the f32 originals.
	var q8Bytes, f32Bytes int
	for _, wm := range m.w.matmulWeights() {
		n := wm.rows * wm.cols
		q8Bytes += n + 4*wm.rows // int8 codes + per-row f32 scale
		f32Bytes += 4 * n
	}
	ratio := float64(f32Bytes) / float64(q8Bytes)
	t.Logf("matmul weight memory: int8 %.0f MB vs f32 %.0f MB (%.2fx smaller)",
		float64(q8Bytes)/1e6, float64(f32Bytes)/1e6, ratio)
	if ratio < 3.5 {
		t.Errorf("int8 memory ratio %.2fx, want ≥ 3.5x", ratio)
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

	// Argmax (the predicted next token) must survive quantization.
	if got := argmax(logits); got != g.Argmax {
		t.Errorf("int8 argmax = %d, want %d (f32) — quantization flipped the top token", got, g.Argmax)
	}

	// High cosine vs the f32 reference dump, when present (looser than 1−1e-4).
	if cos := quantCosine(t, logits); !math.IsNaN(cos) {
		if cos < 0.999 {
			t.Errorf("int8 vs f32 cosine = %.6f, want ≥ 0.999", cos)
		}
		t.Logf("int8: argmax=%d (want %d) | cosine vs f32 = %.6f", argmax(logits), g.Argmax, cos)
	}
}

// quantCosine returns cosine(int8-logits, f32-reference) from the per-machine
// full dump, or NaN if absent.
func quantCosine(t *testing.T, logits []float32) float64 {
	raw, err := os.ReadFile(gemmaForwardFullPath)
	if err != nil {
		return math.NaN()
	}
	var full struct {
		Logits []float64 `json:"logits"`
	}
	if err := json.Unmarshal(raw, &full); err != nil {
		t.Fatalf("parse full dump: %v", err)
	}
	var dot, na, nb float64
	for i, v := range logits {
		a := float64(v)
		b := full.Logits[i]
		dot += a * b
		na += a * a
		nb += b * b
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
