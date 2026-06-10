package encoder

import (
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

func cos32(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// TestPooling_meanEndToEnd exercises the mean-pooling path through the REAL
// forward (the poolOne unit test covers the math; this covers the wiring): mean
// must differ from CLS, be finite, and — the key check — batched mean must equal
// single mean for the same text, proving forward_batch's realLen masking averages
// only the real tokens. Model-gated (skips on CI). Not a parity check (that needs
// a mean-pooled reference model); it guards the wiring.
func TestPooling_meanEndToEnd(t *testing.T) {
	if _, err := os.Stat(modelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no model at %s", modelDir)
	}
	m, err := Load(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	const text = "parse a user authentication token from the request header"

	cls, err := m.Encode(text, false) // default CLS
	if err != nil {
		t.Fatal(err)
	}
	m.weights.Cfg.pooling = poolMean
	mean, err := m.Encode(text, false)
	if err != nil {
		t.Fatal(err)
	}

	for _, v := range mean {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("mean pooling produced a non-finite value")
		}
	}
	if c := cos32(cls, mean); c > 0.999 {
		t.Errorf("mean vs CLS cosine %.5f — mean pooling appears not applied", c)
	}

	// Batched mean must match single mean for the same text → realLen masking is
	// averaging only the real tokens (not padding) in the batched forward.
	texts := []string{"short", text, "a third sentence of a different length entirely"}
	batch, err := m.EncodeBatch(texts, []bool{false, false, false}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if c := cos32(batch[1], mean); c < 0.999 {
		t.Errorf("batched mean vs single mean cosine %.6f < 0.999 — realLen masking is wrong", c)
	}
}
