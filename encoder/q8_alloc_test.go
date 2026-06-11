package encoder

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"testing"
)

// rerankCorpus builds n code-snippet candidates for a rerank benchmark.
func rerankCorpus(n int) ([]string, []bool) {
	templates := []string{
		"def %s(x):\n    if x < 2:\n        return x\n    return %s(x-1) + %s(x-2)",
		"class %s:\n    def __init__(self, v):\n        self.v = v\n    def %s(self):\n        return self.v",
		"func %s(in []byte) ([]byte, error) {\n    if len(in) == 0 {\n        return nil, fmt.Errorf(\"empty\")\n    }\n    return in, nil\n}",
	}
	names := []string{"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
		"iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma",
		"tau", "upsilon", "phi", "chi", "psi", "omega", "add", "sub", "mul", "div"}
	texts := make([]string, n)
	isQ := make([]bool, n)
	for i := range n {
		nm := names[i%len(names)]
		texts[i] = fmt.Sprintf(templates[i%len(templates)], nm, nm, nm)
	}
	return texts, isQ
}

// BenchmarkRerankN50_f32_vs_q8 times a 50-doc rerank under the f32 (Load) and int8
// (LoadQ8) encoders with -benchmem. The int8 path used to churn ~12× the bytes of
// f32 (4.4 GiB vs 380 MiB cold) because it allocated per-call/per-layer scratch where
// f32 pools it — making q8 ~5× slower on arm64 despite the SDOT kernel. After pooling
// the q8 scratch (matmulBTQ8Into + the shared scratch arena), q8's allocs/op and B/op
// should be in line with f32, and the int8 dot path should make it at worst
// competitive. This bench guards against a silent regression.
func BenchmarkRerankN50_f32_vs_q8(b *testing.B) {
	if _, err := os.Stat(modelDir); errors.Is(err, fs.ErrNotExist) {
		b.Skipf("no model at %s", modelDir)
	}
	texts, isQ := rerankCorpus(50)

	b.Run("f32", func(b *testing.B) {
		m, err := Load(modelDir)
		if err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := m.EncodeBatch(texts, isQ, 0); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("q8", func(b *testing.B) {
		m, err := LoadQ8(modelDir)
		if err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := m.EncodeBatch(texts, isQ, 0); err != nil {
				b.Fatal(err)
			}
		}
	})
}
