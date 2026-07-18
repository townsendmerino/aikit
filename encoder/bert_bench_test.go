package encoder

import (
	"os"
	"strings"
	"testing"
)

// BenchmarkBERT_embed measures a single MiniLM forward's time + allocations —
// the scratch-arena rewrite removes the per-(layer,head) qH/kH/vHT + L² scores
// mallocs (12 layers × 12 heads × ~4 buffers) plus the per-layer Q/K/V/ctx/out/
// inter/ffn allocations. Model-gated like the parity test.
func BenchmarkBERT_embed(b *testing.B) {
	const dir = "../testdata/minilm-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		b.Skipf("no MiniLM model at %s", dir)
	}
	m, err := LoadBERT(dir)
	if err != nil {
		b.Fatal(err)
	}
	ids := m.tok.Encode(strings.Repeat("the quick brown fox jumps over the lazy dog. ", 6))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = m.Embed(ids)
	}
}
