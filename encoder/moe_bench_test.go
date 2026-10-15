package encoder

import (
	"path/filepath"
	"strings"
	"testing"
)

func loadMoE(tb testing.TB) *Model {
	tb.Helper()
	const dir = "../testdata/moe-model"
	// The configs/tokenizer are committed but the weights (*.safetensors) are
	// gitignored, so in CI the directory exists while the model file does not —
	// stat the weights, not the directory, or Load fatals. See scripts/README.md.
	if m, _ := filepath.Glob(filepath.Join(dir, "*.safetensors")); len(m) == 0 {
		tb.Skip("testdata/moe-model/*.safetensors not present; see scripts/README.md")
	}
	m, err := Load(dir)
	if err != nil {
		tb.Fatalf("Load: %v", err)
	}
	return m
}

func moeText(words int) string {
	w := strings.Fields("encoding json unmarshal struct reflection buffer reader context " +
		"deadline cancel goroutine channel select mutex atomic pointer slice map")
	var b strings.Builder
	for i := range words {
		b.WriteString(w[i%len(w)])
		b.WriteByte(' ')
	}
	return b.String()
}

// BenchmarkMoEEncode is the arbiter for perf-campaign item 33. nomic-embed-text-v2-moe
// has 8 experts, top-k 2, an MoE layer every 2 of 12 — so at L tokens each MoE
// layer issues L×2×2 single-row GEMM calls.
func BenchmarkMoEEncode(b *testing.B) {
	m := loadMoE(b)
	defer func() { _ = m.Close() }()
	for _, words := range []int{16, 128, 400} {
		text := moeText(words)
		b.Run(shapeName("w", words), func(b *testing.B) {
			b.ReportAllocs()
			rec := trackScratch(b)
			for b.Loop() {
				v, err := m.Encode(text, false)
				if err != nil {
					b.Fatal(err)
				}
				sinkF32 = v
			}
			rec.report(b)
		})
	}
}
