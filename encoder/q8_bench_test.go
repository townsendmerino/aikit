package encoder

import (
	"os"
	"strings"
	"testing"
)

// BenchmarkQ8Encode is the arbiter for perf-campaign item 22 (the q8 path
// re-widening the whole int8 weight matrix to f32 per matmul). The doc's claim
// is ~113 M converts per forward INDEPENDENT of L, so the cost should amortize
// as 1/M — visible at short L, small at long L. Both are measured here.
func BenchmarkQ8Encode(b *testing.B) {
	const dir = "../testdata/encoder-model"
	if _, err := os.Stat(dir); err != nil {
		b.Skip("testdata/encoder-model/ not present; see scripts/README.md")
	}
	m, err := LoadQ8(dir)
	if err != nil {
		b.Fatalf("LoadQ8: %v", err)
	}
	defer func() { _ = m.Close() }()

	for _, words := range []int{8, 64, 256} {
		var sb strings.Builder
		w := []string{"func", "parse", "json", "into", "a", "generic", "struct", "with", "reflection"}
		for i := range words {
			sb.WriteString(w[i%len(w)])
			sb.WriteByte(' ')
		}
		text := sb.String()
		b.Run(shapeName("w", words), func(b *testing.B) {
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
