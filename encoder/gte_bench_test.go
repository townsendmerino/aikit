package encoder

import (
	"os"
	"strings"
	"testing"
)

func loadBenchGTE(b *testing.B) *GTE {
	b.Helper()
	const dir = "../testdata/arctic2-m"
	if _, err := os.Stat(dir); err != nil {
		b.Skip("testdata/arctic2-m/ not present; see scripts/README.md")
	}
	g, err := LoadGTE(dir)
	if err != nil {
		b.Fatalf("LoadGTE: %v", err)
	}
	return g
}

func gteBenchText(n int) string {
	var s strings.Builder
	words := []string{"how", "do", "i", "parse", "json", "in", "go", "with", "generic", "structs",
		"machine", "learning", "for", "semantic", "search", "over", "code", "repositories"}
	for i := range n {
		s.WriteString(words[i%len(words)])
		s.WriteByte(' ')
	}
	return s.String()
}

// BenchmarkGTEEncode is the arbiter for perf-campaign item 8: the fused up/gate
// buffer (L*2*intermediate) and the per-call RoPE table are both allocated
// outside the pooled scratch arena, so bytes/op should scale with L.
func BenchmarkGTEEncode(b *testing.B) {
	g := loadBenchGTE(b)
	for _, L := range []int{16, 128, 512} {
		text := gteBenchText(L)
		b.Run(shapeName("L", L), func(b *testing.B) {
			b.ReportAllocs()
			rec := trackScratch(b)
			for b.Loop() {
				v, err := g.Encode(text)
				if err != nil {
					b.Fatal(err)
				}
				sinkF32 = v
			}
			rec.report(b)
		})
	}
}
