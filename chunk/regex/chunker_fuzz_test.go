package regex

import (
	"strings"
	"testing"
)

// FuzzRegexChunk drives the per-language regex chunkers on arbitrary source and
// language strings (including unsupported langs, which fall back). The regexes
// run on untrusted bytes — must never panic, chunks stay structurally valid.
func FuzzRegexChunk(f *testing.F) {
	seeds := []struct {
		src, lang string
	}{
		{"func main() {}", "go"}, {"def f():\n  pass", "python"},
		{"class A {}", "java"}, {"function x(){}", "typescript"},
		{"fn main() {}", "rust"}, {"", "go"}, {"unbalanced {{{", "go"},
		{"plain text", "unsupported-lang"}, {"\x00\xff garbage", "python"},
		{strings.Repeat("func a(){}\n", 400), "go"},
		{strings.Repeat("{", 3000), "rust"},
	}
	for _, s := range seeds {
		f.Add([]byte(s.src), s.lang)
	}
	c := New()
	f.Fuzz(func(t *testing.T, src []byte, lang string) {
		chunks, err := c.Chunk(src, lang, 50)
		if err != nil {
			return
		}
		for _, ch := range chunks {
			if ch.StartLine < 1 || ch.EndLine < 1 {
				t.Fatalf("non-positive line numbers: [%d, %d]", ch.StartLine, ch.EndLine)
			}
			if len(ch.Text) > len(src) {
				t.Fatalf("chunk Text len %d exceeds source len %d", len(ch.Text), len(src))
			}
		}
	})
}
