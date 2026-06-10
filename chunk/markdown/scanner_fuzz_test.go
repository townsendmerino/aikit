package markdown

import (
	"strings"
	"testing"
)

// FuzzMarkdownChunk drives the markdown header scanner on arbitrary bytes,
// including pathological header runs and unterminated fences. Must never panic;
// chunks stay structurally valid.
func FuzzMarkdownChunk(f *testing.F) {
	for _, s := range []string{
		"", "# Title\n\nbody text", "## A\n### B\n#### C\n", "```go\ncode\n```",
		"```\nunterminated fence", strings.Repeat("#", 2000), "#\n#\n#\n",
		"no headers, just prose\nover two lines", "#nospace\n# space\n",
		"\x00\xff# weird\n", strings.Repeat("# h\ntext\n", 500),
	} {
		f.Add([]byte(s))
	}
	c := New()
	f.Fuzz(func(t *testing.T, src []byte) {
		chunks, err := c.Chunk(src, "markdown", 50)
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
