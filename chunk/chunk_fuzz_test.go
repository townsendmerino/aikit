package chunk

import (
	"strings"
	"testing"
)

// checkChunks asserts the structural invariants every chunker must hold: 1-based
// line numbers and a Text no longer than the whole source.
func checkChunks(t *testing.T, chunks []Chunk, srcLen int) {
	t.Helper()
	for _, c := range chunks {
		if c.StartLine < 1 || c.EndLine < 1 {
			t.Fatalf("non-positive line numbers: [%d, %d]", c.StartLine, c.EndLine)
		}
		if len(c.Text) > srcLen {
			t.Fatalf("chunk Text len %d exceeds source len %d", len(c.Text), srcLen)
		}
	}
}

// FuzzLineChunker drives the universal line chunker (the indexing fallback)
// through its public entry, ChunkFile, on arbitrary bytes and chunk sizes
// (including 0 / negative). Must never panic.
func FuzzLineChunker(f *testing.F) {
	seeds := []struct {
		src string
		sz  int
	}{
		{"", 50}, {"a\nb\nc", 2}, {"single line, no newline", 50}, {"\n\n\n", 1},
		{"trailing newline\n", 50}, {strings.Repeat("x\n", 1000), 50},
		{"zero size", 0}, {"negative size", -5}, {"\x00\xff", 3}, {"a\r\nb\r\n", 2},
	}
	for _, s := range seeds {
		f.Add([]byte(s.src), s.sz)
	}
	f.Fuzz(func(t *testing.T, src []byte, chunkSize int) {
		chunks, err := ChunkFile("line", "f.txt", src, chunkSize)
		if err != nil {
			return
		}
		checkChunks(t, chunks, len(src))
	})
}
