package bm25

import (
	"strings"
	"testing"
)

// FuzzTokenize asserts the code-tuned tokenizer never panics and never emits an
// empty token, on arbitrary (including non-UTF8) input. A panic here takes down
// an indexing pipeline (§3.4).
func FuzzTokenize(f *testing.F) {
	for _, s := range []string{
		"", "getUserName", "snake_case_id", "ALLCAPS", "café  Über", "a1b2c3",
		"\x00\xff\xfe", "   ", "日本語テスト", "_", "___", "x__y", "HTTPSConn",
		strings.Repeat("x_", 2000), strings.Repeat("A", 5000),
	} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		for _, tok := range Tokenize(s) {
			if tok == "" {
				t.Fatalf("Tokenize(%q) produced an empty token", s)
			}
		}
		for _, tok := range TokenizePlain(s) {
			if tok == "" {
				t.Fatalf("TokenizePlain(%q) produced an empty token", s)
			}
		}
	})
}
