package bm25

import (
	"strings"
	"unicode"
)

// TokenizePlain is the general-text analyzer: a Unicode word tokenizer that
// lowercases and breaks on any non-letter/non-digit, with NO identifier
// (snake/camel) splitting. It's the analyzer to use for natural-language corpora,
// where Tokenize's code-tuned behavior is wrong — Tokenize splits getUserName into
// get/user/name/getusername and snake_case on underscores, which over-fragments
// prose and breaks hyphenated or apostrophed words.
//
//	TokenizePlain("The quick brown fox!")  → [the quick brown fox]
//	TokenizePlain("GPT-4 and BM25.")       → [gpt 4 and bm25]
//	TokenizePlain("Café Über-cool")        → [café über cool]
//	Tokenize("getUserName")                → [get user name getusername]  (code-tuned)
//
// Tokenize stays the default for code retrieval; pick whichever matches your
// corpus and feed its output to Build / Query (both take pre-tokenized docs).
// Like Tokenize, this drops the separators and keeps only the word tokens; BM25's
// IDF already downweights common words, so no stopword list is applied.
func TokenizePlain(text string) []string {
	// Pre-size for the common case of ~1 token per 6 chars.
	out := make([]string, 0, len(text)/6+1)
	var b strings.Builder
	flush := func() {
		if b.Len() > 0 {
			out = append(out, b.String())
			b.Reset()
		}
	}
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(unicode.ToLower(r))
		} else {
			flush()
		}
	}
	flush()
	return out
}
