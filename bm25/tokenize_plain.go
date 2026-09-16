package bm25

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

var asciiTable = func() (t [256]uint8) {
	for c := 'a'; c <= 'z'; c++ {
		t[c] = 1
	}
	for c := '0'; c <= '9'; c++ {
		t[c] = 1
	}
	for c := 'A'; c <= 'Z'; c++ {
		t[c] = 2
	}
	for c := 0x80; c <= 0xFF; c++ {
		t[c] = 3
	}
	return t
}()

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
	n := len(text)
	if n == 0 {
		return nil
	}
	out := make([]string, 0, n/6+1)
	var stackBuf [64]byte

	i := 0
	for i < n {
		// 1. Skip delimiters
		for i < n {
			b := text[i]
			c := asciiTable[b]
			if c == 0 {
				i++
			} else if c == 3 {
				r, size := utf8.DecodeRuneInString(text[i:])
				if unicode.IsLetter(r) || unicode.IsDigit(r) {
					break // found start of a non-ASCII token
				}
				i += size
			} else {
				break // found start of an ASCII token
			}
		}
		if i >= n {
			break
		}

		// 2. Start of a token at index i
		start := i
		if asciiTable[text[i]] == 3 {
			// Starts with non-ASCII
			var b strings.Builder
			for i < n {
				if text[i] < 0x80 {
					c := asciiTable[text[i]]
					if c == 1 {
						b.WriteByte(text[i])
						i++
					} else if c == 2 {
						b.WriteByte(text[i] + ('a' - 'A'))
						i++
					} else {
						break // ASCII delimiter
					}
				} else {
					r, size := utf8.DecodeRuneInString(text[i:])
					if unicode.IsLetter(r) || unicode.IsDigit(r) {
						b.WriteRune(unicode.ToLower(r))
						i += size
					} else {
						break // non-ASCII delimiter
					}
				}
			}
			out = append(out, b.String())
			continue
		}

		// Starts with ASCII
		hasUpper := false
		for i < n {
			c := asciiTable[text[i]]
			if c == 1 {
				i++
			} else if c == 2 {
				hasUpper = true
				i++
			} else {
				break
			}
		}

		// Check if we stopped on a non-ASCII letter/digit
		if i < n && asciiTable[text[i]] == 3 {
			r, size := utf8.DecodeRuneInString(text[i:])
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				// Continuation into non-ASCII
				var b strings.Builder
				b.Grow((i - start) + 8)
				for j := start; j < i; j++ {
					ch := text[j]
					if ch >= 'A' && ch <= 'Z' {
						b.WriteByte(ch + ('a' - 'A'))
					} else {
						b.WriteByte(ch)
					}
				}
				b.WriteRune(unicode.ToLower(r))
				i += size
				for i < n {
					if text[i] < 0x80 {
						c := asciiTable[text[i]]
						if c == 1 {
							b.WriteByte(text[i])
							i++
						} else if c == 2 {
							b.WriteByte(text[i] + ('a' - 'A'))
							i++
						} else {
							break
						}
					} else {
						r, size := utf8.DecodeRuneInString(text[i:])
						if unicode.IsLetter(r) || unicode.IsDigit(r) {
							b.WriteRune(unicode.ToLower(r))
							i += size
						} else {
							break
						}
					}
				}
				out = append(out, b.String())
				continue
			}
		}

		// Pure ASCII token from start to i
		tokLen := i - start
		if !hasUpper {
			out = append(out, text[start:i])
		} else if tokLen <= len(stackBuf) {
			for j := 0; j < tokLen; j++ {
				ch := text[start+j]
				if ch >= 'A' && ch <= 'Z' {
					stackBuf[j] = ch + ('a' - 'A')
				} else {
					stackBuf[j] = ch
				}
			}
			out = append(out, string(stackBuf[:tokLen]))
		} else {
			buf := make([]byte, tokLen)
			for j := 0; j < tokLen; j++ {
				ch := text[start+j]
				if ch >= 'A' && ch <= 'Z' {
					buf[j] = ch + ('a' - 'A')
				} else {
					buf[j] = ch
				}
			}
			out = append(out, string(buf))
		}
	}
	return out
}
