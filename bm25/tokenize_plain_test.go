package bm25

import (
	"reflect"
	"testing"
)

func TestTokenizePlain(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"The quick brown fox!", []string{"the", "quick", "brown", "fox"}},
		{"GPT-4 and BM25.", []string{"gpt", "4", "and", "bm25"}},
		{"Café Über-cool", []string{"café", "über", "cool"}},
		{"version2.0 build", []string{"version2", "0", "build"}},
		{"  leading\tand\ntrailing  ", []string{"leading", "and", "trailing"}},
		{"", nil},
		{"!@#$%", nil},
		{"don't", []string{"don", "t"}}, // apostrophe is a separator (plain-text contract)
	}
	for _, c := range cases {
		got := TokenizePlain(c.in)
		if len(got) == 0 && len(c.want) == 0 {
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("TokenizePlain(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

// TestTokenizePlain_vsTokenize pins the documented difference: TokenizePlain does
// NOT split identifiers, where the code-tuned Tokenize does.
func TestTokenizePlain_vsTokenize(t *testing.T) {
	plain := TokenizePlain("getUserName snake_case")
	if !reflect.DeepEqual(plain, []string{"getusername", "snake", "case"}) {
		t.Errorf("plain over-split or wrong: %v", plain)
	}
	// snake_case still splits on '_' under plain (it's a non-alnum separator), but
	// camelCase does not — that's the contract: punctuation splits, case does not.
	code := Tokenize("getUserName")
	if len(code) <= len(TokenizePlain("getUserName")) {
		t.Errorf("expected code Tokenize to split getUserName more than plain: code=%v", code)
	}
}
