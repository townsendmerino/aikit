package bm25

import (
	"math/rand"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/internal/accum"
)

// scoreReference is the pre-item-10/29 scoring expression, character for
// character, kept as the oracle. Items 10 and 29 claim to change only WHERE the
// numbers come from — a precomputed length norm and a narrower posting — never
// the arithmetic. That claim is only worth anything if something checks it.
func (ix *Index) scoreReference(query []string) map[int]float64 {
	out := map[int]float64{}
	for i, term := range query {
		if dupTerm(query, i) {
			continue
		}
		idf := ix.idf(term)
		if idf == 0 {
			continue
		}
		for _, p := range ix.list(ix.entry(term)) {
			var norm float64
			if ix.avgdl > 0 {
				norm = float64(ix.docLen[p.doc]) / ix.avgdl
			}
			denom := float64(p.tf) + ix.K1*(1-ix.B+ix.B*norm)
			out[int(p.doc)] += idf * (float64(p.tf) * (ix.K1 + 1)) / denom
		}
	}
	return out
}

// TestScoring_bitIdenticalToReference requires EXACT equality, not a tolerance.
// A tolerance would pass a genuinely reassociated formula, which is the specific
// thing being ruled out here — BM25 scores feed a ranking, and a 1-ULP drift can
// reorder a tie.
func TestScoring_bitIdenticalToReference(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	vocab := make([]string, 400)
	for i := range vocab {
		vocab[i] = string(rune('a'+i%26)) + string(rune('a'+(i/26)%26)) + string(rune('0'+i%10))
	}
	for _, nDocs := range []int{1, 5, 300} {
		docs := make([][]string, nDocs)
		for d := range docs {
			n := 1 + rng.Intn(200)
			toks := make([]string, n)
			for i := range toks {
				toks[i] = vocab[rng.Intn(len(vocab))]
			}
			docs[d] = toks
		}
		ix := Build(docs)
		// Non-default K1/B too: they are exported, so a caller may tune them
		// after Build, and item 10 deliberately did NOT fold them into the
		// postings for exactly that reason.
		for _, kb := range [][2]float64{{DefaultK1, DefaultB}, {0.9, 0.4}, {2.0, 0.0}, {1.2, 1.0}} {
			ix.K1, ix.B = kb[0], kb[1]
			for trial := range 20 {
				q := make([]string, 1+trial%5)
				for i := range q {
					q[i] = vocab[rng.Intn(len(vocab))]
				}
				want := ix.scoreReference(q)
				got := ix.Scores(q)
				for d, w := range want {
					if got[d] != w {
						t.Fatalf("nDocs=%d K1=%v B=%v q=%v doc=%d: got %v, reference %v",
							nDocs, kb[0], kb[1], q, d, got[d], w)
					}
				}
				for d, g := range got {
					if g != 0 && want[d] != g {
						t.Fatalf("nDocs=%d doc=%d: extra score %v not in reference", nDocs, d, g)
					}
				}
			}
		}
		ix.K1, ix.B = DefaultK1, DefaultB
	}
}

// TestPostingWidth documents the size item 29 is about, so a future field
// addition that silently doubles the scan cost fails here first.
func TestPostingWidth(t *testing.T) {
	if got := int(unsafeSizeofPosting()); got != 8 {
		t.Errorf("posting is %d bytes, want 8 — the scoring scan is memory-bound "+
			"and this struct's width is the scan cost (item 29)", got)
	}
}

func unsafeSizeofPosting() uintptr { return sizeofPosting }

// tokenizeReference is the pre-arena tokenizer: one string allocation per
// lowered token. Kept as the oracle for item 30, which claims to change only
// where the lowered bytes LIVE, never what they are.
func tokenizeReference(text string) []string {
	var out []string
	var scratch []byte
	runStart := -1
	emit := func(run string) {
		compound := lowerString(run, &scratch)
		var parts []string
		if strings.IndexByte(run, '_') >= 0 {
			start := 0
			for i := 0; i <= len(run); i++ {
				if i == len(run) || run[i] == '_' {
					if i > start {
						parts = append(parts, lowerString(run[start:i], &scratch))
					}
					start = i + 1
				}
			}
		} else {
			b := &tokBuffers{}
			camelSplitBytesInto(run, b)
			lowered := string(b.arena)
			for _, r := range b.parts {
				if r.arena {
					parts = append(parts, lowered[r.lo:r.hi])
				} else {
					parts = append(parts, r.s)
				}
			}
		}
		out = append(out, compound)
		if len(parts) >= 2 {
			out = append(out, parts...)
		}
	}
	for i := range len(text) {
		c := text[i]
		if runStart < 0 {
			if isIdentStartByte(c) {
				runStart = i
			}
			continue
		}
		if isIdentContByte(c) {
			continue
		}
		emit(text[runStart:i])
		runStart = -1
	}
	if runStart >= 0 {
		emit(text[runStart:])
	}
	return out
}

// TestTokenize_arenaMatchesPerTokenAllocation gates item 30: the token stream
// must be identical to the per-token-allocation form, including that
// arena-backed tokens compare equal to independently allocated ones and work as
// map keys — the only thing the indexer does with them.
func TestTokenize_arenaMatchesPerTokenAllocation(t *testing.T) {
	inputs := []string{
		"", "   ", "123", "_", "__", "a",
		"parseHTTPResponse and snake_case_name plus CONST_VALUE",
		"XMLHttpRequest getHTTPResponseCode a1b2 IPv6Addr",
		"MixedCASE_with_UNDERSCORES and camelCaseWords x_1_y",
		"ALLUPPER lowercase Mixed",
	}
	if src, err := os.ReadFile("../encoder/bert.go"); err == nil {
		inputs = append(inputs, string(src))
	}
	for _, in := range inputs {
		got := Tokenize(in)
		want := tokenizeReference(in)
		if len(got) != len(want) {
			t.Fatalf("%.40q: %d tokens, reference %d", in, len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("%.40q: token %d = %q, reference %q", in, i, got[i], want[i])
			}
		}
		if (got == nil) != (want == nil) {
			t.Errorf("%.40q: nil-ness differs (got nil=%v, reference nil=%v)", in, got == nil, want == nil)
		}
		m := map[string]int{}
		for _, tok := range got {
			m[tok]++
		}
		for _, tok := range want {
			if m[tok] == 0 {
				t.Fatalf("%.40q: token %q missing when used as a map key", in, tok)
			}
		}
	}
}

// TestOrderTouched_matchesSort gates item 44: the Q-way run merge must produce
// exactly what slices.Sort produced. It is driven through the real scoreQuery so
// the run boundaries come from real posting lists, not a hand-built fixture —
// the bookkeeping (empty runs from terms whose docs were all already touched, an
// odd run count, single-term queries) is the part most likely to be wrong.
func TestOrderTouched_matchesSort(t *testing.T) {
	rng := rand.New(rand.NewSource(44))
	vocab := make([]string, 300)
	for i := range vocab {
		vocab[i] = string(rune('a'+i%26)) + string(rune('a'+(i/26)%26)) + string(rune('0'+i%10))
	}
	for _, nDocs := range []int{1, 2, 17, 500} {
		docs := make([][]string, nDocs)
		for d := range docs {
			n := 1 + rng.Intn(60)
			toks := make([]string, n)
			for i := range toks {
				toks[i] = vocab[rng.Intn(len(vocab))]
			}
			docs[d] = toks
		}
		ix := Build(docs)
		// Query widths 1..8 so the merge sees odd and even run counts, plus
		// duplicate and unknown terms (which contribute no run at all).
		for width := 1; width <= 8; width++ {
			for trial := range 40 {
				q := make([]string, width)
				for i := range q {
					switch {
					case trial%7 == 0 && i > 0:
						q[i] = q[0] // duplicate term: dupTerm skips it, no run
					case trial%5 == 0 && i == width-1:
						q[i] = "zzz_not_in_corpus" // idf==0, no run
					default:
						q[i] = vocab[rng.Intn(len(vocab))]
					}
				}
				a := ix.scoreQuery(q)
				got := append([]int32(nil), a.Touched...)
				accum.Put(a)

				want := append([]int32(nil), got...)
				slices.Sort(want)

				if !slices.Equal(got, want) {
					t.Fatalf("nDocs=%d width=%d q=%v: merged order != sorted order\n got %v\nwant %v",
						nDocs, width, q, got, want)
				}
				// And distinctness, which the merge's exactness argument relies on.
				for i := 1; i < len(got); i++ {
					if got[i] == got[i-1] {
						t.Fatalf("nDocs=%d q=%v: duplicate doc %d in touched set", nDocs, q, got[i])
					}
				}
			}
		}
	}
}
