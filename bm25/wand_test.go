package bm25

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"

	"github.com/townsendmerino/aikit/internal/accum"
)

// wandCorpus builds a synthetic corpus with a realistic term distribution: a
// Zipf-ish vocabulary, so a few terms appear in most documents and most terms
// appear in a handful. That skew IS the thing WAND exploits — a uniform
// vocabulary gives every term the same upper bound, the pivot never advances
// past the first cursor, and the algorithm degenerates into the exhaustive scan
// while still passing every correctness test.
func wandCorpus(nDocs, vocab, docLen int, seed int64) [][]string {
	rng := rand.New(rand.NewSource(seed))
	terms := make([]string, vocab)
	for i := range terms {
		terms[i] = fmt.Sprintf("t%05d", i)
	}
	docs := make([][]string, nDocs)
	for d := range docs {
		// Vary document length: the length normalization only matters, and the
		// minNorm half of the bound is only exercised, when lengths differ.
		n := docLen/2 + rng.Intn(docLen)
		toks := make([]string, n)
		for i := range toks {
			// Zipf-ish: index ~ vocab * u^3 concentrates draws near 0.
			u := rng.Float64()
			toks[i] = terms[int(float64(vocab)*u*u*u)%vocab]
		}
		docs[d] = toks
	}
	return docs
}

// TestWAND_matchesExhaustive is item 39's gate, and it is EXACT rather than
// statistical: dynamic pruning must return the same documents with the same
// scores in the same order, bit for bit, as scoring everything.
//
// Two properties have to hold together for that, and the shapes below are
// chosen to stress both. Pruning must never discard a document that belonged
// (the upper bound must be a real bound), and the surviving candidates must be
// summed in QUERY order so the float64 low bits — and therefore which member of
// a tie is kept — match the term-at-a-time loop.
func TestWAND_matchesExhaustive(t *testing.T) {
	rng := rand.New(rand.NewSource(39))
	for _, corpus := range []struct {
		name                 string
		nDocs, vocab, docLen int
		seed                 int64
	}{
		{"small", 500, 200, 30, 1},
		{"skewed", 20_000, 3_000, 60, 2},
		{"tiny-vocab", 2_000, 20, 40, 3}, // every term is a head term
		{"long-docs", 1_000, 500, 400, 4},
	} {
		ix := Build(wandCorpus(corpus.nDocs, corpus.vocab, corpus.docLen, corpus.seed))
		terms := make([]string, 0, len(ix.terms))
		for term := range ix.terms {
			terms = append(terms, term)
		}
		for _, k1b := range []struct{ k1, b float64 }{
			{DefaultK1, DefaultB},
			{0, 0},   // no tf saturation, no length normalization
			{2.5, 1}, // full length normalization
			{1.2, 0.4},
		} {
			ix.K1, ix.B = k1b.k1, k1b.b
			for range 200 {
				// Queries of 1..6 terms, including repeats and unknown terms.
				q := make([]string, 1+rng.Intn(6))
				for i := range q {
					switch rng.Intn(10) {
					case 0:
						q[i] = "zzz-not-in-corpus"
					case 1:
						q[i] = q[0] // a duplicate term must contribute once
					default:
						q[i] = terms[rng.Intn(len(terms))]
					}
				}
				for _, k := range []int{0, 1, 5, 10, 100} {
					got := ix.TopK(q, k)
					want := ix.topKExhaustive(q, k)
					if len(got) != len(want) {
						t.Fatalf("%s k1=%v b=%v q=%v k=%d: pruned returned %d results, exhaustive %d",
							corpus.name, k1b.k1, k1b.b, q, k, len(got), len(want))
					}
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("%s k1=%v b=%v q=%v k=%d result %d: pruned %+v, exhaustive %+v",
								corpus.name, k1b.k1, k1b.b, q, k, i, got[i], want[i])
						}
					}
				}
			}
		}
		ix.K1, ix.B = DefaultK1, DefaultB
	}
}

// TestWAND_actuallyPrunes is the other half of the gate, and the one that would
// catch a "correct" implementation that never skips anything. An exactness test
// alone is satisfied by falling back to the exhaustive scan on every query —
// which is exactly what a broken pivot rule degrades into.
//
// It counts documents evaluated rather than timing anything, so it is a
// property, not a benchmark.
//
// The query shapes are the point. Pruning can only skip a term's postings once
// the threshold exceeds that term's upper bound, which needs the query to MIX
// selectivities: a common term next to a rare one, so the rare one lifts the
// threshold past what the common one could ever contribute alone. A query of
// three equally-common terms has nothing to skip — every document contains all
// three and every one is a genuine candidate — so it is here as a
// no-regression case, not as a win.
func TestWAND_actuallyPrunes(t *testing.T) {
	ix := Build(wandCorpus(50_000, 5_000, 60, 39))
	byDF := termsByDF(ix)
	pick := func(i int) string { return byDF[i] }

	for _, tc := range []struct {
		name    string
		q       []string
		maxFrac float64 // evaluated / exhaustively-scored must not exceed this
	}{
		{"head+2 rare", []string{pick(0), pick(len(byDF) * 3 / 4), pick(len(byDF) - 5)}, 0.10},
		{"head+rare", []string{pick(0), pick(len(byDF) / 4)}, 0.20},
		{"head+mid+rare", []string{pick(0), pick(len(byDF) / 3), pick(len(byDF) * 4 / 5)}, 0.15},
		// No mixing, so nothing to prune. Must not do MORE work than the scan.
		{"3 head (no win expected)", []string{pick(0), pick(1), pick(2)}, 1.0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			scored := ix.exhaustiveScored(tc.q)
			evaluated := ix.wandEvaluated(tc.q, 10)
			frac := float64(evaluated) / float64(scored)
			dfs := make([]int, len(tc.q))
			for i, term := range tc.q {
				dfs[i] = ix.DF(term)
			}
			t.Logf("df=%v: %d documents evaluated of %d scored exhaustively (%.1f%%)",
				dfs, evaluated, scored, 100*frac)
			if frac > tc.maxFrac {
				t.Errorf("evaluated %.1f%% of the scored set, want <= %.1f%%", 100*frac, 100*tc.maxFrac)
			}
		})
	}
}

// termsByDF returns every term, most frequent first.
func termsByDF(ix *Index) []string {
	out := make([]string, 0, len(ix.terms))
	for term := range ix.terms {
		out = append(out, term)
	}
	sort.Slice(out, func(i, j int) bool {
		if ix.DF(out[i]) != ix.DF(out[j]) {
			return ix.DF(out[i]) > ix.DF(out[j])
		}
		return out[i] < out[j] // deterministic across map iteration orders
	})
	return out
}

// TestWAND_declinesWhenBoundUnsound checks the guard rather than trusting it. A
// negative K1 or B breaks the monotonicity the upper bound is derived from, so
// the pruning path must refuse and TopK must still be correct.
func TestWAND_declinesWhenBoundUnsound(t *testing.T) {
	ix := Build(wandCorpus(2_000, 300, 40, 7))
	q := headTerms(ix, 3)
	// B > 1 joins K1 < 0 and B < 0 here (audit C-03): outside the standard
	// 0 <= B <= 1 range the denominator can change sign across a posting list,
	// so the bound evaluated at minNorm bounds nothing. See wandUsable.
	for _, tc := range []struct{ k1, b float64 }{{-1, 0.75}, {1.5, -0.2}, {1.5, 1.5}, {1.5, 2.0}} {
		ix.K1, ix.B = tc.k1, tc.b
		if _, ok := ix.topKWAND(q, 10); ok {
			t.Errorf("K1=%v B=%v: pruning accepted a query whose bound is unsound", tc.k1, tc.b)
		}
		got, want := ix.TopK(q, 10), ix.topKExhaustive(q, 10)
		if len(got) != len(want) {
			t.Fatalf("K1=%v B=%v: %d results vs %d", tc.k1, tc.b, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("K1=%v B=%v result %d: %+v vs %+v", tc.k1, tc.b, i, got[i], want[i])
			}
		}
	}
	// There is no longer a "postings but no stats" state to guard against: the
	// bound's extrema live on termEntry, so any term Build indexed carries them.
	// What remains is a zero-value Index, which must not panic on either path.
	ix.K1, ix.B = DefaultK1, DefaultB
	var zero Index
	zero.K1, zero.B = DefaultK1, DefaultB
	if got := zero.TopK([]string{"anything"}, 10); len(got) != 0 {
		t.Errorf("zero-value Index returned %d results", len(got))
	}
	if got := zero.topKExhaustive([]string{"anything"}, 10); len(got) != 0 {
		t.Errorf("zero-value Index exhaustive scan returned %d results", len(got))
	}
}

// TestWAND_degenerateQueries covers the shapes that reach the pivot loop with
// nothing to pivot on.
func TestWAND_degenerateQueries(t *testing.T) {
	ix := Build(wandCorpus(500, 100, 30, 11))
	for _, tc := range []struct {
		name string
		q    []string
	}{
		{"empty query", nil},
		{"all unknown", []string{"nope", "nada"}},
		{"single term", headTerms(ix, 1)},
		{"all duplicates", []string{headTerms(ix, 1)[0], headTerms(ix, 1)[0], headTerms(ix, 1)[0]}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, want := ix.TopK(tc.q, 10), ix.topKExhaustive(tc.q, 10)
			if len(got) != len(want) {
				t.Fatalf("%d results vs %d", len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("result %d: %+v vs %+v", i, got[i], want[i])
				}
			}
		})
	}
}

// TestAdvanceTo checks the galloping skip directly, including the boundaries
// the pivot loop rarely reaches but depends on: a target past the end, a target
// already at the cursor, and every position in a short list.
func TestAdvanceTo(t *testing.T) {
	post := make([]posting, 0, 64)
	for i := range 64 {
		post = append(post, posting{doc: int32(i * 3), tf: 1})
	}
	for i := range len(post) {
		for target := int32(0); target <= int32(len(post)*3+2); target++ {
			got := advanceTo(post, i, target)
			// Reference: linear scan from i.
			want := i
			for want < len(post) && post[want].doc < target {
				want++
			}
			if got != want {
				t.Fatalf("advanceTo(i=%d, target=%d) = %d, want %d", i, target, got, want)
			}
		}
	}
	if got := advanceTo(post, len(post), 0); got != len(post) {
		t.Errorf("advanceTo from an exhausted cursor = %d, want %d", got, len(post))
	}
}

// wandEvaluated runs the pruning path with its own scratch and reports how many
// documents it fully scored.
func (ix *Index) wandEvaluated(query []string, k int) int {
	w := new(wandState)
	if _, ok := ix.topKWANDState(query, k, w); !ok {
		panic("bm25: pruning declined the query")
	}
	return w.evaluated
}

// exhaustiveScored reports how many documents the term-at-a-time path scores —
// the quantity pruning is trying to reduce.
func (ix *Index) exhaustiveScored(query []string) int {
	a := ix.scoreQuery(query)
	defer accum.Put(a)
	return len(a.Touched)
}

// TestWAND_declinesLongQueries pins the length guard. Past maxWandTerms the
// pruning path must decline — and TopK must still be correct, which is the part
// a guard can silently get wrong by declining and then falling through to
// nothing.
func TestWAND_declinesLongQueries(t *testing.T) {
	ix := Build(wandCorpus(5_000, 1_000, 40, 13))
	byDF := termsByDF(ix)
	q := make([]string, 0, maxWandTerms+4)
	for i := range maxWandTerms + 4 {
		q = append(q, byDF[i*7])
	}
	if _, ok := ix.topKWAND(q, 10); ok {
		t.Errorf("pruning accepted a %d-term query; maxWandTerms is %d", len(q), maxWandTerms)
	}
	// Exactly at the limit it must still be used.
	if _, ok := ix.topKWAND(q[:maxWandTerms], 10); !ok {
		t.Errorf("pruning declined a %d-term query, which is within maxWandTerms", maxWandTerms)
	}
	got, want := ix.TopK(q, 10), ix.topKExhaustive(q, 10)
	if len(got) != len(want) {
		t.Fatalf("%d results vs %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("result %d: %+v vs %+v", i, got[i], want[i])
		}
	}
}

// TestWAND_bGreaterThanOneMatchesExhaustive is the C-03 regression, kept
// separate from the table above because it needs a corpus shaped to expose the
// defect rather than the uniform one wandCorpus builds: 200 long documents set
// avgdl high, and 20 two-token documents sharing a term then sit at a norm
// small enough that tf + K1*(1-B+B*norm) is negative for them while positive
// for the long ones. With B = 2 and the old `B >= 0` admission this made TopK
// return ZERO results where the exhaustive scan returns ten — silent and total,
// which is why the guard is a correctness fix and not a tuning one.
func TestWAND_bGreaterThanOneMatchesExhaustive(t *testing.T) {
	var docs [][]string
	for range 200 {
		d := make([]string, 0, 401)
		for range 400 {
			d = append(d, "filler")
		}
		docs = append(docs, append(d, "target"))
	}
	for range 20 {
		docs = append(docs, []string{"target", "rare"})
	}
	ix := Build(docs)
	q := []string{"target"}
	for _, b := range []float64{1.0, 1.000001, 1.5, 2.0, 10.0} {
		ix.K1, ix.B = 1.5, b
		got, want := ix.TopK(q, 10), ix.topKExhaustive(q, 10)
		if len(got) != len(want) {
			t.Fatalf("B=%v: TopK returned %d results, exhaustive %d", b, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("B=%v result %d: TopK %+v vs exhaustive %+v", b, i, got[i], want[i])
			}
		}
	}
}
