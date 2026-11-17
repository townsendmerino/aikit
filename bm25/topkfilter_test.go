package bm25

import (
	"math/rand"
	"slices"
	"testing"
)

// bruteFilter is the reference TopKFilter is held to: score every document, then
// filter, then select. topKExhaustive(q, -1) is the full ranking, and its scores
// are summed in query order exactly as WAND's are, so this is bit-comparable.
func bruteFilter(ix *Index, q []string, k int, allow func(int) bool) []Result {
	out := []Result{}
	for _, r := range ix.topKExhaustive(q, -1) {
		if allow(r.Doc) {
			out = append(out, r)
		}
	}
	if len(out) > k {
		out = out[:k]
	}
	return out
}

// TestTopKFilter_matchesBruteForce is the property: on randomized corpora, query
// shapes, parameters, k and scopes, TopKFilter returns exactly what
// score-all-then-filter returns (documents, scores, order), with IDF still
// computed over the whole corpus.
func TestTopKFilter_matchesBruteForce(t *testing.T) {
	rng := rand.New(rand.NewSource(1117))
	for _, corpus := range []struct {
		name                 string
		nDocs, vocab, docLen int
		seed                 int64
	}{
		{"small", 500, 200, 30, 1},
		{"skewed", 20_000, 3_000, 60, 2},
		{"tiny-vocab", 2_000, 20, 40, 3},
		{"long-docs", 1_000, 500, 400, 4},
	} {
		ix := Build(wandCorpus(corpus.nDocs, corpus.vocab, corpus.docLen, corpus.seed))
		terms := make([]string, 0, len(ix.terms))
		for term := range ix.terms {
			terms = append(terms, term)
		}
		slices.Sort(terms)
		for _, k1b := range []struct{ k1, b float64 }{{DefaultK1, DefaultB}, {0, 0}, {2.5, 1}, {1.2, 0.4}} {
			ix.K1, ix.B = k1b.k1, k1b.b
			for range 120 {
				q := make([]string, 1+rng.Intn(10)) // up to 10: crosses maxWandTerms
				for i := range q {
					switch rng.Intn(10) {
					case 0:
						q[i] = "zzz-not-in-corpus"
					case 1:
						q[i] = q[0]
					default:
						q[i] = terms[rng.Intn(len(terms))]
					}
				}
				// Scope: random fraction of docs, or a contiguous run.
				var allow func(int) bool
				if rng.Intn(2) == 0 {
					frac := []float64{0, 0.01, 0.1, 0.5, 0.95, 1}[rng.Intn(6)]
					set := make([]bool, ix.N())
					for i := range set {
						set[i] = rng.Float64() < frac
					}
					allow = func(d int) bool { return set[d] }
				} else {
					lo := rng.Intn(ix.N())
					hi := lo + rng.Intn(ix.N()/5+1)
					allow = func(d int) bool { return d >= lo && d < hi }
				}
				for _, k := range []int{0, 1, 5, 10, 100} {
					got := ix.TopKFilter(q, k, allow)
					want := bruteFilter(ix, q, k, allow)
					if !slices.Equal(got, want) {
						t.Fatalf("%s k1=%v b=%v q=%v k=%d:\n got %v\nwant %v", corpus.name, k1b.k1, k1b.b, q, k, got, want)
					}
				}
			}
		}
		ix.K1, ix.B = DefaultK1, DefaultB
	}
}

func TestTopKFilter_emptyAndNilAllow(t *testing.T) {
	ix := Build(wandCorpus(2_000, 300, 40, 7))
	q := termsByDF(ix)[:3]
	none := func(int) bool { return false }
	for _, k := range []int{0, 1, 10, -1} {
		if got := ix.TopKFilter(q, k, none); len(got) != 0 {
			t.Fatalf("k=%d: empty allow set returned %v", k, got)
		}
	}
	if got, want := ix.TopKFilter(q, 10, nil), ix.TopK(q, 10); !slices.Equal(got, want) {
		t.Fatalf("nil allow != TopK:\n got %v\nwant %v", got, want)
	}
	// k<0 asks for every allowed scorer.
	even := func(d int) bool { return d%2 == 0 }
	if got, want := ix.TopKFilter(q, -1, even), bruteFilter(ix, q, 1<<30, even); !slices.Equal(got, want) {
		t.Fatalf("k=-1 mismatch: %d vs %d results", len(got), len(want))
	}
}

// TestTopKFilter_skipsDisallowedWithoutScoring pins the point of the method: a
// disallowed pivot is never scored, so WAND's evaluated count cannot exceed the
// number of allowed documents holding a query term.
func TestTopKFilter_skipsDisallowedWithoutScoring(t *testing.T) {
	ix := Build(wandCorpus(20_000, 3_000, 60, 2))
	byDF := termsByDF(ix)
	q := []string{byDF[0], byDF[len(byDF)/2]}
	allow := func(d int) bool { return d%20 == 0 }
	w := new(wandState)
	if _, ok := ix.topKWANDStateFilter(q, 10, w, allow); !ok {
		t.Fatal("WAND declined")
	}
	touching := 0
	for d := 0; d < ix.N(); d += 20 {
		if ix.Scores(q)[d] > 0 {
			touching++
		}
	}
	if w.evaluated > touching {
		t.Fatalf("evaluated %d documents, but only %d allowed documents hold a query term", w.evaluated, touching)
	}
}

// BenchmarkTopKFilter reports unfiltered TopK, TopKFilter and the exhaustive
// score-then-filter it replaces, for a 10% scope (scattered and clustered), on the
// shapes of BenchmarkWAND. It is a benchmark rather than a test because a scope
// cannot make a mixed-selectivity query as cheap as the unfiltered one (theta
// stays near the k-th best head-only score), and a wall-clock assertion would
// flake on shared CI runners; the deterministic property is
// TestTopKFilter_skipsDisallowedWithoutScoring.
func BenchmarkTopKFilter(b *testing.B) {
	ix, _ := buildScaleIndex(200_000, 120, 30_000)
	byDF := termsByDF(ix)
	n := len(byDF)
	scopes := []struct {
		name  string
		allow func(int) bool
	}{
		{"scattered10pct", func(d int) bool { return d%10 == 0 }},
		{"clustered10pct", func(d int) bool { return d >= 50_000 && d < 70_000 }},
	}
	for _, sh := range []struct {
		name string
		q    []string
	}{
		{"3head", []string{byDF[0], byDF[1], byDF[2]}},
		{"head+2rare", []string{byDF[0], byDF[n*3/4], byDF[n-5]}},
	} {
		b.Run(sh.name+"/unfiltered", func(b *testing.B) {
			for b.Loop() {
				ix.TopK(sh.q, 10)
			}
		})
		for _, sc := range scopes {
			b.Run(sh.name+"/"+sc.name+"/TopKFilter", func(b *testing.B) {
				for b.Loop() {
					ix.TopKFilter(sh.q, 10, sc.allow)
				}
			})
			b.Run(sh.name+"/"+sc.name+"/exhaustive", func(b *testing.B) {
				for b.Loop() {
					ix.topKExhaustiveFilter(sh.q, 10, sc.allow)
				}
			})
		}
	}
}
