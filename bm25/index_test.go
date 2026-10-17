package bm25

import (
	"math"
	"testing"
)

func TestBM25_RanksRelevantDocFirst(t *testing.T) {
	docs := [][]string{
		Tokenize("the cat sat on the mat"),
		Tokenize("a fast quick brown fox jumps"),
		Tokenize("quick quick quick sort algorithm"),
		Tokenize("unrelated content about databases"),
	}
	ix := Build(docs)
	if ix.N() != 4 {
		t.Fatalf("N = %d, want 4", ix.N())
	}

	got := ix.TopK(Tokenize("quick"), 10)
	if len(got) != 2 {
		t.Fatalf("TopK(quick) returned %d docs, want 2 (%v)", len(got), got)
	}
	// Doc 2 has tf=3 for "quick"; doc 1 has tf=1 → doc 2 ranks first.
	if got[0].Doc != 2 {
		t.Errorf("top doc = %d, want 2 (higher tf); results=%v", got[0].Doc, got)
	}
	if got[0].Score <= got[1].Score {
		t.Errorf("scores not descending: %v", got)
	}
}

func TestBM25_IDFNonNegativeAndRareTermWeightsMore(t *testing.T) {
	docs := [][]string{
		Tokenize("common common rare"),
		Tokenize("common common common"),
		Tokenize("common nothing"),
	}
	ix := Build(docs)
	if idf := ix.idf("common"); idf < 0 {
		t.Errorf("idf(common) = %v, must be non-negative (Lucene variant)", idf)
	}
	if ix.idf("rare") <= ix.idf("common") {
		t.Errorf("rare term must out-weigh common: idf(rare)=%v idf(common)=%v",
			ix.idf("rare"), ix.idf("common"))
	}
	if ix.idf("absent") != 0 {
		t.Errorf("idf(absent term) = %v, want 0", ix.idf("absent"))
	}
}

func TestBM25_EmptyCorpus(t *testing.T) {
	ix := Build(nil)
	if ix.N() != 0 {
		t.Fatalf("N = %d, want 0", ix.N())
	}
	if got := ix.TopK(Tokenize("anything"), 5); len(got) != 0 {
		t.Errorf("TopK on empty corpus = %v, want empty", got)
	}
}

// TestBuild_termEntryMatchesBruteForce is the gate for the single-map rewrite
// (lens doc §3.7). Build now maintains four things per term in one pass through
// one map — the posting list, the document frequency, and the two extrema item
// 39's pruning bound is reconstructed from — where it used to maintain them
// across three maps and a second pass over every posting list.
//
// It is checked structurally, against an independent recount, rather than
// differentially against the previous implementation. A differential test would
// need the old code kept alive; this states what the fields MEAN, which is what
// a future rewrite has to preserve, and it catches the WAND extrema too — those
// have no other direct test, only the bound's end-to-end exactness gate, which
// a too-LARGE bound would satisfy while silently costing pruning power.
func TestBuild_termEntryMatchesBruteForce(t *testing.T) {
	docs := [][]string{
		{"alpha", "beta", "alpha", "gamma"},
		{"beta"},
		{},
		{"gamma", "gamma", "gamma", "gamma", "delta", "alpha"},
		{"alpha", "alpha"},
		{"epsilon"},
	}
	ix := Build(docs)

	// Every term the corpus contains must be indexed, and no others.
	want := map[string]bool{}
	for _, d := range docs {
		for _, tok := range d {
			want[tok] = true
		}
	}
	if len(ix.terms) != len(want) {
		t.Fatalf("indexed %d terms, corpus has %d distinct", len(ix.terms), len(want))
	}
	for term := range want {
		e := ix.entry(term)
		if e == nil {
			t.Fatalf("term %q is in the corpus but not indexed", term)
		}

		// Brute-force recount.
		var wantPostings []posting
		wantMaxTf, wantMinLen := int32(0), int32(math.MaxInt32)
		for d, toks := range docs {
			n := int32(0)
			for _, tok := range toks {
				if tok == term {
					n++
				}
			}
			if n == 0 {
				continue
			}
			wantPostings = append(wantPostings, posting{doc: int32(d), tf: n})
			wantMaxTf = max(wantMaxTf, n)
			wantMinLen = min(wantMinLen, int32(len(toks)))
		}

		if int(e.df) != len(wantPostings) {
			t.Errorf("%q: df %d, want %d", term, e.df, len(wantPostings))
		}
		if len(ix.list(e)) != len(wantPostings) {
			t.Fatalf("%q: %d postings, want %d", term, len(ix.list(e)), len(wantPostings))
		}
		for i := range wantPostings {
			if ix.list(e)[i] != wantPostings[i] {
				t.Errorf("%q posting %d: %+v, want %+v", term, i, ix.list(e)[i], wantPostings[i])
			}
			// Doc-ordered, which the WAND cursors and the touched-run merge both
			// depend on and neither would obviously fail without.
			if i > 0 && ix.list(e)[i].doc <= ix.list(e)[i-1].doc {
				t.Errorf("%q: postings not strictly doc-ordered at %d", term, i)
			}
		}
		if e.maxTf != wantMaxTf {
			t.Errorf("%q: maxTf %d, want %d", term, e.maxTf, wantMaxTf)
		}
		if e.minLen != wantMinLen {
			t.Errorf("%q: minLen %d, want %d", term, e.minLen, wantMinLen)
		}
		// minNorm must be exactly the norm of the shortest document holding the
		// term — the bound's monotonicity argument rests on that identity.
		var wantMinNorm float64
		if ix.avgdl > 0 {
			wantMinNorm = math.MaxFloat64
			for _, p := range wantPostings {
				wantMinNorm = min(wantMinNorm, ix.norm[p.doc])
			}
		}
		if got := ix.minNorm(e); got != wantMinNorm {
			t.Errorf("%q: minNorm %v, want %v (the norm of its shortest document)", term, got, wantMinNorm)
		}
	}
	if ix.entry("not-in-corpus") != nil {
		t.Error("entry returned non-nil for an unindexed term")
	}
	if ix.DF("not-in-corpus") != 0 {
		t.Error("DF returned non-zero for an unindexed term")
	}
}
