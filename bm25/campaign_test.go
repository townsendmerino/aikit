package bm25

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash"
	"math"
	"math/rand"
	"runtime"
	"testing"
)

// The memory campaign's BM25 gates (docs/internal/task-memory-campaign.md items 3 and 6) all check one claim:
// a change to how the index is built or stored changes no result. campaignCorpus is the seeded corpus they share
// and campaignHash is the fingerprint of everything a query can observe.

// campaignVocab is a 5,000-term vocabulary; campaignCorpus draws from it with a Zipf skew, so a few terms have
// very long posting lists and most have short ones, as in real text.
func campaignVocab() []string {
	v := make([]string, 5000)
	for i := range v {
		v[i] = fmt.Sprintf("t%04x", i*2654435761%65521)
	}
	return v
}

// campaignCorpus is n seeded documents of 1 to 300 tokens (some empty when empties is set).
func campaignCorpus(n int, empties bool) [][]string {
	vocab := campaignVocab()
	rng := rand.New(rand.NewSource(int64(n)))
	z := rand.NewZipf(rng, 1.1, 2, uint64(len(vocab)-1))
	docs := make([][]string, n)
	for d := range docs {
		if empties && d%97 == 0 {
			continue
		}
		toks := make([]string, 1+rng.Intn(300))
		for i := range toks {
			toks[i] = vocab[z.Uint64()]
		}
		docs[d] = toks
	}
	return docs
}

// campaignQueries is 300 seeded queries of 1 to 6 terms, with repeats, unseen terms and the empty query.
func campaignQueries() [][]string {
	vocab := campaignVocab()
	rng := rand.New(rand.NewSource(77))
	qs := [][]string{{}, {"never-indexed"}}
	for len(qs) < 300 {
		q := make([]string, 1+rng.Intn(6))
		for i := range q {
			q[i] = vocab[rng.Intn(len(vocab))]
		}
		if rng.Intn(5) == 0 {
			q = append(q, q[0]) // a repeated term
		}
		qs = append(qs, q)
	}
	return qs
}

func hashF64(h hash.Hash, v float64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
	h.Write(b[:])
}

func hashInt(h hash.Hash, v int) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(v))
	h.Write(b[:])
}

// campaignHash fingerprints ix: N, IDF and DF of every vocabulary term, and for every campaign query under two
// (K1, B) settings, Scores, TopK at k = 1, 10 and 100, and TopKBatch at k = 10.
func campaignHash(ix *Index) string {
	h := sha256.New()
	hashInt(h, ix.N())
	for _, term := range campaignVocab() {
		hashF64(h, ix.IDF(term))
		hashInt(h, ix.DF(term))
	}
	qs := campaignQueries()
	for _, kb := range [][2]float64{{DefaultK1, DefaultB}, {0.9, 0.4}} {
		ix.K1, ix.B = kb[0], kb[1]
		for _, q := range qs {
			for _, s := range ix.Scores(q) {
				hashF64(h, s)
			}
			for _, k := range []int{1, 10, 100} {
				for _, r := range ix.TopK(q, k) {
					hashInt(h, r.Doc)
					hashF64(h, r.Score)
				}
			}
		}
		for _, rs := range ix.TopKBatch(qs, 10, 4) {
			for _, r := range rs {
				hashInt(h, r.Doc)
				hashF64(h, r.Score)
			}
		}
	}
	ix.K1, ix.B = DefaultK1, DefaultB
	return hex.EncodeToString(h.Sum(nil))
}

// campaignHashBefore is campaignHash(Build(campaignCorpus(20000, true))), captured on the tree before the postings
// arena (task-memory-campaign.md item 3), when each term held its own posting slice.
const campaignHashBefore = "83b381ce0bcbed9bd23d1574a9016de7f373421ff4f242c4b12609137d52561d"

// Items 3 and 6: Build's results are bit-identical to the pre-arena tree's.
func TestBuild_campaignHashMatchesBeforeArena(t *testing.T) {
	got := campaignHash(Build(campaignCorpus(20000, true)))
	t.Logf("campaign hash %s", got)
	if got != campaignHashBefore {
		t.Fatalf("campaign hash %s, before the arena %s: a result changed", got, campaignHashBefore)
	}
}

// indexHeapBytes is the heap a built index holds: HeapAlloc after building it from docs (already resident) and
// GC, minus HeapAlloc before. Min of 3 builds.
func indexHeapBytes(docs [][]string) uint64 {
	best := uint64(math.MaxUint64)
	for range 3 {
		runtime.GC()
		var a, b runtime.MemStats
		runtime.ReadMemStats(&a)
		ix := Build(docs)
		runtime.GC()
		runtime.ReadMemStats(&b)
		runtime.KeepAlive(ix)
		best = min(best, b.HeapAlloc-a.HeapAlloc)
	}
	return best
}

// indexHeapBefore is indexHeapBytes(campaignCorpus(20000, true)) on the pre-arena tree (M1 Pro).
var indexHeapBefore uint64 = 19_568_608

// Item 3, gate (b): the bytes a built index holds, before and after the arena (reported; the bar is "smaller").
func TestBuild_indexBytes(t *testing.T) {
	docs := campaignCorpus(20000, true)
	got := indexHeapBytes(docs)
	t.Logf("built index holds %d B = %.2f MiB (before the arena: %.2f MiB)", got, float64(got)/(1<<20), float64(indexHeapBefore)/(1<<20))
	if indexHeapBefore != 0 && got >= indexHeapBefore {
		t.Fatalf("index holds %d B, not less than before the arena (%d B)", got, indexHeapBefore)
	}
	runtime.KeepAlive(docs)
}
