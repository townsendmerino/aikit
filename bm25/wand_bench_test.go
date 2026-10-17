package bm25

import (
	"fmt"
	"testing"
)

// BenchmarkWAND is item 39's arbiter, and it A/Bs the two implementations in
// ONE binary — TopK takes the pruning path, topKExhaustive is the pre-item-39
// scan over the same index and query. No cross-invocation drift to argue about.
//
// The query shapes matter more here than the corpus size, because pruning is
// entirely a function of how mixed the query's selectivities are:
//
//	head+2 rare  — one near-universal term with two rare ones. The realistic
//	               shape ("the quick brown fox"), and where pruning pays: the
//	               rare terms lift the threshold past anything the common term
//	               can contribute, so almost all of its postings are skipped.
//	head+rare    — the two-term version of the same.
//	3 head       — three equally-common terms. NOTHING to skip: every document
//	               contains all three and every one is a genuine candidate. This
//	               is the shape BenchmarkTopK/common has always used, and it is
//	               the no-regression case, not a win.
//	3 rare       — the selective end. Few postings either way, so this measures
//	               whether the cursor setup costs more than it saves.
//
// Reporting only the first two would flatter the change; reporting only the
// last two would bury it.
func BenchmarkWAND(b *testing.B) {
	ix, _ := buildScaleIndex(200_000, 120, 30_000)
	byDF := termsByDF(ix)
	pick := func(i int) string { return byDF[i] }
	n := len(byDF)

	shapes := []struct {
		name string
		q    []string
	}{
		{"head+2rare", []string{pick(0), pick(n * 3 / 4), pick(n - 5)}},
		{"head+rare", []string{pick(0), pick(n / 4)}},
		{"3head", []string{pick(0), pick(1), pick(2)}},
		{"3rare", []string{pick(n - 1), pick(n - 2), pick(n - 3)}},
	}
	for _, sh := range shapes {
		dfs := make([]int, len(sh.q))
		for i, term := range sh.q {
			dfs[i] = ix.DF(term)
		}
		b.Run(sh.name+"/exhaustive", func(b *testing.B) {
			b.ReportAllocs()
			b.Logf("df=%v, %d docs scored", dfs, ix.exhaustiveScored(sh.q))
			b.ResetTimer()
			for b.Loop() {
				_ = ix.topKExhaustive(sh.q, 10)
			}
		})
		b.Run(sh.name+"/wand", func(b *testing.B) {
			b.ReportAllocs()
			b.Logf("df=%v, %d docs evaluated", dfs, ix.wandEvaluated(sh.q, 10))
			b.ResetTimer()
			for b.Loop() {
				_ = ix.TopK(sh.q, 10)
			}
		})
	}
}

// BenchmarkWANDk sweeps k, because pruning power falls as k grows: a larger k
// means a lower threshold, which means fewer documents can be excluded. A
// change measured only at k=10 would not show where the win ends.
func BenchmarkWANDk(b *testing.B) {
	ix, _ := buildScaleIndex(200_000, 120, 30_000)
	byDF := termsByDF(ix)
	n := len(byDF)
	q := []string{byDF[0], byDF[n*3/4], byDF[n-5]}
	for _, k := range []int{1, 10, 100, 1000} {
		b.Run(fmt.Sprintf("k%d/exhaustive", k), func(b *testing.B) {
			for b.Loop() {
				_ = ix.topKExhaustive(q, k)
			}
		})
		b.Run(fmt.Sprintf("k%d/wand", k), func(b *testing.B) {
			b.Logf("%d docs evaluated", ix.wandEvaluated(q, k))
			b.ResetTimer()
			for b.Loop() {
				_ = ix.TopK(q, k)
			}
		})
	}
}

// BenchmarkWANDQueryLength finds where pruning stops paying as queries get
// longer, which is not a detail: the pivot loop re-orders and re-walks EVERY
// cursor on every iteration, so its per-iteration cost is O(query terms) while
// the exhaustive scan's is O(1) per posting. Somewhere the first overtakes the
// second, and a change measured only at three terms would not say where.
//
// Each query mixes one head term with a growing tail of rarer ones — the shape
// pruning is supposed to be good at — so this isolates length from selectivity.
func BenchmarkWANDQueryLength(b *testing.B) {
	ix, _ := buildScaleIndex(200_000, 120, 30_000)
	byDF := termsByDF(ix)
	n := len(byDF)
	for _, terms := range []int{2, 3, 6, 12, 24} {
		q := []string{byDF[0]}
		for i := 1; i < terms; i++ {
			q = append(q, byDF[n/2+(i*97)%(n/2-1)])
		}
		postings := 0
		for _, t := range q {
			postings += len(ix.list(ix.entry(t)))
		}
		b.Run(fmt.Sprintf("terms%d/exhaustive", terms), func(b *testing.B) {
			b.Logf("%d postings", postings)
			b.ResetTimer()
			for b.Loop() {
				_ = ix.topKExhaustive(q, 10)
			}
		})
		b.Run(fmt.Sprintf("terms%d/wand", terms), func(b *testing.B) {
			b.Logf("%d evaluated", ix.wandEvaluated(q, 10))
			b.ResetTimer()
			for b.Loop() {
				_ = ix.TopK(q, 10)
			}
		})
	}
}
