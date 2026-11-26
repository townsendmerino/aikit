package bitvec

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// BenchmarkHammingRows sizes perf-campaign item 38's first stage. Three things
// need separating, and a single end-to-end number would conflate them:
//
//   - the POPCNTQ kernel versus the portable SWAR popcount, which is the cost
//     of GOAMD64=v1 being the default and the reason hamming_amd64.s exists;
//   - either of those versus an exact float32 dot scan over the same corpus,
//     which is the ratio the item's "~10× first stage" claims;
//   - both at the real dims (256 Model2Vec, 768 CodeRankEmbed), since the win
//     is a memory-traffic ratio and dim sets it.
//
// ns/vec is the reported metric rather than ns/op: the whole point is the
// per-candidate cost, and it makes the three variants directly comparable at a
// glance without dividing by n in your head.
func BenchmarkHammingRows(b *testing.B) {
	const n = 100_000
	for _, dim := range []int{256, 768} {
		words := PackedWords(dim)
		rng := rand.New(rand.NewSource(int64(dim)))
		codes := make([]uint64, n*words)
		for i := range codes {
			codes[i] = rng.Uint64()
		}
		q := make([]uint64, words)
		for i := range q {
			q[i] = rng.Uint64()
		}
		dst := make([]uint16, n)

		// The exact float32 scan the prefilter is meant to shrink, built over
		// the same n and dim so the comparison is like for like.
		flat := make([]float32, n*dim)
		for i := range flat {
			flat[i] = float32(rng.NormFloat64())
		}
		qf := make([]float32, dim)
		for i := range qf {
			qf[i] = float32(rng.NormFloat64())
		}
		scores := make([]float32, n)

		perVec := func(b *testing.B) {
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(n), "ns/vec")
		}

		b.Run(fmt.Sprintf("d%d/popcnt", dim), func(b *testing.B) {
			b.SetBytes(int64(n) * int64(words) * 8)
			for b.Loop() {
				HammingRows(q, codes, words, n, dst)
			}
			perVec(b)
		})
		b.Run(fmt.Sprintf("d%d/generic", dim), func(b *testing.B) {
			b.SetBytes(int64(n) * int64(words) * 8)
			for b.Loop() {
				hammingRowsGeneric(q, codes, words, n, dst)
			}
			perVec(b)
		})
		b.Run(fmt.Sprintf("d%d/exactdot", dim), func(b *testing.B) {
			b.SetBytes(int64(n) * int64(dim) * 4)
			for b.Loop() {
				linalg.MatmulBT(qf, flat, scores, 1, dim, n)
			}
			perVec(b)
		})
	}
}
