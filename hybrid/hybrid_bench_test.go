package hybrid

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/bm25"
)

func benchmarkCorpus(nDocs, d int, seed int64) (*ann.Flat, *bm25.Index) {
	rng := rand.New(rand.NewSource(seed))
	vecs := make([][]float32, nDocs)
	docs := make([][]string, nDocs)
	vocab := []string{"apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon"}

	for i := 0; i < nDocs; i++ {
		v := make([]float32, d)
		for j := 0; j < d; j++ {
			v[j] = float32(rng.NormFloat64())
		}
		vecs[i] = v

		tokCount := 5 + rng.Intn(10)
		toks := make([]string, tokCount)
		for j := 0; j < tokCount; j++ {
			toks[j] = vocab[rng.Intn(len(vocab))]
		}
		docs[i] = toks
	}
	return ann.New(vecs), bm25.Build(docs)
}

func BenchmarkHybridQuery(b *testing.B) {
	dense, lexical := benchmarkCorpus(1000, 64, 42)
	r := New(dense, lexical)

	queryVec := make([]float32, 64)
	for i := range queryVec {
		queryVec[i] = 1.0
	}
	queryTokens := []string{"apple", "banana", "cherry"}

	for _, k := range []int{10, 50, 100} {
		b.Run(fmt.Sprintf("k=%d", k), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				_ = r.Query(queryVec, queryTokens, k)
			}
		})
	}
}
