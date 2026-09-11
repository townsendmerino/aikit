package late

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// BenchmarkMaxSim covers the shape ColBERT-style late interaction actually
// runs: a short query token matrix against a document's token matrix, scored
// as a max over every (query, doc) pair. The audit noted `late` had no
// benchmark at all, which is why M-20 sat unmeasured.
//
// nQ=32 is a typical expanded query; nDoc sweeps a short passage to a long
// chunk. d=128 is ColBERT's projected token dim.
func BenchmarkMaxSim(b *testing.B) {
	const d = 128
	rng := rand.New(rand.NewPCG(11, 13))
	mk := func(n int) [][]float32 {
		out := make([][]float32, n)
		for i := range out {
			v := make([]float32, d)
			for j := range v {
				v[j] = float32(rng.NormFloat64())
			}
			out[i] = v
		}
		return out
	}
	q := mk(32)
	for _, nDoc := range []int{64, 256, 1024} {
		doc := mk(nDoc)
		b.Run(fmt.Sprintf("nQ32/nDoc%d", nDoc), func(b *testing.B) {
			b.SetBytes(int64(len(q)) * int64(nDoc) * int64(d) * 4)
			for b.Loop() {
				sinkMaxSim = MaxSim(q, doc)
			}
		})
	}
}

var sinkMaxSim float64
