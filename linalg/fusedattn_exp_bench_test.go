package linalg

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// Audit M-09: AttendTileFused's f64 math.Exp against the contract-exp variant,
// at the kernel rather than through a whole tower.
//
// The tower benchmark dilutes this badly and that is not a measurement artefact
// — it is the shape of the problem. Attention (and its one exp per score
// element) is O(np^2) while the projections and MLP around it are O(np), so the
// exp's share of a tower grows with np. The in-tree tower fixtures stop at
// np=576; the audit's bound is for so400m at np=4096, which nothing in the tree
// benchmarks. This sweep shows the kernel effect directly and how it scales.
func BenchmarkAttendTileFused_expKind(b *testing.B) {
	const hd = 80
	rng := rand.New(rand.NewPCG(9, 9))
	rv := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	for _, np := range []int{196, 576, 1024, 2048} {
		qh, kh, vBlk := rv(np*hd), rv(np*hd), rv(np*hd)
		ch := make([]float32, np*hd)
		sc := NewFusedAttnScratch(np, hd)
		lo, hi := make([]int, np), make([]int, np)
		for i := range np {
			lo[i], hi[i] = 0, np-1 // full (bidirectional) attention, what a ViT runs
		}
		scale := 1.0 / 8.944
		for _, tc := range []struct {
			name string
			fn   func() bool
		}{
			{"f64math", func() bool {
				return AttendTileFused(MatmulBT, qh, kh, vBlk, ch, sc, np, hd, np, scale, lo, hi)
			}},
			{"contract", func() bool {
				return AttendTileFusedContractExp(MatmulBT, qh, kh, vBlk, ch, sc, np, hd, np, scale, lo, hi)
			}},
		} {
			b.Run(fmt.Sprintf("np%d/%s", np, tc.name), func(b *testing.B) {
				for range b.N {
					if !tc.fn() {
						b.Fatal("declined")
					}
				}
				// exp count per call is np*np (one per score element).
				b.ReportMetric(float64(np)*float64(np)/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "Gexp/s")
			})
		}
	}
}
