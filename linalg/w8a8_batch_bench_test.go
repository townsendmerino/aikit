package linalg

import (
	"fmt"
	"math/rand"
	"testing"
)

// Baseline for audit M-01 (docs/audit-2026-09-10.md §1.A): MatmulBTW8A8Batch's
// span runs its own per-(row, column) dotI8 loop and never reaches the S-01b
// W8A8 tile that w8a8Span dispatches to at M>=4. These are goinfer's actual
// batched prefill shapes on Qwen2.5-Coder-1.5B (hidden 1536, inter 8960,
// 12 q heads / 2 kv heads at hd 128), which is where the bypass is reached:
//
//	q‖k‖v   K=1536, N = 1536 + 256 + 256
//	gate‖up K=1536, N = 8960 + 8960
//
// M=1 is the control: the tile does not engage there, so it must not move.
// The M>=4 rows are where the recorded 3.5-3.9x (arm64) / 1.17-1.44x (amd64)
// is expected to appear once the span is routed through w8a8Span.
func BenchmarkMatmulBTW8A8Batch_prefill(b *testing.B) {
	const K = 1536
	groups := []struct {
		name string
		ns   []int
	}{
		{"qkv", []int{1536, 256, 256}},
		{"gateup", []int{8960, 8960}},
	}
	rng := rand.New(rand.NewSource(11))
	for _, g := range groups {
		ops := make([]W8A8Op, len(g.ns))
		for i, n := range g.ns {
			w := make([]float32, n*K)
			for j := range w {
				w[j] = float32(rng.NormFloat64())
			}
			q, scales := QuantizeRowsInt8(w, n, K)
			ops[i] = W8A8Op{BQ: q, Scales: scales, N: n}
		}
		for _, m := range []int{1, 4, 8, 32, 128} {
			a := make([]float32, m*K)
			for j := range a {
				a[j] = float32(rng.NormFloat64())
			}
			// Dst is per-M, so rebuild the op slice with correctly sized dsts.
			cur := make([]W8A8Op, len(ops))
			copy(cur, ops)
			var macs int64
			for i := range cur {
				cur[i].Dst = make([]float32, m*cur[i].N)
				macs += int64(m) * int64(K) * int64(cur[i].N)
			}
			b.Run(fmt.Sprintf("%s/M%d", g.name, m), func(b *testing.B) {
				var ws Workspace
				b.ResetTimer()
				for range b.N {
					MatmulBTW8A8Batch(&ws, a, m, K, cur)
				}
				// GMAC/s is the comparable number across M and group.
				b.ReportMetric(float64(macs)/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "GMAC/s")
			})
		}
	}
}
