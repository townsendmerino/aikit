package linalg

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// BenchmarkW8A8SpanShapes exists because v1.17.0 shipped a W8A8 regression that a
// single-shape benchmark could not see, and it covers BOTH regimes on purpose.
//
// The eight-column kernel v1.17.0 shipped (dotI8Cols8, deleted in v1.17.1) was
// measured at K=768 with a small N — where
// B is cache-resident — and reported +30%. At the shapes production actually uses,
// where B is far larger than L3 and is streamed from DRAM, the same kernel was 3.5-5%
// SLOWER, because it advances eight streams K bytes apart where the shipped span walks
// B linearly. goinfer measured ~3% end-to-end on decode from it.
//
// The lesson is not "that kernel was bad". It is that an int8 matmul has at least two
// regimes and a benchmark that samples one of them is not evidence about the other.
// The shapes below straddle the boundary deliberately:
//
//	cache-resident : B under ~32 MB. Arithmetic efficiency dominates.
//	streamed       : B far above it. Access pattern dominates.
//
// Anything proposing to change w8a8Span should show numbers for every row here, and
// should expect the two halves to disagree.
// regimeTag names the memory regime a W8A8 shape falls in from B's byte size (int8 weights,
// 1 byte each) against a nominal ~32 MB last-level cache. "resident" is where arithmetic
// efficiency dominates and a wider kernel wins; "streamed" is where the access pattern
// dominates and the same kernel can lose. The boundary is nominal, not the exact LLC of any
// one box — the point is that both sides are always sampled, not that the split is precise.
func regimeTag(K, N int) string {
	const nominalLLC = 32 << 20 // bytes
	if K*N < nominalLLC {
		return "resident"
	}
	return "streamed"
}

func BenchmarkW8A8SpanShapes(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	for _, s := range []struct {
		K, N int
		note string
	}{
		{768, 8192, "cache-resident; the shape v1.17.0's +30% was measured at"},
		{2048, 2048, "cache-resident"},
		{3584, 4096, "cache-resident; a 7B attention projection"},
		{768, 200000, "STREAMED; FlatI8's CPU scan over a real corpus"},
		{1536, 8960, "STREAMED; a 1.5B model's FFN"},
		{3584, 18944, "STREAMED; a 7B model's FFN — where goinfer saw ~3% on decode"},
	} {
		K, N := s.K, s.N
		aq := make([]int8, K)
		for i := range aq {
			aq[i] = int8(rng.Intn(255) - 127)
		}
		bq := make([]int8, K*N)
		for i := range bq {
			bq[i] = int8(rng.Intn(255) - 127)
		}
		as := []float32{0.01}
		bs := make([]float32, N)
		for i := range bs {
			bs[i] = 0.01
		}
		dst := make([]float32, N)
		var ws Workspace
		// PIN THE KERNEL SERIAL (audit G-01). Three of these shapes sit within a
		// few percent of parThreshold on a 16-thread box (K3584_N4096 is 14.7M
		// MACs against 16.78M), so an unpinned run measures whether the fork/join
		// happened, not what the kernel did — which is why those rows have carried
		// ±11-35% floors and read BLIND in every recorded perfgate run. The fan-out
		// is worth measuring, but not in the benchmark whose job is the kernel.
		ws.SetThreshold(math.MaxInt)
		mb := float64(K) * float64(N) / (1 << 20)
		// The regime is in the sub-benchmark NAME, not just a log line: the A/B gate
		// (tools/perfgate) and any benchstat run parse the name, and the whole point of
		// this benchmark is that the two regimes must be told apart in the output, not
		// only in a comment. regimeTag keys off B's byte size vs a nominal LLC.
		b.Run(fmt.Sprintf("K%d_N%d_%s", K, N, regimeTag(K, N)), func(b *testing.B) {
			b.Logf("B is %.1f MB — %s", mb, s.note)
			for b.Loop() {
				MatmulBTW8A8Pre(&ws, aq, as, bq, bs, dst, 1, K, N)
			}
		})
	}
}
