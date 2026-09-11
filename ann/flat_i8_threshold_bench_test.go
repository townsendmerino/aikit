package ann

import (
	"fmt"
	"testing"
)

// Audit M-19 (docs/audit-2026-09-10.md §1.D): FlatI8.Query's pooled Workspace
// is a zero value, so it inherited the process-wide parThreshold (1<<24 MACs)
// — goinfer's DECODE tuning — instead of Flat's measured flatParallelThreshold
// (1<<19). The scan's MAC count is 1*n*dim, so at dim 256 the int8 scan stayed
// single-core until n = 65,536 while the f32 scan shards from n = 2,048.
//
// This sweep straddles that boundary: N=4k..50k are inside the dead zone (they
// should go from serial to sharded), N=100k is already above the old threshold
// and is the control that must not move.
func BenchmarkFlatI8QueryThreshold(b *testing.B) {
	for _, d := range []int{256, 768} {
		for _, n := range []int{4_000, 16_000, 50_000, 100_000} {
			b.Run(fmt.Sprintf("d%d/N%d", d, n), func(b *testing.B) {
				corpus := makeUnitVectors(n, d, 0xfeed)
				query := makeUnitVectors(1, d, 0xdade)[0]
				f := NewFlatI8(corpus)
				b.SetBytes(int64(n) * int64(d)) // int8: 1 byte per dim
				b.ResetTimer()
				for range b.N {
					sinkHits = f.Query(query, 10)
				}
			})
		}
	}
}

// BenchmarkFlatI8QueryBatchCPU covers audit M-18: with no device attached,
// QueryBatch fell through to a per-query loop, so the corpus weight matrix was
// streamed once PER QUERY instead of once for the batch. The batched M-row
// W8A8 GEMM was already in the file, reachable only through the GPU shard
// split.
func BenchmarkFlatI8QueryBatchCPU(b *testing.B) {
	const d = 256
	for _, n := range []int{10_000, 100_000} {
		for _, batch := range []int{8, 64} {
			b.Run(fmt.Sprintf("N%d/batch%d", n, batch), func(b *testing.B) {
				corpus := makeUnitVectors(n, d, 0xfeed)
				queries := makeUnitVectors(batch, d, 0xdade)
				f := NewFlatI8(corpus)
				b.ResetTimer()
				for b.Loop() {
					sinkHitsBatch = f.QueryBatch(queries, 10)
				}
			})
		}
	}
}

var sinkHitsBatch [][]Hit
