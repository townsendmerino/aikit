//go:build !arm64

package linalg

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// q8SpanBlocked at every block width, including ones that do not divide the span
// and the jb=1 column form, against q8SpanColumn: raw-bit equality on every output.
func TestQ8SpanBlocked_bitIdenticalAtEveryWidth(t *testing.T) {
	rng := rand.New(rand.NewSource(0x9b10))
	for _, sh := range []struct{ M, K, N int }{{1, 37, 21}, {2, 64, 33}, {9, 101, 50}, {33, 1537, 19}} {
		M, K, N := sh.M, sh.K, sh.N
		a, bQ, bScales := q8SpanOperands(rng, M, K, N)
		want := make([]float32, M*N)
		for j := range N {
			q8SpanColumn(a, bQ, bScales, want, M, K, N, j, make([]float32, K))
		}
		got := make([]float32, M*N)
		for _, jb := range []int{1, 2, 3, 5, 8, 16, N, N + 7} {
			deq := make([]float32, jb*K)
			for i := range got {
				got[i] = float32(math.NaN())
			}
			q8SpanBlocked(a, bQ, bScales, got, M, K, N, 0, 5, jb, deq)
			q8SpanBlocked(a, bQ, bScales, got, M, K, N, 5, N, jb, deq)
			compareBits(t, fmt.Sprintf("M=%d K=%d N=%d jb=%d", M, K, N, jb), got, want)
		}
	}
}

func TestQ8SpanCols_boundsTheBlock(t *testing.T) {
	for _, c := range []struct{ K, want int }{{0, 16}, {1, 16}, {4096, 16}, {14336, 16}, {16384, 16}, {32768, 8}, {1 << 19, 1}} {
		if got := q8SpanCols(c.K); got != c.want {
			t.Errorf("q8SpanCols(%d) = %d, want %d", c.K, got, c.want)
		}
		if n := q8SpanScratchLen(1<<21, c.K); c.K > 0 && n > max(q8SpanBlockFloats, c.K) {
			t.Errorf("q8SpanScratchLen(2Mi, %d) = %d, over the %d-float bound", c.K, n, q8SpanBlockFloats)
		}
	}
	// The column form needs one widened row, so the pooled scratch stays K floats
	// wherever q8Span does not block.
	for _, c := range []struct{ M, K, want int }{{1, 4096, 4096}, {64, 14336, 14336}, {512, 4096, 4096}, {513, 4096, 16 * 4096}, {256, 14336, 16 * 14336}} {
		if got := q8SpanScratchLen(c.M, c.K); got != c.want {
			t.Errorf("q8SpanScratchLen(%d, %d) = %d, want %d", c.M, c.K, got, c.want)
		}
	}
}

// The serial path takes its block scratch from the Workspace, so a warm call at a
// prefill shape allocates nothing.
func TestMatmulBTQ8Into_serialBlockScratchReused(t *testing.T) {
	const M, K, N = 16, 4096, 40
	a, bQ, bScales := q8SpanOperands(rand.New(rand.NewSource(4)), M, K, N)
	dst := make([]float32, M*N)
	var ws Workspace
	ws.SetThreshold(1 << 62)
	MatmulBTQ8Into(&ws, a, bQ, bScales, dst, M, K, N)
	if allocs := testing.AllocsPerRun(20, func() {
		MatmulBTQ8Into(&ws, a, bQ, bScales, dst, M, K, N)
	}); allocs != 0 {
		t.Fatalf("MatmulBTQ8Into serial path: %.1f allocs/call, want 0", allocs)
	}
}

func q8SpanOperands(rng *rand.Rand, M, K, N int) (a []float32, bQ []int8, bScales []float32) {
	a = make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	bQ = make([]int8, N*K)
	for i := range bQ {
		bQ[i] = int8(rng.Intn(255) - 127)
	}
	bScales = make([]float32, N)
	for i := range bScales {
		bScales[i] = float32(rng.NormFloat64()) * 0.01
	}
	return a, bQ, bScales
}

// BenchmarkQ8SpanBlock times the parallel weight-only int8 matmul at the decoder's
// projection shapes: the column form (jb=1, the form before blocking), the blocked
// form at q8SpanCols(K), and q8Span's own choice between them, in one binary so the
// forms can be interleaved.
func BenchmarkQ8SpanBlock(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	for _, sh := range []struct{ K, N int }{{4096, 14336}, {14336, 4096}, {4096, 4096}} {
		K, N := sh.K, sh.N
		_, bQ, bScales := q8SpanOperands(rng, 1, K, N)
		for _, M := range []int{1, 64, 128, 256, 512, 2048} {
			a, _, _ := q8SpanOperands(rng, M, K, 1)
			dst := make([]float32, M*N)
			for _, form := range []struct {
				name string
				span func(j0, j1 int, deq []float32)
			}{
				{"column", func(j0, j1 int, deq []float32) { q8SpanBlocked(a, bQ, bScales, dst, M, K, N, j0, j1, 1, deq) }},
				{"blocked", func(j0, j1 int, deq []float32) {
					q8SpanBlocked(a, bQ, bScales, dst, M, K, N, j0, j1, q8SpanCols(K), deq)
				}},
				{"span", func(j0, j1 int, deq []float32) { q8Span(a, bQ, bScales, dst, M, K, N, j0, j1, deq) }},
			} {
				b.Run(fmt.Sprintf("%s/M%d_K%d_N%d", form.name, M, K, N), func(b *testing.B) {
					b.SetBytes(int64(M) * int64(N) * int64(K) * 2) // FLOPs, so MB/s reads as MFLOP/s
					var ws Workspace
					for b.Loop() {
						ws.parallel(N, func(j0, j1 int) {
							deq := q8SpanScratchGet(q8SpanCols(K) * K)
							form.span(j0, j1, *deq)
							q8SpanScratchPut(deq)
						})
					}
				})
			}
		}
	}
}
