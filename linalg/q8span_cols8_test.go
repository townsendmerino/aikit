package linalg

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// q8SpanColumn is the definition of MatmulBTQ8's arithmetic; on arm64 q8Span
// runs eight columns through dotNEON8x4 instead. The contract is raw-bit
// equality on every output, over: K with a K%4 dot tail and a K%16 widen tail,
// N below eight, N not a multiple of eight (remainder columns), M=1 and M>1,
// serial and parallel dispatch, and column ranges that do not start at 0.
func TestQ8Span8Cols_bitIdenticalToColumnForm(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5107))
	shapes := []struct{ M, K, N int }{
		{1, 4, 8}, {1, 3, 8}, {1, 1, 9}, {1, 5, 7}, {1, 101, 3},
		{1, 16, 16}, {1, 100, 40}, {1, 101, 41}, {1, 103, 63}, {1, 1536, 8},
		{1, 1536, 100}, {3, 768, 24}, {8, 768, 512}, {5, 1537, 17}, {1, 2048, 4096},
		{2, 8960, 12},
	}
	for _, sh := range shapes {
		M, K, N := sh.M, sh.K, sh.N
		a := make([]float32, M*K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		bQ := make([]int8, N*K)
		for i := range bQ {
			bQ[i] = int8(rng.Intn(255) - 127)
		}
		bScales := make([]float32, N)
		for i := range bScales {
			bScales[i] = float32(rng.NormFloat64()) * 0.01
		}
		want := make([]float32, M*N)
		deq1 := make([]float32, K)
		for j := range N {
			q8SpanColumn(a, bQ, bScales, want, M, K, N, j, deq1)
		}

		// The span over the whole range, and over sub-ranges with odd starts.
		got := make([]float32, M*N)
		deq := make([]float32, q8SpanScratchRows*K)
		q8Span(a, bQ, bScales, got, M, K, N, 0, N, deq)
		compareBits(t, fmt.Sprintf("M=%d K=%d N=%d full", M, K, N), got, want)
		if N > 3 {
			for i := range got {
				got[i] = float32(math.NaN())
			}
			q8Span(a, bQ, bScales, got, M, K, N, 0, 3, deq)
			q8Span(a, bQ, bScales, got, M, K, N, 3, N, deq)
			compareBits(t, fmt.Sprintf("M=%d K=%d N=%d split@3", M, K, N), got, want)
		}

		// Through the public entry points: serial and forced-parallel.
		for _, thr := range []int{1 << 62, 0} {
			var ws Workspace
			ws.SetThreshold(thr)
			for i := range got {
				got[i] = float32(math.NaN())
			}
			MatmulBTQ8Into(&ws, a, bQ, bScales, got, M, K, N)
			compareBits(t, fmt.Sprintf("M=%d K=%d N=%d Into(thr=%d)", M, K, N, thr), got, want)
		}
	}
}

func compareBits(t *testing.T, label string, got, want []float32) {
	t.Helper()
	for idx := range want {
		if math.Float32bits(got[idx]) != math.Float32bits(want[idx]) {
			t.Fatalf("%s idx=%d: got %v (%08x) want %v (%08x)", label, idx,
				got[idx], math.Float32bits(got[idx]), want[idx], math.Float32bits(want[idx]))
		}
	}
}

// The parallel path must not allocate a scratch per worker per call once the pool
// is warm (the make it replaced was ~16 MB/token in int8 mode). One fork/join's
// goroutines and closure are allowed; the widened-row buffers are not.
func TestMatmulBTQ8Into_parallelScratchPooled(t *testing.T) {
	const M, K, N = 2, 1536, 4096
	rng := rand.New(rand.NewSource(3))
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	bQ := make([]int8, N*K)
	for i := range bQ {
		bQ[i] = int8(rng.Intn(255) - 127)
	}
	bScales := make([]float32, N)
	for i := range bScales {
		bScales[i] = 0.01
	}
	dst := make([]float32, M*N)
	var ws Workspace
	ws.SetThreshold(0)
	for range 4 {
		MatmulBTQ8Into(&ws, a, bQ, bScales, dst, M, K, N) // warm the pool
	}
	workers := resolveWidth(0)
	allocs := testing.AllocsPerRun(20, func() {
		MatmulBTQ8Into(&ws, a, bQ, bScales, dst, M, K, N)
	})
	// Bookkeeping per call: the closure, the WaitGroup-captured goroutine
	// frames, one per worker — generously bounded. A per-worker K-float make
	// would add `workers` allocations of 6–49 KB on top, which is what this
	// catches; the pool makes them zero after warm-up.
	if allocs > float64(workers+3) {
		t.Fatalf("MatmulBTQ8Into parallel path: %.1f allocs/call, want ≤ %d (per-worker scratch is leaking past the pool)", allocs, workers+3)
	}
}

func BenchmarkMatmulBTQ8_span(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	for _, sh := range []struct{ M, K, N int }{{1, 1536, 1536}, {1, 1536, 8960}, {4, 1536, 8960}, {1, 2048, 152064}} {
		M, K, N := sh.M, sh.K, sh.N
		a := make([]float32, M*K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		bQ := make([]int8, N*K)
		for i := range bQ {
			bQ[i] = int8(rng.Intn(255) - 127)
		}
		bScales := make([]float32, N)
		for i := range bScales {
			bScales[i] = 0.01
		}
		dst := make([]float32, M*N)
		var ws Workspace
		ws.SetThreshold(1 << 62) // serial: the kernel, not the fan-out
		b.Run(fmt.Sprintf("cols8/M%d_K%d_N%d", M, K, N), func(b *testing.B) {
			b.SetBytes(int64(N) * int64(K))
			for b.Loop() {
				MatmulBTQ8Into(&ws, a, bQ, bScales, dst, M, K, N)
			}
		})
		deq := make([]float32, K)
		b.Run(fmt.Sprintf("column/M%d_K%d_N%d", M, K, N), func(b *testing.B) {
			b.SetBytes(int64(N) * int64(K))
			for b.Loop() {
				for j := range N {
					q8SpanColumn(a, bQ, bScales, dst, M, K, N, j, deq)
				}
			}
		})
	}
}
