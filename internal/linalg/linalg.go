// Package linalg holds aikit's shared SIMD compute kernels: the dot-product
// kernels (AVX2+FMA on amd64, NEON on arm64, scalar elsewhere — selected by
// build tag and runtime CPU detection) and a row-parallel float32 matmul built
// on them. It is the single home for the hand-written assembly so the encoder
// and the Gemma decoder share one copy (gemma-decoder-plan.md §1).
package linalg

import (
	"runtime"
	"sync"
)

// Dot returns Σ a[i]*b[i]. len(a) must equal len(b).
func Dot(a, b []float32) float32 { return dotF32(a, b) }

// Dot4x4 / Dot8x4 are the register-blocked micro-kernels: each computes 4 / 8
// dot products of the shared row `a` against consecutive b-rows, writing every
// row's full sum into the first lane of its 4-lane block in sums (the rest
// zero). n4 is len/4 (K is a multiple of 4 in the matmul hot path). Exposed
// for the encoder's cache-blocked matmul; the decoder uses MatmulBT/Dot.
func Dot4x4(a, b0, b1, b2, b3 *float32, n4 int, sums *[16]float32) {
	dotNEON4x4(a, b0, b1, b2, b3, n4, sums)
}

func Dot8x4(a, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n4 int, sums *[32]float32) {
	dotNEON8x4(a, b0, b1, b2, b3, b4, b5, b6, b7, n4, sums)
}

// parThreshold is the MAC count below which MatmulBT runs serially — under it
// the goroutine fan-out costs more than it saves.
const parThreshold = 1 << 15

// MatmulBT computes dst[M,N] = a[M,K] · b[N,K]ᵀ — the PyTorch [out,in] weight
// layout the safetensors checkpoints store, so no transpose copy is needed.
// Each output is a Dot of an a-row against a b-row; work is parallelized over
// the N output columns (always large in the transformer projections, and the
// only dimension with parallelism on the M=1 single-token decode path).
func MatmulBT(a, b, dst []float32, M, K, N int) {
	parallelCols(M*N*K, N, func(j0, j1 int) {
		for i := 0; i < M; i++ {
			arow := a[i*K : i*K+K]
			drow := dst[i*N : i*N+N]
			for j := j0; j < j1; j++ {
				drow[j] = dotF32(arow, b[j*K:j*K+K])
			}
		}
	})
}

// parallelCols runs fn over the [0,N) output columns, split into one chunk per
// worker. Serial below parThreshold MACs (work) where the goroutine fan-out
// would cost more than it saves. Both the f32 and int8 matmuls share it.
func parallelCols(work, N int, fn func(j0, j1 int)) {
	if work < parThreshold || N < 2 {
		fn(0, N)
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > N {
		workers = N
	}
	chunk := (N + workers - 1) / workers
	var wg sync.WaitGroup
	for j0 := 0; j0 < N; j0 += chunk {
		j1 := j0 + chunk
		if j1 > N {
			j1 = N
		}
		wg.Add(1)
		go func(j0, j1 int) {
			defer wg.Done()
			fn(j0, j1)
		}(j0, j1)
	}
	wg.Wait()
}
