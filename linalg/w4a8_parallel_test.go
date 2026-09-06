//go:build arm64

package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// TestW4A8_parallelScaling checks how much the REAL parallel MatmulBTW4A8Into
// (fork-join across ws.parallel) actually gains over its own serial rate, at
// the real 1.5B model's actual gate/up-proj shape ([8960,1536] -- embedding
// 1536, FFN intermediate 8960, from the real GGUF metadata, not the borrowed
// 27B-model shape the ops-per-byte harness used). Gate 0 flagged: a
// single-call microbenchmark's cold rate is within ~13% of the real decode's
// matmul-only implied rate -- suggesting the parallel fan-out isn't buying
// much. Cycles through NLAYERS distinct weight matrices (matching real
// decode's per-layer-fresh-weights access pattern) so this isn't just
// measuring cache-resident reuse of one matrix.
func TestW4A8_parallelScaling(t *testing.T) {
	harnessOnly(t)
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const (
		K       = 1536
		group   = 32
		N       = 8960
		M       = 1
		NLAYERS = 8 // 8 * ~4.4MB packed = ~35MB, past this chip's shared cache
	)
	rng := rand.New(rand.NewSource(1))
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}

	type layer struct {
		packed []byte
		scales []float32
	}
	layers := make([]layer, NLAYERS)
	for l := range layers {
		w := make([]float32, N*K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		p, s := QuantizeGroupsInt4(w, N, K, group)
		layers[l] = layer{p, s}
	}
	dst := make([]float32, M*N)

	run := func(workers int) float64 {
		var ws Workspace
		ws.SetWorkers(workers)
		ws.SetThreshold(0) // force-parallel regardless of the MAC-count gate
		best := math.Inf(1)
		for range 3 {
			i := 0
			r := testing.Benchmark(func(b *testing.B) {
				for b.Loop() {
					l := layers[i]
					MatmulBTW4A8Into(&ws, a, l.packed, l.scales, dst, M, K, N, group)
					i++
					if i == NLAYERS {
						i = 0
					}
				}
			})
			best = min(best, float64(r.NsPerOp()))
		}
		return best
	}

	bytesPerRow := float64((K+1)/2 + (K/group)*4)
	totalBytes := float64(N) * bytesPerRow

	var base float64
	for _, workers := range []int{1, 2, 4, 6, 8} {
		ns := run(workers)
		gbs := totalBytes / ns
		if workers == 1 {
			base = ns
		}
		t.Logf("workers=%d: %.0f ns/matmul  %.2f GB/s  (%.2fx vs 1-worker)",
			workers, ns, gbs, base/ns)
	}
}
