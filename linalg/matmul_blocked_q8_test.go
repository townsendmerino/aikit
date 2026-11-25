package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// q8Shapes are the real CodeRankEmbed/Nomic linear shapes the fused kernel must match,
// across the M values that exercise every code path: M=1 (MoE single-token), M=2/3 (the
// Dot2x8 row-pair boundary and an odd tail row), and prefill lengths.
var q8Shapes = []struct {
	name string
	K, N int
}{
	{"Wqkv", 768, 2304},
	{"OutProj", 768, 768},
	{"fc11", 768, 3072},
	{"fc2", 3072, 768},
}

var q8Ms = []int{1, 2, 3, 10, 80, 91, 357}

// randQ8 builds random int8 weights, per-row f32 scales, and f32 activations.
func randQ8(rng *rand.Rand, M, K, N int) (a []float32, bQ []int8, bScales []float32) {
	a = make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	bQ = make([]int8, N*K)
	for i := range bQ {
		bQ[i] = int8(rng.Intn(256) - 128)
	}
	bScales = make([]float32, N)
	for i := range bScales {
		bScales[i] = float32(rng.Float64()*0.02 + 0.001) // realistic per-row weight scale
	}
	return a, bQ, bScales
}

// q8Reference is the path the fused kernel replaces: widen the WHOLE weight matrix to
// f32, then the vectorized f32 GEMM.
func q8Reference(a []float32, bQ []int8, bScales []float32, M, K, N int) []float32 {
	w := make([]float32, N*K)
	DequantizeRowsInt8Into(w, bQ, bScales, N, K)
	dst := make([]float32, M*N)
	MatmulBTInto(dst, a, w, M, K, N)
	return dst
}

// TestMatmulBTQ8Fused_bitIdentical is item 22 fix (b)'s acceptance gate: the fused
// widen-in-pack kernel must produce EXACTLY the same bits as dequant-then-f32-GEMM on
// the arch where it runs (arm64, HasFusedQ8Kernel). The equality is not by tolerance —
// the widen value float32(int8)*scale is identical, the k-tiling matches the reference,
// and the same Dot2x8/Dot8x4 kernels see the same b in the same order.
func TestMatmulBTQ8Fused_bitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, sh := range q8Shapes {
		for _, M := range q8Ms {
			a, bQ, bScales := randQ8(rng, M, sh.K, sh.N)
			ref := q8Reference(a, bQ, bScales, M, sh.K, sh.N)

			serial := make([]float32, M*sh.N)
			MatmulBTQ8FusedInto(serial, a, bQ, bScales, M, sh.K, sh.N)
			par := make([]float32, M*sh.N)
			MatmulBTQ8Fused(par, a, bQ, bScales, M, sh.K, sh.N)

			// Serial and column-parallel fused paths are the same kernel — exactly
			// equal on every arch (a partitions columns, which are independent).
			for i := range serial {
				if serial[i] != par[i] {
					t.Fatalf("%s M=%d: serial vs parallel differ at %d: %v != %v",
						sh.name, M, i, serial[i], par[i])
				}
			}

			if hasFusedQ8Kernel {
				// The live arch: bit-identical to the replaced path.
				for i := range ref {
					if serial[i] != ref[i] {
						t.Fatalf("%s M=%d: fused vs reference differ at %d (row %d col %d): %v != %v",
							sh.name, M, i, i/sh.N, i%sh.N, serial[i], ref[i])
					}
				}
			} else {
				// Off-arch the fused kernel is not used, but it must still be a correct
				// GEMM (different kernel ⇒ different rounding, so compare with tolerance).
				for i := range ref {
					d := math.Abs(float64(serial[i] - ref[i]))
					if d > 1e-3*(1+math.Abs(float64(ref[i]))) {
						t.Fatalf("%s M=%d: fused off-arch wrong at %d: %v vs ref %v (Δ%.2g)",
							sh.name, M, i, serial[i], ref[i], d)
					}
				}
			}
		}
	}
}

// TestMatmulBTQ8Fused_mutationDetected guards against a degenerate kernel that ignores
// its weights: flipping a single int8 weight must change the output. Without it the
// bit-identity test above would pass a kernel that returned, say, all zeros both ways.
func TestMatmulBTQ8Fused_mutationDetected(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const M, K, N = 10, 768, 2304
	a, bQ, bScales := randQ8(rng, M, K, N)

	base := make([]float32, M*N)
	MatmulBTQ8FusedInto(base, a, bQ, bScales, M, K, N)

	bQ[5*K+100] += 40 // perturb one weight (row 5, k=100)
	mut := make([]float32, M*N)
	MatmulBTQ8FusedInto(mut, a, bQ, bScales, M, K, N)

	diffs := 0
	for i := range base {
		if base[i] != mut[i] {
			diffs++
		}
	}
	// Column 5 is read by every one of the M rows, so exactly M outputs move.
	if diffs != M {
		t.Fatalf("mutating one weight changed %d outputs, want %d — kernel may not read all weights", diffs, M)
	}
}

// benchQ8Kernel isolates the fused widen-in-pack GEMM against dequant-then-f32-GEMM at
// one shape. It rotates a single L3-resident weight, so it UNDERSTATES the fusion's win
// (which comes from killing the DRAM round-trip of a cold 9.4 MB deqW across 60 matmuls);
// BenchmarkQ8Encode in the encoder is the end-to-end arbiter. Kept as a kernel-level
// regression guard: fused must not be slower than the reference at the same shape.
func benchQ8Kernel(b *testing.B, fused bool, M, K, N int) {
	rng := rand.New(rand.NewSource(3))
	a, bQ, bScales := randQ8(rng, M, K, N)
	dst := make([]float32, M*N)
	w := make([]float32, N*K)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if fused {
			MatmulBTQ8FusedInto(dst, a, bQ, bScales, M, K, N)
		} else {
			DequantizeRowsInt8Into(w, bQ, bScales, N, K)
			MatmulBTInto(dst, a, w, M, K, N)
		}
	}
}

func BenchmarkQ8Kernel_fc2_M80_fused(b *testing.B)  { benchQ8Kernel(b, true, 80, 3072, 768) }
func BenchmarkQ8Kernel_fc2_M80_ref(b *testing.B)    { benchQ8Kernel(b, false, 80, 3072, 768) }
func BenchmarkQ8Kernel_wqkv_M80_fused(b *testing.B) { benchQ8Kernel(b, true, 80, 768, 2304) }
func BenchmarkQ8Kernel_wqkv_M80_ref(b *testing.B)   { benchQ8Kernel(b, false, 80, 768, 2304) }
