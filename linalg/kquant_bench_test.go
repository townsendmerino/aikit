package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// The A1 go/no-go (docs/internal/archive/task-q8k-integer-accum.md §5.4): does the native Q6_K/Q4_K integer-accum GEMV
// beat the current decode path — dequant→int8-requant→W8A8, which at decode time is just
// MatmulBTW8A8 over the already-int8 resident weight — by ≥1.3× at M=1? The native path reads fewer
// weight bytes (Q6_K 210 B/256 = 0.82 B/w, Q4_K 144/256 = 0.56 B/w, vs int8's 1 B/w) but pays a
// per-superblock unpack. Run: go test -bench GEMV -benchmem ./linalg
//
// NOTE: the weight unpack is still scalar Go here; a "loses" result is therefore not yet the final
// no-go (SIMD unpack / the quad-interleaved layout are the remaining levers). A "wins" result is a
// clean go.

func benchDecodeShapes() []struct{ K, N int } {
	return []struct{ K, N int }{
		{2048, 2048},
		{4096, 4096},
	}
}

func BenchmarkGEMV_Q6K_native(b *testing.B) {
	rng := rand.New(rand.NewSource(10))
	for _, s := range benchDecodeShapes() {
		nb := s.K / qkK
		a := make([]float32, s.K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		w := randQ6KRow(rng, s.N*nb) // N rows × nb superblocks
		dst := make([]float32, s.N)
		b.Run(shapeName(s.K, s.N), func(b *testing.B) {
			b.SetBytes(int64(s.N * nb * q6kBlockB))
			for i := 0; i < b.N; i++ {
				MatmulBTGGUFQ6K(a, w, dst, 1, s.K, s.N)
			}
		})
	}
}

// gemvW8A8Shapes spans BOTH memory regimes on the M=1 GEMV path, the same discipline as
// BenchmarkW8A8SpanShapes: a cache-resident pair (where the v1.17.0 eight-column kernel was
// measured and looked like a win) AND a streamed pair (where it was the ~3% decode regression).
// A GEMV benchmark that samples only the resident regime is exactly what let v1.17.0 tag green.
func gemvW8A8Shapes() []struct{ K, N int } {
	return []struct{ K, N int }{
		{2048, 2048},  // resident
		{4096, 4096},  // resident
		{768, 200000}, // streamed — FlatI8's CPU scan over a real corpus
		{3584, 18944}, // streamed — a 7B model's FFN, where goinfer saw ~3% on decode
	}
}

func BenchmarkGEMV_W8A8_baseline(b *testing.B) {
	rng := rand.New(rand.NewSource(11))
	for _, s := range gemvW8A8Shapes() {
		a := make([]float32, s.K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		bq := make([]int8, s.N*s.K) // int8 resident weight
		for i := range bq {
			bq[i] = int8(rng.Intn(255) - 127)
		}
		bScales := make([]float32, s.N)
		for i := range bScales {
			bScales[i] = 0.01
		}
		dst := make([]float32, s.N)
		// Workspace form, pinned serial (audit G-01, G-08). This called the
		// ALLOCATING MatmulBTW8A8 wrapper, so every ns/op carried a make of the
		// quantized activation — and at K4096_N4096 the shape clears parThreshold,
		// so the baseline fanned out while the Q6K kernel it is compared against
		// runs serial, inflating that comparison by the worker count. Both are
		// artefacts of the harness rather than of the kernel.
		var ws Workspace
		ws.SetThreshold(math.MaxInt)
		b.Run(shapeName(s.K, s.N)+"_"+regimeTag(s.K, s.N), func(b *testing.B) {
			b.SetBytes(int64(s.N * s.K)) // int8: 1 byte/weight
			for i := 0; i < b.N; i++ {
				MatmulBTW8A8Into(&ws, a, bq, bScales, dst, 1, s.K, s.N)
			}
		})
	}
}

func shapeName(k, n int) string {
	return "K" + itoa(k) + "_N" + itoa(n)
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}
