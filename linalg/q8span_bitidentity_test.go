package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// TestQ8Span_bitIdenticalToScalarWiden asserts, rather than argues, that swapping q8Span's
// scalar int8→f32 widen for the SIMD dequantRowInt8(_, _, 1.0) leaves the LM-head output
// byte-for-byte unchanged (P2 item 4). The reference computes the pre-change math explicitly —
// scalar widen, then the same dotF32, then the row scale — and every output float is compared
// by its raw bits against matmulBTQ8. Shapes cover M=1 serial, M=1 parallel (the large-vocab
// case), M>1 prefill, and a K that is NOT a multiple of 16 to exercise the SIMD widen's scalar
// tail.
func TestQ8Span_bitIdenticalToScalarWiden(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	shapes := []struct {
		M, K, N int
		note    string
	}{
		{1, 2048, 4096, "M=1 serial"},
		{1, 2048, 152064, "M=1 parallel — the large-vocab LM head"},
		{1, 100, 40, "K not a multiple of 16 — SIMD tail"},
		{8, 768, 512, "M>1 prefill"},
	}
	for _, s := range shapes {
		M, K, N := s.M, s.K, s.N
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

		// reference: the pre-P2 math — scalar widen, same dotF32, same post-scale, same order.
		ref := make([]float32, M*N)
		deq := make([]float32, K)
		for j := range N {
			bq := bQ[j*K : j*K+K]
			for k := range K {
				deq[k] = float32(bq[k])
			}
			sc := bScales[j]
			for i := range M {
				ref[i*N+j] = dotF32(a[i*K:i*K+K], deq) * sc
			}
		}

		got := make([]float32, M*N)
		matmulBTQ8(a, bQ, bScales, got, M, K, N)

		for idx := range got {
			if math.Float32bits(got[idx]) != math.Float32bits(ref[idx]) {
				t.Fatalf("%s (M=%d K=%d N=%d) idx=%d: SIMD widen %08x != scalar widen %08x",
					s.note, M, K, N, idx, math.Float32bits(got[idx]), math.Float32bits(ref[idx]))
			}
		}
		t.Logf("%s (M=%d K=%d N=%d): %d outputs byte-identical", s.note, M, K, N, M*N)
	}
}
