package encoder

import (
	"math/rand/v2"
	"testing"
)

// TestParallelMatmul_exactlyMatchesSerial: row-splitting must be
// numerically EXACT vs the serial blocked fill — not just within
// tolerance. Each output row is computed by an identical tiling over
// the same a-row and shared b, so the f32 reduction order is
// unchanged; any diff means a row-slicing/offset bug.
func TestParallelMatmul_exactlyMatchesSerial(t *testing.T) {
	cases := []struct {
		name    string
		M, K, N int
	}{
		{"L256_fc11", 256, 768, 3072},
		{"L512_fc11", 512, 768, 3072},
		{"L512_fc2", 512, 3072, 768},
		{"odd_rows", 257, 768, 2304}, // M not divisible by worker count
		{"few_rows", 3, 768, 3072},   // fewer rows than workers
	}
	rng := rand.New(rand.NewPCG(1, 2))
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := randMat(rng, c.M*c.K)
			b := randMat(rng, c.N*c.K)

			serial := make([]float32, c.M*c.N)
			matmulBTBlockedInto(a, b, serial, c.M, c.K, c.N)

			par := make([]float32, c.M*c.N)
			matmulBTBlockedIntoParallel(a, b, par, c.M, c.K, c.N)

			for i := range serial {
				if serial[i] != par[i] {
					t.Fatalf("idx %d: serial=%v parallel=%v (must be bit-identical)",
						i, serial[i], par[i])
				}
			}
		})
	}
}

// TestWantParallelMatmul_gating verifies the in-flight gate: parallel
// only when ≤1 forward is in flight and the shape is big + tall enough.
func TestWantParallelMatmul_gating(t *testing.T) {
	// Baseline: no forward in flight, large tall shape → yes.
	if !wantParallelMatmul(512, 768, 3072) {
		t.Error("large lone-forward shape should parallelize")
	}
	// Below the FLOP threshold (8.4M < 32M) but plenty of rows → the
	// FLOP gate (not the row gate) must keep it serial.
	if wantParallelMatmul(128, 256, 256) {
		t.Error("sub-threshold shape should stay serial")
	}
	// Big FLOPs but too few rows to split → row gate keeps it serial.
	if wantParallelMatmul(16, 4096, 4096) {
		t.Error("too-few-rows shape should stay serial")
	}
	// Tall enough rows but a second forward in flight → no.
	enterForward()
	enterForward() // count = 2
	if wantParallelMatmul(512, 768, 3072) {
		t.Error("with 2 forwards in flight, matmul must stay serial")
	}
	leaveForward()
	leaveForward() // count back to 0
	// Restored.
	if !wantParallelMatmul(512, 768, 3072) {
		t.Error("after forwards drained, large shape should parallelize again")
	}
}

// Benchmark the win: serial blocked vs row-parallel at the single-doc
// shapes that clear the threshold. Run with -cpu to see scaling, e.g.
//
//	go test ./encoder -run XXX -bench 'MatmulParallel|MatmulBT_blocked_L512'
func benchShapeInto(b *testing.B, M, K, N int, parallel bool) {
	rng := rand.New(rand.NewPCG(1, 2))
	a := randMat(rng, M*K)
	w := randMat(rng, N*K)
	dst := make([]float32, M*N)
	b.SetBytes(int64(2 * M * K * N))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if parallel {
			matmulBTBlockedIntoParallel(a, w, dst, M, K, N)
		} else {
			matmulBTBlockedInto(a, w, dst, M, K, N)
		}
	}
}

func BenchmarkMatmulParallel_L256_fc11(b *testing.B) { benchShapeInto(b, 256, 768, 3072, true) }
func BenchmarkMatmulSerial_L256_fc11(b *testing.B)   { benchShapeInto(b, 256, 768, 3072, false) }
func BenchmarkMatmulParallel_L512_fc11(b *testing.B) { benchShapeInto(b, 512, 768, 3072, true) }
func BenchmarkMatmulSerial_L512_fc11(b *testing.B)   { benchShapeInto(b, 512, 768, 3072, false) }
func BenchmarkMatmulParallel_L512_fc2(b *testing.B)  { benchShapeInto(b, 512, 3072, 768, true) }
func BenchmarkMatmulSerial_L512_fc2(b *testing.B)    { benchShapeInto(b, 512, 3072, 768, false) }

// Threshold-tuning probes: smaller shapes near/below parallelThreshold.
// L80 fc11 = 188M FLOP, L80 outproj = 47M, attention-ish = 17M.
func BenchmarkMatmulParallel_L80_fc11(b *testing.B)    { benchShapeInto(b, 80, 768, 3072, true) }
func BenchmarkMatmulSerial_L80_fc11(b *testing.B)      { benchShapeInto(b, 80, 768, 3072, false) }
func BenchmarkMatmulParallel_L80_outproj(b *testing.B) { benchShapeInto(b, 80, 768, 768, true) }
func BenchmarkMatmulSerial_L80_outproj(b *testing.B)   { benchShapeInto(b, 80, 768, 768, false) }
func BenchmarkMatmulParallel_attn512(b *testing.B)     { benchShapeInto(b, 512, 64, 512, true) }
func BenchmarkMatmulSerial_attn512(b *testing.B)       { benchShapeInto(b, 512, 64, 512, false) }
