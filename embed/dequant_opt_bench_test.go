package embed

// Paired before/after benchmarks for the dequant optimizations in
// dequant_opt_test.go. Each Benchmark<Kernel>_preOpt / _optimized pair runs the
// same synthetic super-blocks through the pre-change and post-change kernel, so
// `go test -bench . -count=N` output is directly comparable with benchstat.

import (
	"math/rand/v2"
	"testing"
)

func randRawBlocks(blockBytes, nBlocks int, seed uint64) []byte {
	rng := rand.New(rand.NewPCG(seed, seed^0xabcdef))
	raw := make([]byte, nBlocks*blockBytes)
	for i := range raw {
		raw[i] = byte(rng.UintN(256))
	}
	return raw
}

func runDequantBench(b *testing.B, blockBytes, elems, nBlocks int, fn func(raw []byte, sb int, out []float32)) {
	b.Helper()
	raw := randRawBlocks(blockBytes, nBlocks, 42)
	out := make([]float32, elems)
	b.SetBytes(int64(blockBytes * nBlocks))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for sb := range nBlocks {
			fn(raw, sb, out)
		}
	}
}

const benchBlocks = 64 // ~16K elements/iteration, matches typical tensor-row chunk sizes

func BenchmarkDequantIQ2S_preOpt(b *testing.B) {
	runDequantBench(b, 82, 256, benchBlocks, dequantIQ2SBlockPreOpt)
}
func BenchmarkDequantIQ2S_optimized(b *testing.B) {
	runDequantBench(b, 82, 256, benchBlocks, dequantIQ2SBlock)
}

func BenchmarkDequantIQ3S_preOpt(b *testing.B) {
	runDequantBench(b, 110, 256, benchBlocks, dequantIQ3SBlockPreOpt)
}
func BenchmarkDequantIQ3S_optimized(b *testing.B) {
	runDequantBench(b, 110, 256, benchBlocks, dequantIQ3SBlock)
}

func BenchmarkDequantQ4K_preOpt(b *testing.B) {
	runDequantBench(b, 144, 256, benchBlocks, dequantQ4KBlockPreOpt)
}
func BenchmarkDequantQ4K_optimized(b *testing.B) {
	runDequantBench(b, 144, 256, benchBlocks, dequantQ4KBlock)
}

func BenchmarkDequantQ5K_preOpt(b *testing.B) {
	runDequantBench(b, 176, 256, benchBlocks, dequantQ5KBlockPreOpt)
}
func BenchmarkDequantQ5K_optimized(b *testing.B) {
	runDequantBench(b, 176, 256, benchBlocks, dequantQ5KBlock)
}

func BenchmarkDequantQ2K_preOpt(b *testing.B) {
	runDequantBench(b, 84, 256, benchBlocks, dequantQ2KBlockPreOpt)
}
func BenchmarkDequantQ2K_optimized(b *testing.B) {
	runDequantBench(b, 84, 256, benchBlocks, dequantQ2KBlock)
}

func BenchmarkDequantQ3K_preOpt(b *testing.B) {
	runDequantBench(b, 110, 256, benchBlocks, dequantQ3KBlockPreOpt)
}
func BenchmarkDequantQ3K_optimized(b *testing.B) {
	runDequantBench(b, 110, 256, benchBlocks, dequantQ3KBlock)
}
