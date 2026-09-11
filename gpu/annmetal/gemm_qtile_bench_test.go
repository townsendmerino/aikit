//go:build darwin

package annmetal

import (
	"fmt"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/gpu"
)

// BenchmarkMetalGEMMKernels compares the two batched score GEMMs directly
// (audit M-16): the 16x16 byte-tiled form and the SIMD-group-per-row QTILE
// form that replaced it.
//
// This comparison is the point. The July dead end recorded "the GPU batch GEMM
// loses ~5x to the SIMD CPU" having measured the tiled kernel — the same shape
// CUDA's own sweep ranked WORST — so the conclusion was about that kernel
// rather than about Metal. Benchmarking the two against each other is what
// separates the two claims.
//
// Env-gated like the other GPU benches; a plain `go test` stays green.
func BenchmarkMetalGEMMKernels(b *testing.B) {
	if os.Getenv("AIKIT_GPU_BENCH") == "" {
		b.Skip("periodic GPU pass — set AIKIT_GPU_BENCH=1 to run")
	}
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no Metal device: %v", err)
	}
	defer dev.ReleaseObjects()
	lib, err := dev.CompileLibrary(w8a8Src, gpu.MSL3_1)
	if err != nil {
		b.Fatalf("CompileLibrary: %v", err)
	}
	gemm, err := dev.NewComputePipeline(lib, "gemm_w8a8_tiled")
	if err != nil {
		b.Fatalf("tiled pipeline: %v", err)
	}
	gemmq, err := dev.NewComputePipeline(lib, "gemm_w8a8_qtile")
	if err != nil {
		b.Fatalf("qtile pipeline: %v", err)
	}
	wq := gemmq.ThreadExecutionWidth()
	if wq <= 0 {
		wq = 32
	}
	be := &metalBackend{dev: dev, q: dev.NewCommandQueue(), gemm: gemm, gemmq: gemmq, gemmqW: wq}
	const K = 256
	rng := rand.New(rand.NewSource(3))
	for _, n := range []int{100_000, 500_000} {
		codes := make([]int8, n*K)
		for i := range codes {
			codes[i] = int8(rng.Intn(255) - 127)
		}
		scales := make([]float32, n)
		for i := range scales {
			scales[i] = 0.01
		}
		dCodes := gpu.NewBufferOf(be.dev, codes)
		dScales := gpu.NewBufferOf(be.dev, scales)
		kBuf := gpu.NewBufferOf(be.dev, []uint32{uint32(K)})
		for _, M := range []int{8, 32, 64} {
			qi8 := make([]int8, M*K)
			for i := range qi8 {
				qi8[i] = int8(rng.Intn(255) - 127)
			}
			qs := make([]float32, M)
			for i := range qs {
				qs[i] = 0.01
			}
			dQ := gpu.NewBufferOf(be.dev, qi8)
			dQs := gpu.NewBufferOf(be.dev, qs)
			nBuf := gpu.NewBufferOf(be.dev, []uint32{uint32(n)})
			mBuf := gpu.NewBufferOf(be.dev, []uint32{uint32(M)})
			out := be.dev.NewBufferLen(M * n)
			bufs := []gpu.Buffer{dCodes, dQ, dScales, kBuf, nBuf, dQs, out, mBuf}

			macs := int64(M) * int64(n) * int64(K)
			b.Run(fmt.Sprintf("N%d/M%d/tiled", n, M), func(b *testing.B) {
				const tile = 16
				gx, gy := (n+tile-1)/tile, (M+tile-1)/tile
				for b.Loop() {
					be.runLocked2D(be.gemm, gx, gy, tile, tile, bufs...)
				}
				b.ReportMetric(float64(macs)/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "GMAC/s")
			})
			b.Run(fmt.Sprintf("N%d/M%d/qtile", n, M), func(b *testing.B) {
				for b.Loop() {
					be.runGEMM(M, n, bufs...)
				}
				b.ReportMetric(float64(macs)/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "GMAC/s")
			})
		}
	}
}
