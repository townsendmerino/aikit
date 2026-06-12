package linalg

import (
	"math/rand/v2"
	"testing"
)

// Tile-sweep benchmarks for the blocked GEMM (hoisted from encoder with the kernel). The
// M10 sweep picked 32×32×768 (= the {m,n,k}BlockDefault constants); these let it be
// re-run if hardware changes, by calling blockedFill directly with custom tile sizes.

func randMat(rng *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rng.NormFloat64() * 0.1)
	}
	return out
}

// M10 tile sweep: parameterized matmul bench so we can search for the
// post-8x4-kernel optimum without recompiling between runs. The shape
// is L80_fc11 (M=80, K=768, N=3072) — the largest forward-pass GEMM
// and the one most likely to benefit from larger tiles.
func benchShapeTiled(b *testing.B, M, K, N, mB, nB, kB int) {
	rng := rand.New(rand.NewPCG(1, 2))
	a := randMat(rng, M*K)
	w := randMat(rng, N*K)
	dst := make([]float32, M*N)
	b.SetBytes(int64(2 * M * K * N))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range dst {
			dst[j] = 0
		}
		blockedFill(a, w, dst, M, K, N, 0, N, mB, nB, kB)
	}
}

// Named sweep cells. Naming: <mBlock>x<nBlock>x<kBlock>.
func BenchmarkMatmulTile_32x32x128(b *testing.B)  { benchShapeTiled(b, 80, 768, 3072, 32, 32, 128) }
func BenchmarkMatmulTile_32x64x128(b *testing.B)  { benchShapeTiled(b, 80, 768, 3072, 32, 64, 128) }
func BenchmarkMatmulTile_64x32x128(b *testing.B)  { benchShapeTiled(b, 80, 768, 3072, 64, 32, 128) }
func BenchmarkMatmulTile_64x64x128(b *testing.B)  { benchShapeTiled(b, 80, 768, 3072, 64, 64, 128) }
func BenchmarkMatmulTile_32x32x256(b *testing.B)  { benchShapeTiled(b, 80, 768, 3072, 32, 32, 256) }
func BenchmarkMatmulTile_64x64x64(b *testing.B)   { benchShapeTiled(b, 80, 768, 3072, 64, 64, 64) }
func BenchmarkMatmulTile_16x32x128(b *testing.B)  { benchShapeTiled(b, 80, 768, 3072, 16, 32, 128) }
func BenchmarkMatmulTile_32x128x128(b *testing.B) { benchShapeTiled(b, 80, 768, 3072, 32, 128, 128) }
func BenchmarkMatmulTile_64x128x128(b *testing.B) { benchShapeTiled(b, 80, 768, 3072, 64, 128, 128) }
func BenchmarkMatmulTile_32x256x128(b *testing.B) { benchShapeTiled(b, 80, 768, 3072, 32, 256, 128) }

// Larger kBlock variants — the 32x32x256 winner suggested kBlock is
// the lever, not mBlock/nBlock. Test K=768 (full K in one tile, no
// k-loop overhead) and intermediate sizes.
func BenchmarkMatmulTile_32x32x384(b *testing.B) { benchShapeTiled(b, 80, 768, 3072, 32, 32, 384) }
func BenchmarkMatmulTile_32x32x768(b *testing.B) { benchShapeTiled(b, 80, 768, 3072, 32, 32, 768) }
func BenchmarkMatmulTile_32x64x256(b *testing.B) { benchShapeTiled(b, 80, 768, 3072, 32, 64, 256) }
func BenchmarkMatmulTile_32x64x768(b *testing.B) { benchShapeTiled(b, 80, 768, 3072, 32, 64, 768) }
func BenchmarkMatmulTile_16x32x256(b *testing.B) { benchShapeTiled(b, 80, 768, 3072, 16, 32, 256) }
func BenchmarkMatmulTile_64x32x256(b *testing.B) { benchShapeTiled(b, 80, 768, 3072, 64, 32, 256) }

// Cross-shape check of the 32x32x768 winner against all 4 forward-pass
// shapes — the fc2 case has K=3072 so 768 splits it into 4 tiles
// (vs 24 with kBlock=128). If any shape regresses, the default needs
// to compromise. Also a 32x32x3072 cell for fc2's K-full case.
func BenchmarkMatmulTile_wqkv_32x32x768(b *testing.B) { benchShapeTiled(b, 80, 768, 2304, 32, 32, 768) }
func BenchmarkMatmulTile_fc2_32x32x768(b *testing.B)  { benchShapeTiled(b, 80, 3072, 768, 32, 32, 768) }
func BenchmarkMatmulTile_fc2_32x32x3072(b *testing.B) {
	benchShapeTiled(b, 80, 3072, 768, 32, 32, 3072)
}
func BenchmarkMatmulTile_outproj_32x32x768(b *testing.B) {
	benchShapeTiled(b, 80, 768, 768, 32, 32, 768)
}
func BenchmarkMatmulTile_L512_fc11_32x32x768(b *testing.B) {
	benchShapeTiled(b, 512, 768, 3072, 32, 32, 768)
}
