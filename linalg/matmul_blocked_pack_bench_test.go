//go:build arm64

package linalg

// Re-verifies matmul_blocked.go's large-K packing dispatch (packKThreshold=2048,
// packStridePad=16) on THIS box: packed vs unpacked, and packed with vs without
// the power-of-two stride pad, at K = 1024/1536/2048/3072/4096. Packing is
// arm64-only (has2x8Kernel), so this only builds/runs there — nothing to sweep
// on amd64 for this specific question.
//
// unpackedFillDirect and packedFillNoPad are test-local copies of blockedFill's
// unpacked branch and packedFill respectively (the const packKThreshold can't be
// overridden at runtime, and the pad decision is inlined in packedFill), so both
// counterfactuals can be forced at any K regardless of the real dispatch.

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// unpackedFillDirect is blockedFill's unpacked branch, forced regardless of K —
// the counterfactual "what if packing never kicked in at this K".
func unpackedFillDirect(a, b, dst []float32, M, K, N, nStart, nEnd int) {
	const mBlock, nBlock, kBlock = mBlockDefault, nBlockDefault, kBlockDefault
	for i0 := 0; i0 < M; i0 += mBlock {
		iEnd := min(i0+mBlock, M)
		for n0 := nStart; n0 < nEnd; n0 += nBlock {
			nTileEnd := min(n0+nBlock, nEnd)
			for k0 := 0; k0 < K; k0 += kBlock {
				kEnd := min(k0+kBlock, K)
				kSpan := kEnd - k0
				k4 := kSpan / 4
				i := i0
				nEndAligned8 := n0 + ((nTileEnd-n0)/8)*8
				var s [64]float32
				for ; i+1 < iEnd; i += 2 {
					a0 := &a[i*K+k0]
					a1 := &a[(i+1)*K+k0]
					n := n0
					for ; n < nEndAligned8; n += 8 {
						Dot2x8(a0, a1,
							&b[n*K+k0], &b[(n+1)*K+k0], &b[(n+2)*K+k0], &b[(n+3)*K+k0],
							&b[(n+4)*K+k0], &b[(n+5)*K+k0], &b[(n+6)*K+k0], &b[(n+7)*K+k0],
							k4, &s)
						for r := range 2 {
							ii := i + r
							base := r * 32
							for j := range 8 {
								sum := s[base+j*4] + s[base+j*4+1] + s[base+j*4+2] + s[base+j*4+3]
								for k := k4 * 4; k < kSpan; k++ {
									sum += a[ii*K+k0+k] * b[(n+j)*K+k0+k]
								}
								dst[ii*N+n+j] += sum
							}
						}
					}
					accumRowRange(a, b, dst, i, K, N, k0, k4, kSpan, n, nTileEnd)
					accumRowRange(a, b, dst, i+1, K, N, k0, k4, kSpan, n, nTileEnd)
				}
				i = blockRows3x4(a, b, dst, i, iEnd, K, N, k0, k4, kSpan, n0, nTileEnd)
				for ; i < iEnd; i++ {
					accumRowRange(a, b, dst, i, K, N, k0, k4, kSpan, n0, nTileEnd)
				}
			}
		}
	}
}

// packedFillNoPad is packedFill with the item-24 stride pad forced off — isolates
// the pad's own contribution from packing's.
func packedFillNoPad(a, b, dst []float32, M, K, N, nStart, nEnd, kBlock int) {
	bufp := packBufPool.Get().(*[]float32)
	defer packBufPool.Put(bufp)
	bPack := *bufp

	n0 := nStart
	for ; n0+8 <= nEnd; n0 += 8 {
		for k0 := 0; k0 < K; k0 += kBlock {
			kEnd := min(k0+kBlock, K)
			kSpan := kEnd - k0
			k4 := kSpan / 4
			stride := kSpan // pad forced off, unlike packedFill
			for bi := range 8 {
				copy(bPack[bi*stride:bi*stride+kSpan], b[(n0+bi)*K+k0:(n0+bi)*K+kEnd])
			}
			p0, p1, p2, p3 := &bPack[0], &bPack[stride], &bPack[2*stride], &bPack[3*stride]
			p4, p5, p6, p7 := &bPack[4*stride], &bPack[5*stride], &bPack[6*stride], &bPack[7*stride]
			i := 0
			var s [64]float32
			for ; i+1 < M; i += 2 {
				Dot2x8(&a[i*K+k0], &a[(i+1)*K+k0], p0, p1, p2, p3, p4, p5, p6, p7, k4, &s)
				for r := range 2 {
					ii := i + r
					base := r * 32
					for j := range 8 {
						sum := s[base+j*4] + s[base+j*4+1] + s[base+j*4+2] + s[base+j*4+3]
						for k := k4 * 4; k < kSpan; k++ {
							sum += a[ii*K+k0+k] * b[(n0+j)*K+k0+k]
						}
						dst[ii*N+n0+j] += sum
					}
				}
			}
			for ; i < M; i++ {
				var s8 [32]float32
				Dot8x4(&a[i*K+k0], p0, p1, p2, p3, p4, p5, p6, p7, k4, &s8)
				for j := range 8 {
					sum := s8[j*4] + s8[j*4+1] + s8[j*4+2] + s8[j*4+3]
					for k := k4 * 4; k < kSpan; k++ {
						sum += a[i*K+k0+k] * b[(n0+j)*K+k0+k]
					}
					dst[i*N+n0+j] += sum
				}
			}
		}
	}
	if n0 < nEnd {
		unpackedFillDirect(a, b, dst, M, K, N, n0, nEnd)
	}
}

func randMatmulOperands(rng *rand.Rand, M, K, N int) (a, b []float32) {
	a = make([]float32, M*K)
	b = make([]float32, N*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	for i := range b {
		b[i] = float32(rng.NormFloat64())
	}
	return a, b
}

// TestMatmulBlockedPack_allArmsMatchNaive is the correctness gate before any
// timing claim: unpacked, packed+pad, and packed+noPad must all agree with the
// naive reference within the same tolerance item24_pad_test.go already uses (f32
// accumulation order differs slightly across the three tiling strategies, so this
// is a closeness bound, not bit-identity — matching this repo's own convention for
// this specific comparison).
func TestMatmulBlockedPack_allArmsMatchNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(2048))
	const M, N = 64, 1024
	for _, K := range []int{1024, 1536, 2048, 3072, 4096} {
		a, b := randMatmulOperands(rng, M, K, N)
		want := naiveMatmulBT(a, b, M, K, N)

		unpacked := make([]float32, M*N)
		unpackedFillDirect(a, b, unpacked, M, K, N, 0, N)

		packed := make([]float32, M*N)
		packedFill(a, b, packed, M, K, N, 0, N, packKBlockFor(K))

		noPad := make([]float32, M*N)
		packedFillNoPad(a, b, noPad, M, K, N, 0, N, packKBlockFor(K))

		for name, got := range map[string][]float32{"unpacked": unpacked, "packed": packed, "packed-noPad": noPad} {
			for i := range want {
				d := math.Abs(float64(got[i] - want[i]))
				if tol := 1e-3 + 1e-3*math.Abs(float64(want[i])); d > tol {
					t.Fatalf("K=%d arm=%s out[%d]=%g want~%g Δ%g > %g", K, name, i, got[i], want[i], d, tol)
				}
			}
		}
	}
}

// TestMatmulBlockedPack_bitIdenticalWhenKBlockMatches proves the corrected claim
// in blockedFill's comment: packing is bit-for-bit identical to the unpacked
// path ONLY when packKBlockFor(K) == kBlockDefault (K%768==0) — same k-tiling,
// just a different memory layout feeding it. K=1536 and K=3072 both satisfy
// that today.
func TestMatmulBlockedPack_bitIdenticalWhenKBlockMatches(t *testing.T) {
	if !has2x8Kernel {
		t.Skip("packing is arm64-only")
	}
	rng := rand.New(rand.NewSource(768))
	const M, N = 64, 1024
	for _, K := range []int{1536, 3072} {
		if packKBlockFor(K) != kBlockDefault {
			t.Fatalf("test premise broken: packKBlockFor(%d)=%d, want %d", K, packKBlockFor(K), kBlockDefault)
		}
		a, b := randMatmulOperands(rng, M, K, N)
		unpacked := make([]float32, M*N)
		unpackedFillDirect(a, b, unpacked, M, K, N, 0, N)
		packed := make([]float32, M*N)
		packedFill(a, b, packed, M, K, N, 0, N, packKBlockFor(K))
		for i := range unpacked {
			if math.Float32bits(unpacked[i]) != math.Float32bits(packed[i]) {
				t.Fatalf("K=%d idx=%d: packed %v (bits %08x) != unpacked %v (bits %08x) — expected exact match when k-tiling matches",
					K, i, packed[i], math.Float32bits(packed[i]), unpacked[i], math.Float32bits(unpacked[i]))
			}
		}
	}
}

// TestMatmulBlockedK1024_routingWithinTolerance gates the production dispatch
// change in blockedFill (K==1024 now routes to packedFill, below
// packKThreshold=2048). K=1024's pack tile (1024) differs from the unpacked
// path's k-tiling (768), so this is NOT bit-identical (see the ad hoc check
// this test's history is built on) — the real invariant is the tolerance the
// encoder's own parity gate holds bge-large/bge-m3/mxbai-large to
// (encoder/coverage_bert_test.go's assertBERTParity: maxΔ<=5e-3, cosine>=0.9999),
// checked here at a much tighter bound since these are single ops, not a whole
// forward pass where error could accumulate across 24 layers.
func TestMatmulBlockedK1024_routingWithinTolerance(t *testing.T) {
	if !has2x8Kernel {
		t.Skip("packing is arm64-only; K=1024 never routes to packedFill elsewhere")
	}
	rng := rand.New(rand.NewSource(1024))
	const K = 1024
	const tolAbs = 1e-4 // far tighter than the encoder's own 5e-3 whole-forward bar
	shapes := []struct{ M, N int }{
		{8, 1024}, {16, 1024}, {64, 1024}, {256, 1024},
		{64, 768}, {64, 3072}, {1, 1024}, {3, 1024}, // odd M too: the Dot8x4 single-row tail
	}
	for _, sh := range shapes {
		a, b := randMatmulOperands(rng, sh.M, K, sh.N)
		unpacked := make([]float32, sh.M*sh.N)
		unpackedFillDirect(a, b, unpacked, sh.M, K, sh.N, 0, sh.N)
		packed := make([]float32, sh.M*sh.N)
		packedFill(a, b, packed, sh.M, K, sh.N, 0, sh.N, packKBlockFor(K))
		var worst float64
		for i := range unpacked {
			d := math.Abs(float64(packed[i] - unpacked[i]))
			if d > worst {
				worst = d
			}
			if tol := tolAbs + tolAbs*math.Abs(float64(unpacked[i])); d > tol {
				t.Fatalf("M=%d N=%d idx=%d: packed %v vs unpacked %v, Δ%.3e > %.3e",
					sh.M, sh.N, i, packed[i], unpacked[i], d, tol)
			}
		}
		t.Logf("M=%d N=%d: worst |packed-unpacked| = %.3e", sh.M, sh.N, worst)
	}
}

func BenchmarkMatmulBlockedPack_unpacked_K1024(b *testing.B) { benchPackK(b, 1024, "unpacked") }
func BenchmarkMatmulBlockedPack_packed_K1024(b *testing.B)   { benchPackK(b, 1024, "packed") }
func BenchmarkMatmulBlockedPack_noPad_K1024(b *testing.B)    { benchPackK(b, 1024, "noPad") }

func BenchmarkMatmulBlockedPack_unpacked_K1536(b *testing.B) { benchPackK(b, 1536, "unpacked") }
func BenchmarkMatmulBlockedPack_packed_K1536(b *testing.B)   { benchPackK(b, 1536, "packed") }
func BenchmarkMatmulBlockedPack_noPad_K1536(b *testing.B)    { benchPackK(b, 1536, "noPad") }

func BenchmarkMatmulBlockedPack_unpacked_K2048(b *testing.B) { benchPackK(b, 2048, "unpacked") }
func BenchmarkMatmulBlockedPack_packed_K2048(b *testing.B)   { benchPackK(b, 2048, "packed") }
func BenchmarkMatmulBlockedPack_noPad_K2048(b *testing.B)    { benchPackK(b, 2048, "noPad") }

func BenchmarkMatmulBlockedPack_unpacked_K3072(b *testing.B) { benchPackK(b, 3072, "unpacked") }
func BenchmarkMatmulBlockedPack_packed_K3072(b *testing.B)   { benchPackK(b, 3072, "packed") }
func BenchmarkMatmulBlockedPack_noPad_K3072(b *testing.B)    { benchPackK(b, 3072, "noPad") }

func BenchmarkMatmulBlockedPack_unpacked_K4096(b *testing.B) { benchPackK(b, 4096, "unpacked") }
func BenchmarkMatmulBlockedPack_packed_K4096(b *testing.B)   { benchPackK(b, 4096, "packed") }
func BenchmarkMatmulBlockedPack_noPad_K4096(b *testing.B)    { benchPackK(b, 4096, "noPad") }

// BenchmarkPackGap fills in the gap below the current packKThreshold=2048: K=1024
// showed a surprisingly large packed win despite sitting below the cutoff (32%,
// bigger than K=1536's 5.3% even though 1536 IS above the cutoff already) — this
// sweeps the region between them at a finer grain to find where the real
// crossover sits, at the same M=64,N=1024 shape as the original 5-point sweep.
func BenchmarkPackGap(b *testing.B) {
	for _, K := range []int{768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048} {
		for _, arm := range []string{"unpacked", "packed"} {
			b.Run(fmt.Sprintf("K%d/%s", K, arm), func(b *testing.B) {
				benchPackK(b, K, arm)
			})
		}
	}
}

// BenchmarkPackShapeRobustness checks whether K=1024's packed win (32% at
// M=64,N=1024) is a property of K alone or an artifact of that specific M/N —
// varies M (a short prefill vs a long one) and N (the real hidden=768 and
// ffn=3072 transformer widths) at the one K that looked most actionable.
func BenchmarkPackShapeRobustness(b *testing.B) {
	const K = 1024
	shapes := []struct{ M, N int }{
		{8, 1024}, {16, 1024}, {64, 1024}, {256, 1024},
		{64, 768}, {64, 3072},
	}
	for _, sh := range shapes {
		for _, arm := range []string{"unpacked", "packed"} {
			b.Run(fmt.Sprintf("M%d_N%d/%s", sh.M, sh.N, arm), func(b *testing.B) {
				benchPackKMN(b, K, sh.M, sh.N, arm)
			})
		}
	}
}

func benchPackK(b *testing.B, K int, arm string) {
	benchPackKMN(b, K, 64, 1024, arm)
}

func benchPackKMN(b *testing.B, K, M, N int, arm string) {
	rng := rand.New(rand.NewSource(int64(K)))
	a, bMat := randMatmulOperands(rng, M, K, N)
	dst := make([]float32, M*N)
	kBlock := packKBlockFor(K)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		zeroSpanF32(dst)
		switch arm {
		case "unpacked":
			unpackedFillDirect(a, bMat, dst, M, K, N, 0, N)
		case "packed":
			packedFill(a, bMat, dst, M, K, N, 0, N, kBlock)
		case "noPad":
			packedFillNoPad(a, bMat, dst, M, K, N, 0, N, kBlock)
		}
	}
}
