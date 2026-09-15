//go:build arm64

package linalg

// packedFillLeftoverSplit — MEASURED NEGATIVE, not wired in.
//
// Explores an idea: instead of running packedFill over the WHOLE of K
// (including whatever small remainder chunk is left after the last full
// kBlock-sized tile), pack only the aligned prefix and run the leftover through
// the plain unpacked accumulation, added into the same dst — the same pattern
// packedFill already uses for its own <8-column remainder (falls back to
// blockedFill). Motivated by the gap-fill sweep in
// matmul_blocked_pack_bench_test.go: K=1152/1280's remainders (128, 256) ARE
// powers of two and already get packStridePad, yet still showed weak wins
// (6.5%, not statistically significant) — so the hypothesis was that the
// problem isn't "missing the pad," it's that a small remainder chunk's fixed
// per-chunk overhead (buffer copy setup, a second Dot2x8/Dot8x4 dispatch
// round) doesn't pay for itself regardless of whether that chunk is a power
// of two, and giving it a cheaper code path (no pack/copy at all) would help.
//
// Correctness held (TestMatmulBlockedLeftoverSplit_matchesNaive) but the
// performance hypothesis did NOT: benchstat n=5 against plain full-packing at
// K=1152/1280/1408/1536/1664/1792/1920 showed a wash at 4 of 7 K values, a
// real but tiny win at 2 (K=1408 -0.9%, K=1664 -2.1%), and a real regression
// at one (K=1920 +2.4%). So splitting the remainder onto its own path isn't
// the fix — the K=1152..1920 range's modest ceiling (2-7% over plain unpacked,
// far below K=1024/2048's 30-40%) is a real property of that range, not an
// artifact of how the last chunk is handled. Kept here, not wired in, per
// this codebase's own convention: a documented negative is worth something,
// a silently-not-tried one is not.

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// packedFillRange is packedFill restricted to processing k0 in [kStart, kEnd)
// only — K remains the real row stride for a/b, kStart/kEnd just bound which
// part of it this call contributes. Accumulates into dst (+=), same contract
// as every k-tile already does internally.
func packedFillRange(a, b, dst []float32, M, K, N, nStart, nEnd, kBlock, kStart, kEnd int) {
	bufp := packBufPool.Get().(*[]float32)
	defer packBufPool.Put(bufp)
	bPack := *bufp

	n0 := nStart
	for ; n0+8 <= nEnd; n0 += 8 {
		for k0 := kStart; k0 < kEnd; k0 += kBlock {
			kTileEnd := min(k0+kBlock, kEnd)
			kSpan := kTileEnd - k0
			k4 := kSpan / 4
			stride := kSpan
			if kSpan&(kSpan-1) == 0 {
				stride += packStridePad
			}
			for bi := range 8 {
				copy(bPack[bi*stride:bi*stride+kSpan], b[(n0+bi)*K+k0:(n0+bi)*K+kTileEnd])
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
		unpackedFillDirectRange(a, b, dst, M, K, N, n0, nEnd, kStart, kEnd)
	}
}

// unpackedFillDirectRange is unpackedFillDirect restricted to k in [kStart, kEnd).
func unpackedFillDirectRange(a, b, dst []float32, M, K, N, nStart, nEnd, kStart, kEnd int) {
	const mBlock, nBlock, kBlock = mBlockDefault, nBlockDefault, kBlockDefault
	for i0 := 0; i0 < M; i0 += mBlock {
		iEnd := min(i0+mBlock, M)
		for n0 := nStart; n0 < nEnd; n0 += nBlock {
			nTileEnd := min(n0+nBlock, nEnd)
			for k0 := kStart; k0 < kEnd; k0 += kBlock {
				kTileEnd := min(k0+kBlock, kEnd)
				kSpan := kTileEnd - k0
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

// packedFillLeftoverSplit packs only the kBlock-aligned prefix of K and runs
// the small remainder through the plain unpacked path, accumulated on top —
// the user's idea, mirroring packedFill's own <8-column fallback but on the K
// axis instead of N.
func packedFillLeftoverSplit(a, b, dst []float32, M, K, N, nStart, nEnd, kBlock int) {
	alignedK := (K / kBlock) * kBlock
	if alignedK > 0 {
		packedFillRange(a, b, dst, M, K, N, nStart, nEnd, kBlock, 0, alignedK)
	}
	if alignedK < K {
		unpackedFillDirectRange(a, b, dst, M, K, N, nStart, nEnd, alignedK, K)
	}
}

func TestMatmulBlockedLeftoverSplit_matchesNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(1152))
	const M, N = 64, 1024
	for _, K := range []int{1152, 1280, 1408, 1664, 1792, 1920} {
		a, b := randMatmulOperands(rng, M, K, N)
		want := naiveMatmulBT(a, b, M, K, N)
		got := make([]float32, M*N)
		packedFillLeftoverSplit(a, b, got, M, K, N, 0, N, packKBlockFor(K))
		for i := range want {
			d := math.Abs(float64(got[i] - want[i]))
			if tol := 1e-3 + 1e-3*math.Abs(float64(want[i])); d > tol {
				t.Fatalf("K=%d idx=%d: leftover-split %g want~%g Δ%g > %g", K, i, got[i], want[i], d, tol)
			}
		}
	}
}

func benchLeftoverK(b *testing.B, K int, arm string) {
	rng := rand.New(rand.NewSource(int64(K)))
	const M, N = 64, 1024
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
		case "leftoverSplit":
			packedFillLeftoverSplit(a, bMat, dst, M, K, N, 0, N, kBlock)
		}
	}
}

func BenchmarkLeftoverSplit(b *testing.B) {
	for _, K := range []int{1152, 1280, 1408, 1536, 1664, 1792, 1920} {
		for _, arm := range []string{"unpacked", "packed", "leftoverSplit"} {
			b.Run(fmt.Sprintf("K%d/%s", K, arm), func(b *testing.B) {
				benchLeftoverK(b, K, arm)
			})
		}
	}
}
