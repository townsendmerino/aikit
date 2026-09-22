//go:build arm64

package linalg

import (
	"math/rand"
	"testing"
	"time"
)

// S-05 (docs/task-simd-audit.md): gates and harness for the centering fold.
//
// Gates (always on, DotProd):
//   - TestW4A8LaneCorrNeg8_matchesScalar — the NEON pre-pass against its Go definition.
//   - TestDotW4A8SplitHalf4RowFold_bitIdenticalToSplitHalf4Row — exact == against BOTH the
//     kernel it replaces and the canonical dotW4A8FoldSDOT, nGroups 1..20 + the production
//     48/280, random activations plus the int8 extremes.
//   - TestMatmulBTW4A8Row4Into_foldBitIdenticalToUnfolded — the dispatch, fold on vs off,
//     serial and 6-way parallel, at both production projections plus a small shape.
//
// Harness (AIKIT_HARNESS=1): TestW4A8RowFold_hotAB (L1-resident single call, the
// TestW4A8Row4ColdFix_warmIntact shape) and TestW4A8RowFold_streamedAB (DRAM-streamed
// 12-matrix bank through the real dispatch, serial and parallel, ABBA paired).

func foldTestQuad(rng *rand.Rand, K int) (packed4 []byte, scales4 []float32, rows [][]byte, scales [][]float32) {
	const group = 32
	nGroups := K / group
	rows = make([][]byte, 4)
	scales = make([][]float32, 4)
	for r := range 4 {
		w := make([]float32, K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		rows[r], scales[r] = QuantizeGroupsInt4(w, 1, K, group)
	}
	packed4 = repackSplitHalf4RowBlock(rows[0], rows[1], rows[2], rows[3], K)
	scales4 = interleaveScales4Row(scales[0], scales[1], scales[2], scales[3], nGroups)
	return
}

// foldTestActivations yields the activation rows the fold gates sweep: random, and
// the int8 extremes the identity must also hold at (−128 everywhere, 127 everywhere,
// and alternating), since |Σ| is largest there.
func foldTestActivations(rng *rand.Rand, K int, trials int) [][]int8 {
	var out [][]int8
	for range trials {
		act := make([]int8, K)
		for i := range act {
			act[i] = int8(rng.Intn(255) - 128)
		}
		out = append(out, act)
	}
	for _, fill := range []func(i int) int8{
		func(int) int8 { return -128 },
		func(int) int8 { return 127 },
		func(i int) int8 {
			if i%2 == 0 {
				return -128
			}
			return 127
		},
	} {
		act := make([]int8, K)
		for i := range act {
			act[i] = fill(i)
		}
		out = append(out, act)
	}
	return out
}

func TestW4A8LaneCorrNeg8_matchesScalar(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	rng := rand.New(rand.NewSource(505))
	for nGroups := 1; nGroups <= 37; nGroups++ {
		K := nGroups * 32
		for _, act := range foldTestActivations(rng, K, 3) {
			want := make([]int32, 4*nGroups)
			w4a8LaneCorrNeg8Scalar(act, want, nGroups)
			got := make([]int32, 4*nGroups)
			w4a8LaneCorrNeg8(&act[0], &got[0], nGroups)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("nGroups=%d corr[%d]: got %d want %d", nGroups, i, got[i], want[i])
				}
			}
		}
	}
}

func TestDotW4A8SplitHalf4RowFold_bitIdenticalToSplitHalf4Row(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	rng := rand.New(rand.NewSource(97))
	counts := make([]int, 0, 22)
	for n := 1; n <= 20; n++ {
		counts = append(counts, n)
	}
	counts = append(counts, 48, 280) // K = 1536, 8960: the production projections
	for _, nGroups := range counts {
		K := nGroups * 32
		for trial, act := range foldTestActivations(rng, K, 10) {
			packed4, scales4, rows, scales := foldTestQuad(rng, K)
			corr := make([]int32, 4*nGroups)
			w4a8LaneCorrNeg8(&act[0], &corr[0], nGroups)

			var got, ref [4]float32
			dotW4A8SplitHalf4RowFold(&act[0], &corr[0], &packed4[0], &scales4[0], &got[0], nGroups)
			dotW4A8SplitHalf4Row(&act[0], &packed4[0], &scales4[0], &ref[0], nGroups)
			for r := range 4 {
				if got[r] != ref[r] {
					t.Fatalf("nGroups=%d trial=%d row=%d: fold %v vs SplitHalf4Row %v (bit mismatch)", nGroups, trial, r, got[r], ref[r])
				}
				canon := dotW4A8FoldSDOT(&act[0], &rows[r][0], &scales[r][0], nGroups)
				if got[r] != canon {
					t.Fatalf("nGroups=%d trial=%d row=%d: fold %v vs canonical %v (bit mismatch)", nGroups, trial, r, got[r], canon)
				}
			}
		}
	}
}

func TestMatmulBTW4A8Row4Into_foldBitIdenticalToUnfolded(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	prev := w4a8RowFold
	defer SetW4A8RowFold(prev)
	rng := rand.New(rand.NewSource(1201))
	const group = 32
	shapes := []struct{ K, N int }{{1536, 8960}, {8960, 1536}, {64, 8}}
	for _, sh := range shapes {
		w := make([]float32, sh.N*sh.K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		packed, scales := QuantizeGroupsInt4(w, sh.N, sh.K, group)
		row4 := RepackW4A8Row4(packed, sh.N, sh.K, group)
		row4s := RepackW4A8Row4Scales(scales, sh.N, sh.K, group)
		for _, par := range []bool{false, true} {
			var ws Workspace
			if par {
				ws.SetThreshold(1)
				ws.SetWorkers(6)
			} else {
				ws.SetThreshold(1 << 60)
			}
			for trial := range 3 {
				a := make([]float32, sh.K)
				for i := range a {
					a[i] = float32(rng.NormFloat64())
				}
				off := make([]float32, sh.N)
				on := make([]float32, sh.N)
				SetW4A8RowFold(false)
				MatmulBTW4A8Row4Into(&ws, a, row4, row4s, off, 1, sh.K, sh.N, group)
				SetW4A8RowFold(true)
				MatmulBTW4A8Row4Into(&ws, a, row4, row4s, on, 1, sh.K, sh.N, group)
				for j := range off {
					if on[j] != off[j] {
						t.Fatalf("K=%d N=%d parallel=%v trial=%d: dst[%d] fold %v vs unfolded %v (bit mismatch)", sh.K, sh.N, par, trial, j, on[j], off[j])
					}
				}
			}
		}
	}
}

// TestW4A8RowFold_hotAB: the L1-resident single-call comparison the S-05 decision
// rule names (TestW4A8Row4ColdFix_warmIntact's shape). One quad, repeated calls, at
// both production group counts; min of 3 testing.Benchmark passes. Also prices the
// correction pre-pass on its own, amortised over the quads of a real projection,
// because the fold is only free if that is small.
func TestW4A8RowFold_hotAB(t *testing.T) {
	harnessOnly(t)
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	rng := rand.New(rand.NewSource(101))
	for _, sh := range []struct{ K, N int }{{1536, 8960}, {8960, 1536}} {
		nGroups := sh.K / 32
		act := make([]int8, sh.K)
		for i := range act {
			act[i] = int8(rng.Intn(255) - 128)
		}
		packed4, scales4, _, _ := foldTestQuad(rng, sh.K)
		corr := make([]int32, 4*nGroups)
		w4a8LaneCorrNeg8(&act[0], &corr[0], nGroups)
		var dst [4]float32

		base := minOf3(func() float64 {
			r := testing.Benchmark(func(b *testing.B) {
				for b.Loop() {
					dotW4A8SplitHalf4Row(&act[0], &packed4[0], &scales4[0], &dst[0], nGroups)
				}
			})
			sinkW4A8F32ARM64 = dst[0]
			return float64(r.NsPerOp())
		})
		fold := minOf3(func() float64 {
			r := testing.Benchmark(func(b *testing.B) {
				for b.Loop() {
					dotW4A8SplitHalf4RowFold(&act[0], &corr[0], &packed4[0], &scales4[0], &dst[0], nGroups)
				}
			})
			sinkW4A8F32ARM64 = dst[0]
			return float64(r.NsPerOp())
		})
		pre := minOf3(func() float64 {
			r := testing.Benchmark(func(b *testing.B) {
				for b.Loop() {
					w4a8LaneCorrNeg8(&act[0], &corr[0], nGroups)
				}
			})
			return float64(r.NsPerOp())
		})
		macs := float64(4 * sh.K)
		t.Logf("K=%d (nGroups=%d), hot single quad: baseline %.1f ns (%.1f GMAC/s) | fold %.1f ns (%.1f GMAC/s) | fold/baseline %.3fx",
			sh.K, nGroups, base, macs/base, fold, macs/fold, base/fold)
		t.Logf("K=%d: corr pre-pass %.1f ns per activation row = %.3f%% of one N=%d projection's %d quad calls at the fold rate",
			sh.K, pre, 100*pre/(fold*float64(sh.N/4)), sh.N, sh.N/4)
	}
}

// TestW4A8RowFold_streamedAB: the real dispatch (MatmulBTW4A8Row4Into, fold on vs
// off via SetW4A8RowFold) over a 12-matrix bank so no call reuses the previous one's
// residency (docs/internal/measuring-performance.md §1.3), serial and 6 workers,
// ABBA-interleaved pairs with a win count (rule 7). The serial row is the S-05
// single-core number in its streamed regime; the parallel row is what decode sees.
func TestW4A8RowFold_streamedAB(t *testing.T) {
	harnessOnly(t)
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	prev := w4a8RowFold
	defer SetW4A8RowFold(prev)
	rng := rand.New(rand.NewSource(4242))
	const group, bank, pairs = 32, 12, 3
	for _, sh := range []struct{ K, N int }{{1536, 8960}, {8960, 1536}} {
		type mat struct {
			row4 []byte
			s4   []float32
		}
		mats := make([]mat, bank)
		for i := range mats {
			w := make([]float32, sh.N*sh.K)
			for j := range w {
				w[j] = float32(rng.NormFloat64())
			}
			packed, scales := QuantizeGroupsInt4(w, sh.N, sh.K, group)
			mats[i] = mat{RepackW4A8Row4(packed, sh.N, sh.K, group), RepackW4A8Row4Scales(scales, sh.N, sh.K, group)}
		}
		a := make([]float32, sh.K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		dst := make([]float32, sh.N)
		bytesPerPass := float64(bank) * float64(len(mats[0].row4)+4*len(mats[0].s4))
		macsPerPass := float64(bank) * float64(sh.K*sh.N)

		for _, par := range []bool{false, true} {
			var ws Workspace
			if par {
				ws.SetThreshold(1)
				ws.SetWorkers(6)
			} else {
				ws.SetThreshold(1 << 60)
			}
			pass := func(on bool) float64 {
				SetW4A8RowFold(on)
				t0 := time.Now()
				for i := range mats {
					MatmulBTW4A8Row4Into(&ws, a, mats[i].row4, mats[i].s4, dst, 1, sh.K, sh.N, group)
				}
				return float64(time.Since(t0).Nanoseconds())
			}
			pass(true) // warm the code paths and the page mappings once
			pass(false)
			var sumOn, sumOff float64
			wins := 0
			for p := 0; p < pairs; p++ {
				var on, off float64
				if p%2 == 0 {
					on, off = pass(true), pass(false)
				} else {
					off, on = pass(false), pass(true)
				}
				sumOn += on
				sumOff += off
				if on < off {
					wins++
				}
				t.Logf("K=%d N=%d parallel=%v pair %d: fold %.2f ms/pass | unfolded %.2f ms/pass | fold is %.3fx", sh.K, sh.N, par, p, on/1e6, off/1e6, off/on)
			}
			t.Logf("K=%d N=%d parallel=%v: paired mean fold %.2f ms/pass (%.1f GMAC/s, %.1f GB/s) vs unfolded %.2f (%.1f GMAC/s) = %.3fx, fold wins %d/%d",
				sh.K, sh.N, par, sumOn/pairs/1e6, macsPerPass/(sumOn/pairs), bytesPerPass/(sumOn/pairs), sumOff/pairs/1e6, macsPerPass/(sumOff/pairs), sumOff/sumOn, wins, pairs)
		}
	}
}
