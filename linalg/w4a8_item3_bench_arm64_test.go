//go:build arm64

package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// TestW4A8Item3SplitHalfVsOriginal is the item-3 harness's go/no-go
// measurement (docs/prompts/w4a8-item3-harness.md): dotW4A8SplitHalfSDOT
// (split-half layout, signed centering, 1-row) against the production
// dotW4A8FoldSDOT (canonical layout, signed centering, 1-row) — same
// centering and row-count, ONLY the layout/unpack-prologue differs, so any
// measured delta isolates item 3's own effect before the row-interleave or
// uncentered-correction axes are added on top.
//
// TestW4A8Item3TwoAccVsOriginal is the follow-up probe (see dot_w4a8_arm64.s
// for the full motivation): dotW4A8FoldSDOT2Acc against the same baseline,
// canonical layout unchanged, testing whether the serial VFMLA fold chain
// (not the unpack prologue) is the real bottleneck.
//
// Real shape from the 1.5B GGUF metadata (gate/up projection): K=1536,
// N=8960 — not the borrowed 27B shape, per probe 2's lesson
// (docs/task-w4a8-neon-bandwidth.md).
func benchW4A8Shape(t *testing.T, seed int64) (act []int8, packed, splitHalf []byte, scales []float32, bpr, nGroups int, bytesPerRow float64) {
	const (
		K     = 1536
		group = 32
		N     = 8960
	)
	nGroups = K / group

	rng := rand.New(rand.NewSource(seed))
	act = make([]int8, K)
	for i := range act {
		act[i] = int8(rng.Intn(255) - 128)
	}

	w := make([]float32, N*K)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	packed, scales = QuantizeGroupsInt4(w, N, K, group)
	bpr = (K + 1) / 2
	bytesPerRow = float64(bpr + nGroups*4)

	splitHalf = make([]byte, len(packed))
	for r := range N {
		row := packed[r*bpr : r*bpr+bpr]
		copy(splitHalf[r*bpr:r*bpr+bpr], repackSplitHalfRow(row, K))
	}
	return
}

func TestW4A8Item3SplitHalfVsOriginal(t *testing.T) {
	harnessOnly(t)
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const N = 8960
	act, packed, splitHalf, scales, bpr, nGroups, bytesPerRow := benchW4A8Shape(t, 21)
	K := len(act)

	runHotOrig := func() float64 {
		row0 := packed[0:bpr]
		s0 := scales[0:nGroups]
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				sinkW4A8F32ARM64 = dotW4A8FoldSDOT(&act[0], &row0[0], &s0[0], nGroups)
			}
		})
		return float64(r.NsPerOp())
	}
	runHotSplit := func() float64 {
		row0 := splitHalf[0:bpr]
		s0 := scales[0:nGroups]
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				sinkW4A8F32ARM64 = dotW4A8SplitHalfSDOT(&act[0], &row0[0], &s0[0], nGroups)
			}
		})
		return float64(r.NsPerOp())
	}
	runColdOrig := func() float64 {
		i := 0
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				row := packed[i*bpr : i*bpr+bpr]
				s := scales[i*nGroups : i*nGroups+nGroups]
				sinkW4A8F32ARM64 = dotW4A8FoldSDOT(&act[0], &row[0], &s[0], nGroups)
				i++
				if i == N {
					i = 0
				}
			}
		})
		return float64(r.NsPerOp())
	}
	runColdSplit := func() float64 {
		i := 0
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				row := splitHalf[i*bpr : i*bpr+bpr]
				s := scales[i*nGroups : i*nGroups+nGroups]
				sinkW4A8F32ARM64 = dotW4A8SplitHalfSDOT(&act[0], &row[0], &s[0], nGroups)
				i++
				if i == N {
					i = 0
				}
			}
		})
		return float64(r.NsPerOp())
	}

	hotOrigNs, hotSplitNs := math.Inf(1), math.Inf(1)
	coldOrigNs, coldSplitNs := math.Inf(1), math.Inf(1)
	for rep := range 3 {
		switch rep {
		case 0:
			hotOrigNs = min(hotOrigNs, runHotOrig())
			hotSplitNs = min(hotSplitNs, runHotSplit())
			coldOrigNs = min(coldOrigNs, runColdOrig())
			coldSplitNs = min(coldSplitNs, runColdSplit())
		case 1:
			hotSplitNs = min(hotSplitNs, runHotSplit())
			coldSplitNs = min(coldSplitNs, runColdSplit())
			hotOrigNs = min(hotOrigNs, runHotOrig())
			coldOrigNs = min(coldOrigNs, runColdOrig())
		default:
			coldSplitNs = min(coldSplitNs, runColdSplit())
			coldOrigNs = min(coldOrigNs, runColdOrig())
			hotSplitNs = min(hotSplitNs, runHotSplit())
			hotOrigNs = min(hotOrigNs, runHotOrig())
		}
	}

	hotOrigG, hotSplitG := float64(K)/hotOrigNs, float64(K)/hotSplitNs
	coldOrigG, coldSplitG := float64(K)/coldOrigNs, float64(K)/coldSplitNs
	coldOrigGBs := bytesPerRow / coldOrigNs
	coldSplitGBs := bytesPerRow / coldSplitNs

	t.Logf("hot  (L1-resident, K=%d): orig %.2f ns/call %.2f GMAC/s | split-half %.2f ns/call %.2f GMAC/s | %.3fx",
		K, hotOrigNs, hotOrigG, hotSplitNs, hotSplitG, hotOrigNs/hotSplitNs)
	t.Logf("cold (streaming %d rows): orig %.2f ns/call %.2f GMAC/s %.2f GB/s | split-half %.2f ns/call %.2f GMAC/s %.2f GB/s | %.3fx",
		N, coldOrigNs, coldOrigG, coldOrigGBs, coldSplitNs, coldSplitG, coldSplitGBs, coldOrigNs/coldSplitNs)
}

func TestW4A8Item3TwoAccVsOriginal(t *testing.T) {
	harnessOnly(t)
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const N = 8960
	act, packed, _, scales, bpr, nGroups, bytesPerRow := benchW4A8Shape(t, 22)
	K := len(act)

	runHotOrig := func() float64 {
		row0 := packed[0:bpr]
		s0 := scales[0:nGroups]
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				sinkW4A8F32ARM64 = dotW4A8FoldSDOT(&act[0], &row0[0], &s0[0], nGroups)
			}
		})
		return float64(r.NsPerOp())
	}
	runHot2Acc := func() float64 {
		row0 := packed[0:bpr]
		s0 := scales[0:nGroups]
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				sinkW4A8F32ARM64 = dotW4A8FoldSDOT2Acc(&act[0], &row0[0], &s0[0], nGroups)
			}
		})
		return float64(r.NsPerOp())
	}
	runColdOrig := func() float64 {
		i := 0
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				row := packed[i*bpr : i*bpr+bpr]
				s := scales[i*nGroups : i*nGroups+nGroups]
				sinkW4A8F32ARM64 = dotW4A8FoldSDOT(&act[0], &row[0], &s[0], nGroups)
				i++
				if i == N {
					i = 0
				}
			}
		})
		return float64(r.NsPerOp())
	}
	runCold2Acc := func() float64 {
		i := 0
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				row := packed[i*bpr : i*bpr+bpr]
				s := scales[i*nGroups : i*nGroups+nGroups]
				sinkW4A8F32ARM64 = dotW4A8FoldSDOT2Acc(&act[0], &row[0], &s[0], nGroups)
				i++
				if i == N {
					i = 0
				}
			}
		})
		return float64(r.NsPerOp())
	}

	hotOrigNs, hot2AccNs := math.Inf(1), math.Inf(1)
	coldOrigNs, cold2AccNs := math.Inf(1), math.Inf(1)
	for rep := range 3 {
		switch rep {
		case 0:
			hotOrigNs = min(hotOrigNs, runHotOrig())
			hot2AccNs = min(hot2AccNs, runHot2Acc())
			coldOrigNs = min(coldOrigNs, runColdOrig())
			cold2AccNs = min(cold2AccNs, runCold2Acc())
		case 1:
			hot2AccNs = min(hot2AccNs, runHot2Acc())
			cold2AccNs = min(cold2AccNs, runCold2Acc())
			hotOrigNs = min(hotOrigNs, runHotOrig())
			coldOrigNs = min(coldOrigNs, runColdOrig())
		default:
			cold2AccNs = min(cold2AccNs, runCold2Acc())
			coldOrigNs = min(coldOrigNs, runColdOrig())
			hot2AccNs = min(hot2AccNs, runHot2Acc())
			hotOrigNs = min(hotOrigNs, runHotOrig())
		}
	}

	hotOrigG, hot2AccG := float64(K)/hotOrigNs, float64(K)/hot2AccNs
	coldOrigG, cold2AccG := float64(K)/coldOrigNs, float64(K)/cold2AccNs
	coldOrigGBs := bytesPerRow / coldOrigNs
	cold2AccGBs := bytesPerRow / cold2AccNs

	t.Logf("hot  (L1-resident, K=%d): orig %.2f ns/call %.2f GMAC/s | 2acc %.2f ns/call %.2f GMAC/s | %.3fx",
		K, hotOrigNs, hotOrigG, hot2AccNs, hot2AccG, hotOrigNs/hot2AccNs)
	t.Logf("cold (streaming %d rows): orig %.2f ns/call %.2f GMAC/s %.2f GB/s | 2acc %.2f ns/call %.2f GMAC/s %.2f GB/s | %.3fx",
		N, coldOrigNs, coldOrigG, coldOrigGBs, cold2AccNs, cold2AccG, cold2AccGBs, coldOrigNs/cold2AccNs)
}

func TestW4A8Item3FourAccVsOriginal(t *testing.T) {
	harnessOnly(t)
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const N = 8960
	act, packed, _, scales, bpr, nGroups, bytesPerRow := benchW4A8Shape(t, 23)
	K := len(act)

	runHotOrig := func() float64 {
		row0 := packed[0:bpr]
		s0 := scales[0:nGroups]
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				sinkW4A8F32ARM64 = dotW4A8FoldSDOT(&act[0], &row0[0], &s0[0], nGroups)
			}
		})
		return float64(r.NsPerOp())
	}
	runHot4Acc := func() float64 {
		row0 := packed[0:bpr]
		s0 := scales[0:nGroups]
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				sinkW4A8F32ARM64 = dotW4A8FoldSDOT4Acc(&act[0], &row0[0], &s0[0], nGroups)
			}
		})
		return float64(r.NsPerOp())
	}
	runColdOrig := func() float64 {
		i := 0
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				row := packed[i*bpr : i*bpr+bpr]
				s := scales[i*nGroups : i*nGroups+nGroups]
				sinkW4A8F32ARM64 = dotW4A8FoldSDOT(&act[0], &row[0], &s[0], nGroups)
				i++
				if i == N {
					i = 0
				}
			}
		})
		return float64(r.NsPerOp())
	}
	runCold4Acc := func() float64 {
		i := 0
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				row := packed[i*bpr : i*bpr+bpr]
				s := scales[i*nGroups : i*nGroups+nGroups]
				sinkW4A8F32ARM64 = dotW4A8FoldSDOT4Acc(&act[0], &row[0], &s[0], nGroups)
				i++
				if i == N {
					i = 0
				}
			}
		})
		return float64(r.NsPerOp())
	}

	hotOrigNs, hot4AccNs := math.Inf(1), math.Inf(1)
	coldOrigNs, cold4AccNs := math.Inf(1), math.Inf(1)
	for rep := range 3 {
		switch rep {
		case 0:
			hotOrigNs = min(hotOrigNs, runHotOrig())
			hot4AccNs = min(hot4AccNs, runHot4Acc())
			coldOrigNs = min(coldOrigNs, runColdOrig())
			cold4AccNs = min(cold4AccNs, runCold4Acc())
		case 1:
			hot4AccNs = min(hot4AccNs, runHot4Acc())
			cold4AccNs = min(cold4AccNs, runCold4Acc())
			hotOrigNs = min(hotOrigNs, runHotOrig())
			coldOrigNs = min(coldOrigNs, runColdOrig())
		default:
			cold4AccNs = min(cold4AccNs, runCold4Acc())
			coldOrigNs = min(coldOrigNs, runColdOrig())
			hot4AccNs = min(hot4AccNs, runHot4Acc())
			hotOrigNs = min(hotOrigNs, runHotOrig())
		}
	}

	hotOrigG, hot4AccG := float64(K)/hotOrigNs, float64(K)/hot4AccNs
	coldOrigG, cold4AccG := float64(K)/coldOrigNs, float64(K)/cold4AccNs
	coldOrigGBs := bytesPerRow / coldOrigNs
	cold4AccGBs := bytesPerRow / cold4AccNs

	t.Logf("hot  (L1-resident, K=%d): orig %.2f ns/call %.2f GMAC/s | 4acc %.2f ns/call %.2f GMAC/s | %.3fx",
		K, hotOrigNs, hotOrigG, hot4AccNs, hot4AccG, hotOrigNs/hot4AccNs)
	t.Logf("cold (streaming %d rows): orig %.2f ns/call %.2f GMAC/s %.2f GB/s | 4acc %.2f ns/call %.2f GMAC/s %.2f GB/s | %.3fx",
		N, coldOrigNs, coldOrigG, coldOrigGBs, cold4AccNs, cold4AccG, cold4AccGBs, coldOrigNs/cold4AccNs)
}

func TestW4A8Item3SplitHalf2AccVsOriginal(t *testing.T) {
	harnessOnly(t)
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const N = 8960
	act, packed, splitHalf, scales, bpr, nGroups, bytesPerRow := benchW4A8Shape(t, 24)
	K := len(act)

	runHotOrig := func() float64 {
		row0 := packed[0:bpr]
		s0 := scales[0:nGroups]
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				sinkW4A8F32ARM64 = dotW4A8FoldSDOT(&act[0], &row0[0], &s0[0], nGroups)
			}
		})
		return float64(r.NsPerOp())
	}
	runHotCombo := func() float64 {
		row0 := splitHalf[0:bpr]
		s0 := scales[0:nGroups]
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				sinkW4A8F32ARM64 = dotW4A8SplitHalf2Acc(&act[0], &row0[0], &s0[0], nGroups)
			}
		})
		return float64(r.NsPerOp())
	}
	runColdOrig := func() float64 {
		i := 0
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				row := packed[i*bpr : i*bpr+bpr]
				s := scales[i*nGroups : i*nGroups+nGroups]
				sinkW4A8F32ARM64 = dotW4A8FoldSDOT(&act[0], &row[0], &s[0], nGroups)
				i++
				if i == N {
					i = 0
				}
			}
		})
		return float64(r.NsPerOp())
	}
	runColdCombo := func() float64 {
		i := 0
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				row := splitHalf[i*bpr : i*bpr+bpr]
				s := scales[i*nGroups : i*nGroups+nGroups]
				sinkW4A8F32ARM64 = dotW4A8SplitHalf2Acc(&act[0], &row[0], &s[0], nGroups)
				i++
				if i == N {
					i = 0
				}
			}
		})
		return float64(r.NsPerOp())
	}

	hotOrigNs, hotComboNs := math.Inf(1), math.Inf(1)
	coldOrigNs, coldComboNs := math.Inf(1), math.Inf(1)
	for rep := range 3 {
		switch rep {
		case 0:
			hotOrigNs = min(hotOrigNs, runHotOrig())
			hotComboNs = min(hotComboNs, runHotCombo())
			coldOrigNs = min(coldOrigNs, runColdOrig())
			coldComboNs = min(coldComboNs, runColdCombo())
		case 1:
			hotComboNs = min(hotComboNs, runHotCombo())
			coldComboNs = min(coldComboNs, runColdCombo())
			hotOrigNs = min(hotOrigNs, runHotOrig())
			coldOrigNs = min(coldOrigNs, runColdOrig())
		default:
			coldComboNs = min(coldComboNs, runColdCombo())
			coldOrigNs = min(coldOrigNs, runColdOrig())
			hotComboNs = min(hotComboNs, runHotCombo())
			hotOrigNs = min(hotOrigNs, runHotOrig())
		}
	}

	hotOrigG, hotComboG := float64(K)/hotOrigNs, float64(K)/hotComboNs
	coldOrigG, coldComboG := float64(K)/coldOrigNs, float64(K)/coldComboNs
	coldOrigGBs := bytesPerRow / coldOrigNs
	coldComboGBs := bytesPerRow / coldComboNs

	t.Logf("hot  (L1-resident, K=%d): orig %.2f ns/call %.2f GMAC/s | split-half+2acc %.2f ns/call %.2f GMAC/s | %.3fx",
		K, hotOrigNs, hotOrigG, hotComboNs, hotComboG, hotOrigNs/hotComboNs)
	t.Logf("cold (streaming %d rows): orig %.2f ns/call %.2f GMAC/s %.2f GB/s | split-half+2acc %.2f ns/call %.2f GMAC/s %.2f GB/s | %.3fx",
		N, coldOrigNs, coldOrigG, coldOrigGBs, coldComboNs, coldComboG, coldComboGBs, coldOrigNs/coldComboNs)
}

func TestW4A8Item3SplitHalf4AccVsOriginal(t *testing.T) {
	harnessOnly(t)
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const N = 8960
	act, packed, splitHalf, scales, bpr, nGroups, bytesPerRow := benchW4A8Shape(t, 25)
	K := len(act)

	runHotOrig := func() float64 {
		row0 := packed[0:bpr]
		s0 := scales[0:nGroups]
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				sinkW4A8F32ARM64 = dotW4A8FoldSDOT(&act[0], &row0[0], &s0[0], nGroups)
			}
		})
		return float64(r.NsPerOp())
	}
	runHotCombo4 := func() float64 {
		row0 := splitHalf[0:bpr]
		s0 := scales[0:nGroups]
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				sinkW4A8F32ARM64 = dotW4A8SplitHalf4Acc(&act[0], &row0[0], &s0[0], nGroups)
			}
		})
		return float64(r.NsPerOp())
	}
	runColdOrig := func() float64 {
		i := 0
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				row := packed[i*bpr : i*bpr+bpr]
				s := scales[i*nGroups : i*nGroups+nGroups]
				sinkW4A8F32ARM64 = dotW4A8FoldSDOT(&act[0], &row[0], &s[0], nGroups)
				i++
				if i == N {
					i = 0
				}
			}
		})
		return float64(r.NsPerOp())
	}
	runColdCombo4 := func() float64 {
		i := 0
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				row := splitHalf[i*bpr : i*bpr+bpr]
				s := scales[i*nGroups : i*nGroups+nGroups]
				sinkW4A8F32ARM64 = dotW4A8SplitHalf4Acc(&act[0], &row[0], &s[0], nGroups)
				i++
				if i == N {
					i = 0
				}
			}
		})
		return float64(r.NsPerOp())
	}

	hotOrigNs, hotCombo4Ns := math.Inf(1), math.Inf(1)
	coldOrigNs, coldCombo4Ns := math.Inf(1), math.Inf(1)
	for rep := range 3 {
		switch rep {
		case 0:
			hotOrigNs = min(hotOrigNs, runHotOrig())
			hotCombo4Ns = min(hotCombo4Ns, runHotCombo4())
			coldOrigNs = min(coldOrigNs, runColdOrig())
			coldCombo4Ns = min(coldCombo4Ns, runColdCombo4())
		case 1:
			hotCombo4Ns = min(hotCombo4Ns, runHotCombo4())
			coldCombo4Ns = min(coldCombo4Ns, runColdCombo4())
			hotOrigNs = min(hotOrigNs, runHotOrig())
			coldOrigNs = min(coldOrigNs, runColdOrig())
		default:
			coldCombo4Ns = min(coldCombo4Ns, runColdCombo4())
			coldOrigNs = min(coldOrigNs, runColdOrig())
			hotCombo4Ns = min(hotCombo4Ns, runHotCombo4())
			hotOrigNs = min(hotOrigNs, runHotOrig())
		}
	}

	hotOrigG, hotCombo4G := float64(K)/hotOrigNs, float64(K)/hotCombo4Ns
	coldOrigG, coldCombo4G := float64(K)/coldOrigNs, float64(K)/coldCombo4Ns
	coldOrigGBs := bytesPerRow / coldOrigNs
	coldCombo4GBs := bytesPerRow / coldCombo4Ns

	t.Logf("hot  (L1-resident, K=%d): orig %.2f ns/call %.2f GMAC/s | split-half+4acc %.2f ns/call %.2f GMAC/s | %.3fx",
		K, hotOrigNs, hotOrigG, hotCombo4Ns, hotCombo4G, hotOrigNs/hotCombo4Ns)
	t.Logf("cold (streaming %d rows): orig %.2f ns/call %.2f GMAC/s %.2f GB/s | split-half+4acc %.2f ns/call %.2f GMAC/s %.2f GB/s | %.3fx",
		N, coldOrigNs, coldOrigG, coldOrigGBs, coldCombo4Ns, coldCombo4G, coldCombo4GBs, coldOrigNs/coldCombo4Ns)
}

// TestW4A8Item4Row4VsBaselines is item 4's harness go/no-go measurement: 4
// REAL output rows per call (dotW4A8SplitHalf4Row, activation load shared)
// against (a) 4 separate calls to the production dotW4A8FoldSDOT and (b) 4
// separate calls to the current-best single-row kernel
// (dotW4A8SplitHalf2Acc) — isolating item 4's own effect on top of the
// already-established layout+accumulator win, per feedback that item 4
// targets a genuinely different resource (activation-load/instruction-count
// amortization across real outputs) than the accumulator-lane question the
// 4-lane check already settled.
func TestW4A8Item4Row4VsBaselines(t *testing.T) {
	harnessOnly(t)
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const (
		K     = 1536
		group = 32
		N     = 8960
	)
	nGroups := K / group
	bpr := (K + 1) / 2
	bytesPerRow := float64(bpr + nGroups*4)

	rng := rand.New(rand.NewSource(43))
	act := make([]int8, K)
	for i := range act {
		act[i] = int8(rng.Intn(255) - 128)
	}
	w := make([]float32, N*K)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	packed, scales := QuantizeGroupsInt4(w, N, K, group)
	splitHalf := make([]byte, len(packed))
	for r := range N {
		row := packed[r*bpr : r*bpr+bpr]
		copy(splitHalf[r*bpr:r*bpr+bpr], repackSplitHalfRow(row, K))
	}
	// Interleave every 4 consecutive rows into item 4's block layout.
	packed4 := make([]byte, N*bpr)
	scales4 := make([]float32, N*nGroups)
	for q := range N / 4 {
		r0, r1, r2, r3 := q*4, q*4+1, q*4+2, q*4+3
		blk := repackSplitHalf4RowBlock(
			packed[r0*bpr:r0*bpr+bpr], packed[r1*bpr:r1*bpr+bpr],
			packed[r2*bpr:r2*bpr+bpr], packed[r3*bpr:r3*bpr+bpr], K)
		copy(packed4[q*4*bpr:(q*4+4)*bpr], blk)
		sblk := interleaveScales4Row(
			scales[r0*nGroups:r0*nGroups+nGroups], scales[r1*nGroups:r1*nGroups+nGroups],
			scales[r2*nGroups:r2*nGroups+nGroups], scales[r3*nGroups:r3*nGroups+nGroups], nGroups)
		copy(scales4[q*4*nGroups:(q*4+4)*nGroups], sblk)
	}

	runHotOrigX4 := func() float64 {
		row0 := packed[0:bpr]
		s0 := scales[0:nGroups]
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				for range 4 {
					sinkW4A8F32ARM64 = dotW4A8FoldSDOT(&act[0], &row0[0], &s0[0], nGroups)
				}
			}
		})
		return float64(r.NsPerOp())
	}
	runHotComboX4 := func() float64 {
		row0 := splitHalf[0:bpr]
		s0 := scales[0:nGroups]
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				for range 4 {
					sinkW4A8F32ARM64 = dotW4A8SplitHalf2Acc(&act[0], &row0[0], &s0[0], nGroups)
				}
			}
		})
		return float64(r.NsPerOp())
	}
	runHotRow4 := func() float64 {
		blk0 := packed4[0 : 4*bpr]
		sblk0 := scales4[0 : 4*nGroups]
		var dst [4]float32
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				dotW4A8SplitHalf4Row(&act[0], &blk0[0], &sblk0[0], &dst[0], nGroups)
			}
		})
		sinkW4A8F32ARM64 = dst[0]
		return float64(r.NsPerOp())
	}
	runColdOrigX4 := func() float64 {
		i := 0
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				for range 4 {
					row := packed[i*bpr : i*bpr+bpr]
					s := scales[i*nGroups : i*nGroups+nGroups]
					sinkW4A8F32ARM64 = dotW4A8FoldSDOT(&act[0], &row[0], &s[0], nGroups)
					i++
					if i == N {
						i = 0
					}
				}
			}
		})
		return float64(r.NsPerOp())
	}
	runColdComboX4 := func() float64 {
		i := 0
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				for range 4 {
					row := splitHalf[i*bpr : i*bpr+bpr]
					s := scales[i*nGroups : i*nGroups+nGroups]
					sinkW4A8F32ARM64 = dotW4A8SplitHalf2Acc(&act[0], &row[0], &s[0], nGroups)
					i++
					if i == N {
						i = 0
					}
				}
			}
		})
		return float64(r.NsPerOp())
	}
	runColdRow4 := func() float64 {
		q := 0
		nQuads := N / 4
		var dst [4]float32
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				blk := packed4[q*4*bpr : q*4*bpr+4*bpr]
				sblk := scales4[q*4*nGroups : q*4*nGroups+4*nGroups]
				dotW4A8SplitHalf4Row(&act[0], &blk[0], &sblk[0], &dst[0], nGroups)
				q++
				if q == nQuads {
					q = 0
				}
			}
		})
		sinkW4A8F32ARM64 = dst[0]
		return float64(r.NsPerOp())
	}

	hotOrigNs, hotComboNs, hotRow4Ns := math.Inf(1), math.Inf(1), math.Inf(1)
	coldOrigNs, coldComboNs, coldRow4Ns := math.Inf(1), math.Inf(1), math.Inf(1)
	for rep := range 3 {
		switch rep {
		case 0:
			hotOrigNs = min(hotOrigNs, runHotOrigX4())
			hotComboNs = min(hotComboNs, runHotComboX4())
			hotRow4Ns = min(hotRow4Ns, runHotRow4())
			coldOrigNs = min(coldOrigNs, runColdOrigX4())
			coldComboNs = min(coldComboNs, runColdComboX4())
			coldRow4Ns = min(coldRow4Ns, runColdRow4())
		case 1:
			hotRow4Ns = min(hotRow4Ns, runHotRow4())
			coldRow4Ns = min(coldRow4Ns, runColdRow4())
			hotComboNs = min(hotComboNs, runHotComboX4())
			coldComboNs = min(coldComboNs, runColdComboX4())
			hotOrigNs = min(hotOrigNs, runHotOrigX4())
			coldOrigNs = min(coldOrigNs, runColdOrigX4())
		default:
			coldRow4Ns = min(coldRow4Ns, runColdRow4())
			coldComboNs = min(coldComboNs, runColdComboX4())
			coldOrigNs = min(coldOrigNs, runColdOrigX4())
			hotRow4Ns = min(hotRow4Ns, runHotRow4())
			hotComboNs = min(hotComboNs, runHotComboX4())
			hotOrigNs = min(hotOrigNs, runHotOrigX4())
		}
	}

	totalMACs := float64(4 * K)
	hotOrigG, hotComboG, hotRow4G := totalMACs/hotOrigNs, totalMACs/hotComboNs, totalMACs/hotRow4Ns
	coldOrigG, coldComboG, coldRow4G := totalMACs/coldOrigNs, totalMACs/coldComboNs, totalMACs/coldRow4Ns
	totalBytes4 := 4 * bytesPerRow
	coldOrigGBs, coldComboGBs, coldRow4GBs := totalBytes4/coldOrigNs, totalBytes4/coldComboNs, totalBytes4/coldRow4Ns

	t.Logf("hot  (L1-resident, 4 rows, K=%d): orig×4 %.2f ns %.2f GMAC/s | combo×4 %.2f ns %.2f GMAC/s | row4 %.2f ns %.2f GMAC/s | row4/orig×4 %.3fx | row4/combo×4 %.3fx",
		K, hotOrigNs, hotOrigG, hotComboNs, hotComboG, hotRow4Ns, hotRow4G, hotOrigNs/hotRow4Ns, hotComboNs/hotRow4Ns)
	t.Logf("cold (streaming %d row-quads): orig×4 %.2f ns %.2f GMAC/s %.2f GB/s | combo×4 %.2f ns %.2f GMAC/s %.2f GB/s | row4 %.2f ns %.2f GMAC/s %.2f GB/s | row4/orig×4 %.3fx | row4/combo×4 %.3fx",
		N/4, coldOrigNs, coldOrigG, coldOrigGBs, coldComboNs, coldComboG, coldComboGBs, coldRow4Ns, coldRow4G, coldRow4GBs, coldOrigNs/coldRow4Ns, coldComboNs/coldRow4Ns)
}
