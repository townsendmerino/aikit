//go:build arm64

package linalg

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// TestW4A8OpsPerByteARM64 is the arm64/NEON sibling of TestW4A8OpsPerByte
// (w4a8_opsperbyte_bench_test.go, amd64/AVX2): same method (hot L1-resident vs
// cold DRAM-streaming throughput of dotW4A8FoldSDOT against dotI8 -- the
// existing W8A8 kernel with neither the nibble-unpack prologue nor the
// per-group scale fold), same shape ([17408,5120], the FFN gate/up/down
// matmul), ported rather than re-derived so the two boxes' numbers sit next
// to each other in the same terms.
//
// Requested from goinfer (docs/prompts/mac-cpu-decode-vs-ollama.md, 2026-08-22)
// after a Mac CPU decode-only peer bench measured goinfer int8int8 at ~42
// GB/s implied weight-stream rate against ollama/llama.cpp's Q4_K_M at ~67
// GB/s on the same M1 Pro -- i.e. a gap survives even past the int4-specific
// unpack tax, and nobody had measured this kernel in isolation on arm64.
// Unlike amd64 (no VNNI on the Ryzen box the AVX2 numbers came from), Apple
// Silicon already has DotProd (SDOT) and dotW4A8FoldSDOT already uses it --
// so this box does NOT start from the amd64 finding's "no faster instruction
// available" wall; whether that closes any of the gap is exactly what this
// measures.
//
// arm64/NEON only, box-specific (Apple M1 Pro) -- run this on the Mac, not the
// amd64 box (different kernel: dot_w4a8_arm64.s vs quant_w4a8_amd64.go).
func TestW4A8OpsPerByteARM64(t *testing.T) {
	harnessOnly(t)
	const (
		K     = 5120
		group = 32
		N     = 17408
	)
	nGroups := K / group

	rng := rand.New(rand.NewSource(1))
	act := make([]int8, K)
	for i := range act {
		act[i] = int8(rng.Intn(255) - 128)
	}

	w := make([]float32, N*K)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	packed, scales := QuantizeGroupsInt4(w, N, K, group)
	bpr := (K + 1) / 2
	bytesPerRow := float64(bpr + nGroups*4) // packed nibbles + f32 group scales

	act8 := make([]int8, K)
	w8 := make([]int8, K)
	for i := range act8 {
		act8[i] = int8(rng.Intn(255) - 128)
		w8[i] = int8(rng.Intn(255) - 128)
	}

	runHot := func() float64 {
		row0 := packed[0:bpr]
		srow0 := scales[0:nGroups]
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				sinkW4A8F32ARM64 = dotW4A8FoldSDOT(&act[0], &row0[0], &srow0[0], nGroups)
			}
		})
		return float64(r.NsPerOp())
	}
	runCold := func() float64 {
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
	runRef := func() float64 {
		if !cpufeat.HasDotProd {
			return 0
		}
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				sinkW4A8I32ARM64 = dotI8SDOT(&act8[0], &w8[0], K)
			}
		})
		return float64(r.NsPerOp())
	}

	// Order-alternated best-of-N (P6a clock-ramp / contention lesson,
	// docs/internal/measuring-performance.md): a single fixed-order pass on
	// this box can be badly contaminated by whichever measurement runs first
	// on a not-yet-boosted core, or by contention from another process — a
	// first attempt at this test measured "hot" 12x SLOWER than "cold" on one
	// fixed-order run, which is physically impossible for L1-resident vs
	// DRAM-streaming and did not reproduce under repetition. Three repeats,
	// rotating which measurement goes first, min-of-N per measurement.
	hotNs, coldNs, i8Ns := math.Inf(1), math.Inf(1), math.Inf(1)
	takeRef := func() {
		if v := runRef(); v > 0 {
			i8Ns = min(i8Ns, v)
		}
	}
	for rep := range 3 {
		switch rep {
		case 0:
			hotNs = min(hotNs, runHot())
			coldNs = min(coldNs, runCold())
			takeRef()
		case 1:
			coldNs = min(coldNs, runCold())
			takeRef()
			hotNs = min(hotNs, runHot())
		default:
			takeRef()
			hotNs = min(hotNs, runHot())
			coldNs = min(coldNs, runCold())
		}
	}
	hotGMACs := float64(K) / hotNs
	coldGMACs := float64(K) / coldNs
	coldGBs := bytesPerRow / coldNs // bytes/ns == GB/s
	i8GMACs := float64(K) / i8Ns

	t.Logf("W4A8 hot  (L1-resident, 1 row reused, K=%d):  %.2f ns/call  %.2f GMAC/s", K, hotNs, hotGMACs)
	t.Logf("W4A8 cold (streaming %d distinct rows, %.1f MB weights+scales): %.2f ns/call  %.2f GMAC/s  %.2f GB/s",
		N, float64(N)*bytesPerRow/1e6, coldNs, coldGMACs, coldGBs)
	if i8Ns > 0 {
		t.Logf("I8SDOT hot (no unpack, no per-group scale, K=%d, reference): %.2f ns/call  %.2f GMAC/s", K, i8Ns, i8GMACs)
		t.Logf("unpack+scale-fold tax: W4A8 hot / I8SDOT hot = %.2fx more time per MAC", hotNs/i8Ns)
	}
	t.Logf("streaming tax: W4A8 cold / hot = %.2fx slower than L1-resident compute", coldNs/hotNs)
	t.Logf("achieved/theoretical: %.1f%% of hot (compute-only) throughput realized cold", 100*coldGMACs/hotGMACs)

	if gflops, ok := MeasuredFMAPeakGFLOPS(); ok {
		t.Logf("context: SIMD f32 FMA peak %.1f GFLOPS (int MAC vs f32 FMA -- not directly comparable, logged for scale only)", gflops)
	}
}

// TestW4A8IssueWidthProbeARM64 is the arm64 sibling of TestW4A8IssueWidthProbe:
// applies the marginal-FMA injection issue-width probe to the COLD
// dotW4A8FoldSDOT call, to settle whether the cold kernel is issue-limited
// (fewer instructions/MAC would directly help) or leaves idle issue slots even
// while streaming from DRAM (the memory system is the ceiling).
func TestW4A8IssueWidthProbeARM64(t *testing.T) {
	harnessOnly(t)
	const (
		K     = 5120
		group = 32
		N     = 17408
	)
	nGroups := K / group

	rng := rand.New(rand.NewSource(2))
	act := make([]int8, K)
	for i := range act {
		act[i] = int8(rng.Intn(255) - 128)
	}
	w := make([]float32, N*K)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	packed, scales := QuantizeGroupsInt4(w, N, K, group)
	bpr := (K + 1) / 2

	const maxN = 64
	acc := make([]float64, maxN)
	ns := []float64{0, 16, 32, 48, 64}
	stackedNs := make([]float64, len(ns))
	aloneNs := make([]float64, len(ns))
	const repeats = 3

	for pt, nf := range ns {
		n := int(nf)

		stackedNs[pt] = math.Inf(1)
		for range repeats {
			resetAcc(acc)
			i := 0
			r := testing.Benchmark(func(b *testing.B) {
				for b.Loop() {
					rrow := packed[i*bpr : i*bpr+bpr]
					srow := scales[i*nGroups : i*nGroups+nGroups]
					sinkW4A8F32ARM64 = dotW4A8FoldSDOT(&act[0], &rrow[0], &srow[0], nGroups)
					deadFMAs(acc[:n])
					i++
					if i == N {
						i = 0
					}
				}
			})
			stackedNs[pt] = min(stackedNs[pt], float64(r.NsPerOp()))
		}

		aloneNs[pt] = math.Inf(1)
		for range repeats {
			resetAcc(acc)
			r := testing.Benchmark(func(b *testing.B) {
				for b.Loop() {
					deadFMAs(acc[:n])
				}
			})
			aloneNs[pt] = min(aloneNs[pt], float64(r.NsPerOp()))
		}
	}
	sinkFMAProbeAcc = acc

	stackedSlope := lstsqSlope(ns, stackedNs)
	aloneSlope := lstsqSlope(ns, aloneNs)
	t.Logf("dotW4A8FoldSDOT(cold) + N dead FMAs, stacked: %v ns/op over N=%v (marginal %.3f ns/FMA)", stackedNs, ns, stackedSlope)
	t.Logf("N dead FMAs alone:                                      %v ns/op over N=%v (marginal %.3f ns/FMA)", aloneNs, ns, aloneSlope)

	if aloneSlope <= 0 || stackedSlope <= 0 {
		t.Fatalf("non-positive marginal slope (stacked=%.4f alone=%.4f) — measurement produced nothing usable", stackedSlope, aloneSlope)
	}
	ratio := stackedSlope / aloneSlope
	switch {
	case ratio < 1.05:
		t.Logf("VERDICT: dotW4A8FoldSDOT (cold) is NOT issue-limited — ratio %.2f (~1.0). Idle issue slots exist even while streaming from DRAM: the memory system, not instruction count, is the ceiling here.", ratio)
	default:
		t.Logf("VERDICT: dotW4A8FoldSDOT (cold) IS issue-limited — ratio %.2f (>1.0). Compute is on the critical path even while cold: fewer instructions per MAC would directly help.", ratio)
	}
}

var (
	sinkW4A8F32ARM64 float32
	sinkW4A8I32ARM64 int32
)
