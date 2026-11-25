//go:build amd64

package linalg

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// TestW4A8OpsPerByte measures dotW4A8FoldAVX2's achieved throughput, hot
// (L1-resident) and cold (streaming from DRAM), against dotI8AVX2 — the
// existing W8A8 kernel, which has neither the nibble-unpack prologue nor the
// per-group scale fold, so its hot throughput is the natural "what would this
// cost with the int4-specific overhead removed" reference.
//
// Why this test exists: goinfer measured Qwen3.8-27B CPU decode at 2.55x
// slower than an Ollama/llama.cpp Q4_K_M engine on the same Ryzen 7 3700X box.
// Two explanations were ruled out with measurements before this one — the
// int4ParThreshold parallelization-threshold bug class (every real projection
// clears it by 5x-1200x) and the quant format's byte-count (~11% difference,
// nowhere near 2.55x). What's left is a claim about the kernel itself: goinfer's
// profile shows 16.8 GMAC/s / 11.7 GB/s achieved against a DDR4-3200 dual-channel
// peak of ~51 GB/s — neither engine saturates memory bandwidth, so this is a
// compute question, not a bandwidth one, and nobody had measured the kernel in
// isolation. This does.
//
// Shape: [17408, 5120] — the FFN gate/up/down matmul, the single largest
// per-token contributor in the goinfer profile (89.1M MACs at M=1).
//
// amd64/AVX2 only, and the numbers are box-specific (Ryzen 7 3700X, no VNNI) —
// run this on the same machine the goinfer profile came from, not the Mac
// (different kernel: dot_w4a8_arm64.s, different memory subsystem).
func TestW4A8OpsPerByte(t *testing.T) {
	harnessOnly(t)
	if !cpufeat.HasAVX2 {
		t.Skip("AVX2 required")
	}
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

	// --- hot: one row reused every call, packed/scales/act all L1-resident ---
	row0 := packed[0:bpr]
	srow0 := scales[0:nGroups]
	hotR := testing.Benchmark(func(b *testing.B) {
		for b.Loop() {
			sinkW4A8F32 = dotW4A8FoldAVX2(&act[0], &row0[0], &srow0[0], nGroups)
		}
	})
	hotNs := float64(hotR.NsPerOp())
	hotGMACs := float64(K) / hotNs

	// --- cold: cycle through all N distinct rows (~55.7 MB, well past this
	// box's L3) so every call's weight+scale read is a genuine DRAM fetch —
	// the activation row does NOT change (matches the real matmul: one
	// quantized activation reused across all N output columns), so it stays
	// L1-resident and is correctly excluded from the bytes-moved accounting.
	coldR := testing.Benchmark(func(b *testing.B) {
		i := 0
		for b.Loop() {
			r := packed[i*bpr : i*bpr+bpr]
			s := scales[i*nGroups : i*nGroups+nGroups]
			sinkW4A8F32 = dotW4A8FoldAVX2(&act[0], &r[0], &s[0], nGroups)
			i++
			if i == N {
				i = 0
			}
		}
	})
	coldNs := float64(coldR.NsPerOp())
	coldGMACs := float64(K) / coldNs
	coldGBs := bytesPerRow / coldNs // bytes/ns == GB/s

	// --- reference: dotI8AVX2, hot, same K — no nibble unpack, no per-group
	// scale fold, the existing proven W8A8 body dotW4A8FoldAVX2 itself reuses.
	act8 := make([]int8, K)
	w8 := make([]int8, K)
	for i := range act8 {
		act8[i] = int8(rng.Intn(255) - 128)
		w8[i] = int8(rng.Intn(255) - 128)
	}
	i8R := testing.Benchmark(func(b *testing.B) {
		for b.Loop() {
			sinkW4A8I32 = dotI8AVX2(&act8[0], &w8[0], K)
		}
	})
	i8Ns := float64(i8R.NsPerOp())
	i8GMACs := float64(K) / i8Ns

	t.Logf("W4A8 hot  (L1-resident, 1 row reused, K=%d):  %.2f ns/call  %.2f GMAC/s", K, hotNs, hotGMACs)
	t.Logf("W4A8 cold (streaming %d distinct rows, %.1f MB weights+scales): %.2f ns/call  %.2f GMAC/s  %.2f GB/s",
		N, float64(N)*bytesPerRow/1e6, coldNs, coldGMACs, coldGBs)
	t.Logf("I8AVX2 hot (no unpack, no per-group scale, K=%d, reference): %.2f ns/call  %.2f GMAC/s", K, i8Ns, i8GMACs)
	t.Logf("unpack+scale-fold tax: W4A8 hot / I8AVX2 hot = %.2fx more time per MAC", hotNs/i8Ns)
	t.Logf("streaming tax: W4A8 cold / hot = %.2fx slower than L1-resident compute", coldNs/hotNs)
	t.Logf("achieved/theoretical: %.1f%% of hot (compute-only) throughput realized cold", 100*coldGMACs/hotGMACs)

	if gflops, ok := MeasuredFMAPeakGFLOPS(); ok {
		t.Logf("context: SIMD f32 FMA peak %.1f GFLOPS (int MAC vs f32 FMA — not directly comparable, logged for scale only)", gflops)
	}
}

// TestW4A8IssueWidthProbe applies the marginal-FMA injection issue-width probe
// (docs/internal/priors-microgpt-c.md §1; method and helpers shared with
// linalg/fma_issue_probe_test.go — deadFMAs/resetAcc/lstsqSlope reused
// unchanged from there) to the COLD dotW4A8FoldAVX2 call from
// TestW4A8OpsPerByte, to settle the question that test's raw throughput
// numbers can't answer alone: is the cold call issue-limited (compute
// genuinely on the critical path — fewer instructions/MAC, e.g. a VNNI
// variant or a cheaper nibble-unpack, would directly help) or does it leave
// idle issue slots even while streaming from DRAM (the memory system is the
// ceiling, and shaving instructions off the inner loop would be hidden under
// stalls rather than showing up as decode-time speedup)?
func TestW4A8IssueWidthProbe(t *testing.T) {
	harnessOnly(t)
	if !cpufeat.HasAVX2 {
		t.Skip("AVX2 required")
	}
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
					sinkW4A8F32 = dotW4A8FoldAVX2(&act[0], &rrow[0], &srow[0], nGroups)
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
	t.Logf("dotW4A8FoldAVX2(cold) + N dead FMAs, stacked: %v ns/op over N=%v (marginal %.3f ns/FMA)", stackedNs, ns, stackedSlope)
	t.Logf("N dead FMAs alone:                                      %v ns/op over N=%v (marginal %.3f ns/FMA)", aloneNs, ns, aloneSlope)

	if aloneSlope <= 0 || stackedSlope <= 0 {
		t.Fatalf("non-positive marginal slope (stacked=%.4f alone=%.4f) — measurement produced nothing usable", stackedSlope, aloneSlope)
	}
	ratio := stackedSlope / aloneSlope
	switch {
	case ratio < 1.05:
		t.Logf("VERDICT: dotW4A8FoldAVX2 (cold) is NOT issue-limited — ratio %.2f (~1.0). Idle issue slots exist even while streaming from DRAM: the memory system, not instruction count, is the ceiling here.", ratio)
	default:
		t.Logf("VERDICT: dotW4A8FoldAVX2 (cold) IS issue-limited — ratio %.2f (>1.0). Compute is on the critical path even while cold: fewer instructions per MAC (nibble-unpack rework, a VNNI variant) would directly help.", ratio)
	}
}

var (
	sinkW4A8F32 float32
	sinkW4A8I32 int32
)
