//go:build amd64

package linalg

import (
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// TestW4A8VNNIOpsPerByte is TestW4A8OpsPerByte's exact hot/cold methodology
// (same K/group/N shape, same box-quiet AIKIT_HARNESS gate — numbers are
// directly comparable to that test's own output) with dotW4A8FoldAVX512VNNI
// added as a third measurement alongside dotW4A8FoldAVX2, hot AND cold.
//
// Why this exists: dotW4A8FoldAVX512VNNI's own benchmark
// (BenchmarkW4A8_AVX512VNNIvsAVX2) is HOT ONLY — one row reused every call,
// everything L1-resident — so it measures compute throughput in isolation
// and says nothing about whether that win survives a real decode access
// pattern, where every call reads a DIFFERENT weight row from DRAM.
// TestW4A8OpsPerByte's own finding for the AVX2 kernel was that cold
// streaming is NOT issue-limited on this box's memory system (see
// TestW4A8IssueWidthProbe) — i.e. AVX2 already has idle issue slots while
// cold, which raises the question this test directly answers: does a
// cheaper per-MAC kernel (fewer instructions, VNNI's fused u8×s8→i32) show
// up as decode-time speedup, or does it get hidden under the same DRAM
// stalls that were already idling AVX2's issue ports?
//
// Only meaningful on a real AVX-512 VNNI+VL host — skips otherwise.
func TestW4A8VNNIOpsPerByte(t *testing.T) {
	harnessOnly(t)
	if !cpufeat.HasAVX2 {
		t.Skip("AVX2 required")
	}
	if !cpufeat.HasAVX512VNNIVL {
		t.Skip("AVX-512 VNNI+VL required")
	}
	const (
		K     = 5120 // same shape as TestW4A8OpsPerByte — numbers are comparable
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
	bytesPerRow := float64(bpr + nGroups*4)
	totalMB := float64(N) * bytesPerRow / 1e6

	row0 := packed[0:bpr]
	srow0 := scales[0:nGroups]

	// --- hot: one row reused every call, L1-resident ---
	hotAVX2 := testing.Benchmark(func(b *testing.B) {
		for b.Loop() {
			sinkW4A8F32 = dotW4A8FoldAVX2(&act[0], &row0[0], &srow0[0], nGroups)
		}
	})
	hotVNNI := testing.Benchmark(func(b *testing.B) {
		for b.Loop() {
			sinkW4A8F32 = dotW4A8FoldAVX512VNNI(&act[0], &row0[0], &srow0[0], nGroups)
		}
	})
	hotAVX2Ns := float64(hotAVX2.NsPerOp())
	hotVNNINs := float64(hotVNNI.NsPerOp())

	// --- cold: cycle through all N distinct rows (~totalMB, well past this
	// box's L3), same activation reused (matches the real matmul: one
	// quantized activation shared across all N output columns) ---
	coldAVX2 := testing.Benchmark(func(b *testing.B) {
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
	coldVNNI := testing.Benchmark(func(b *testing.B) {
		i := 0
		for b.Loop() {
			r := packed[i*bpr : i*bpr+bpr]
			s := scales[i*nGroups : i*nGroups+nGroups]
			sinkW4A8F32 = dotW4A8FoldAVX512VNNI(&act[0], &r[0], &s[0], nGroups)
			i++
			if i == N {
				i = 0
			}
		}
	})
	coldAVX2Ns := float64(coldAVX2.NsPerOp())
	coldVNNINs := float64(coldVNNI.NsPerOp())

	hotSpeedup := hotAVX2Ns / hotVNNINs
	coldSpeedup := coldAVX2Ns / coldVNNINs
	coldAVX2GBs := bytesPerRow / coldAVX2Ns
	coldVNNIGBs := bytesPerRow / coldVNNINs

	t.Logf("hot  AVX2  (L1-resident, K=%d): %.2f ns/call  %.2f GMAC/s", K, hotAVX2Ns, float64(K)/hotAVX2Ns)
	t.Logf("hot  VNNI  (L1-resident, K=%d): %.2f ns/call  %.2f GMAC/s", K, hotVNNINs, float64(K)/hotVNNINs)
	t.Logf("hot  VNNI/AVX2 speedup: %.3fx", hotSpeedup)
	t.Logf("cold AVX2  (streaming %d rows, %.1f MB): %.2f ns/call  %.2f GMAC/s  %.2f GB/s", N, totalMB, coldAVX2Ns, float64(K)/coldAVX2Ns, coldAVX2GBs)
	t.Logf("cold VNNI  (streaming %d rows, %.1f MB): %.2f ns/call  %.2f GMAC/s  %.2f GB/s", N, totalMB, coldVNNINs, float64(K)/coldVNNINs, coldVNNIGBs)
	t.Logf("cold VNNI/AVX2 speedup: %.3fx", coldSpeedup)
	t.Logf("survival: cold speedup is %.1f%% of hot speedup", 100*(coldSpeedup-1)/(hotSpeedup-1+1e-9))

	switch {
	case coldSpeedup < 1.05:
		t.Logf("VERDICT: VNNI's hot win (%.2fx) DOES NOT SURVIVE cold streaming (%.2fx, ~1.0) — this access pattern is memory-bound, VNNI's cheaper per-MAC compute is hidden under DRAM stalls the same way TestW4A8IssueWidthProbe already found AVX2's to be. A faster kernel would not move real decode throughput at this K/group shape.", hotSpeedup, coldSpeedup)
	case coldSpeedup > hotSpeedup*0.8:
		t.Logf("VERDICT: VNNI's win MOSTLY SURVIVES cold streaming (%.2fx cold vs %.2fx hot) — this access pattern is still compute-sensitive enough that the cheaper kernel shows up as real throughput, not just a hot-cache artifact.", coldSpeedup, hotSpeedup)
	default:
		t.Logf("VERDICT: VNNI's win PARTIALLY SURVIVES cold streaming (%.2fx cold vs %.2fx hot) — some of the hot-path gain is masked by DRAM stalls, some is not.", coldSpeedup, hotSpeedup)
	}
}
