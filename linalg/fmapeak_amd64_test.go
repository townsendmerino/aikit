//go:build amd64

package linalg

import (
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// TestFMAPeakAMD64_empirical measures this core's achievable f32 AVX2 FMA ceiling —
// the amd64 twin of TestFMAPeak_empirical, and the denominator every "% of peak"
// claim on an amd64 box needs.
//
// WHY IT HAD TO EXIST. BenchmarkGEMMPeakFraction carries no build tag, so it has
// always run here — against a hardcoded M1 Pro constant (3.2 GHz × 16 f32-FMA/cyc =
// 102.4). On this Zen 2 box it therefore reported the GEMM at "~50 %peak" where the
// real figure is ~38%. A wrong denominator does not announce itself; it produces a
// plausible number, which is worse than no number.
//
// WHAT IT ASSERTS, AND WHY THAT AND NOT A THRESHOLD. It checks the IMPLIED CLOCK, not
// the GFLOPS: Zen 2 retires 2 × 256-bit FMA per cycle = 32 f32 flops/cycle, so
// GFLOPS/32 must land on a frequency the part can actually reach. That makes the
// assertion independent of which amd64 CPU runs it, and it catches the failure that
// matters — a probe that is not measuring FMA throughput at all:
//
//   - too few in-flight accumulators measures FMA LATENCY instead. Verified: cutting
//     the loop from 14 accumulators to 2 gives 27.3 GFLOPS and an implied 0.85 GHz,
//     which this test rejects. (It also confirms Zen 2's ~5-cycle FMA latency, since
//     2 FMAs per 5 cycles is exactly that rate — the reason 14 is the right count.)
//   - elided or optimized-away work reads implausibly high.
//
// A 6-accumulator version would have read ~82 GFLOPS and made the GEMM look like a
// believable, wrong 63% of peak. The clock check is what separates believable from
// true.
func TestFMAPeakAMD64_empirical(t *testing.T) {
	if !cpufeat.HasAVX2 {
		t.Skip("no AVX2+FMA on this CPU")
	}
	const (
		flopsPerCycle  = 32.0 // Zen 2: 2 FMA pipes × 8 f32 lanes × (mul+add)
		minPlausibleHz = 1.5  // below this, suspect the loop is latency-bound
		maxPlausibleHz = 6.0  // above this, suspect work is being elided
	)
	g, ok := MeasuredFMAPeakGFLOPS()
	if !ok {
		t.Fatal("MeasuredFMAPeakGFLOPS reported no measurement on an AVX2 machine")
	}
	impliedGHz := g / flopsPerCycle
	t.Logf("MEASURED single-core f32 AVX2 FMA ceiling: %.1f GFLOPS", g)
	t.Logf("→ implies %.2f GHz at %.0f f32-flops/cycle (2 FMA pipes × 8 lanes × 2)", impliedGHz, flopsPerCycle)

	if impliedGHz < minPlausibleHz || impliedGHz > maxPlausibleHz {
		t.Errorf("implied clock %.2f GHz is outside [%.1f, %.1f] — this probe is not measuring FMA throughput. "+
			"Too few in-flight accumulators would expose FMA latency (~5 cycles on Zen 2) and read low; "+
			"elided work would read high.", impliedGHz, minPlausibleHz, maxPlausibleHz)
	}
}

// NO CONTENTION TEST SHIPS HERE, AND THE ATTEMPT IS WORTH RECORDING.
//
// measureFMAPeak takes the max of several short runs rather than averaging, because a
// throughput CEILING is a maximum and contention can only push an observed rate down.
// The obvious way to pin that behaviour is to measure under deliberate load and assert
// the figure holds. Two attempts, both wrong:
//
//   - 8 loader goroutines DID discriminate, and destabilised embed's
//     TestEncodeBatch_speedup, which asserts a parallel speedup and runs concurrently
//     in another package. `go test ./...` schedules packages together, so a test that
//     saturates the machine is a test that fails its neighbours.
//   - 2 loader goroutines left the neighbours alone and asserted NOTHING: mutating
//     measureFMAPeak from max to average still reported 136.4 GFLOPS under that load
//     (100% retained), because two threads on 16 logical CPUs do not contend. It would
//     have shipped as a green test that could not fail.
//
// Those requirements conflict on a shared machine, so the estimator's robustness is
// documented at measureFMAPeak rather than asserted here, and TestFMAPeakAMD64_empirical's
// implied-clock check is the gate that actually discriminates. Recorded so the next person
// does not re-derive the same dead end — and because a "verified by mutation" claim that
// was never mutation-checked is exactly the kind of comment this repo should not carry.
