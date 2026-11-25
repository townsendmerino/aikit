//go:build arm64

package linalg

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// m1ProPCoreGHz is the M1 Pro P-core's maximum clock, the same constant
// docs/internal/roofline-2026-08.md uses for its 102.4 GFLOPS FMLA ceiling
// (3.2 GHz x 16 f32-FMA/cycle). Ceilings derived from it are what make a reading
// believable or not; the roofline record's whole lesson is that a probe without a
// ceiling assert is not a probe.
const m1ProPCoreGHz = 3.2

// TestSDOTIssuePeak measures SDOT issue width on one P-core and settles
// docs/task-simd-audit.md Appendix B's "four pipes or two" item.
//
// It asserts only the CEILING, never a floor: a result above four SDOTs per cycle
// would mean the loop was not doing what it claims (folded, elided, or the clock
// constant is wrong), which is the failure mode that produced a 135 GB/s
// single-thread bandwidth reading and an 8145 GB/s DtoD reading before it was
// caught. A low result on a busy box is not a failure and must not fail CI, so
// the reading is logged, not gated.
func TestSDOTIssuePeak(t *testing.T) {
	harnessOnly(t)
	if !cpufeat.HasDotProd {
		t.Skip("no FEAT_DotProd on this core")
	}
	const iters = 4_000_000
	var best float64
	for pass := range 5 {
		t0 := time.Now()
		sink := sdotIssuePeak(iters)
		el := time.Since(t0)
		rate := float64(iters) * sdotPerIter / el.Seconds()
		if rate > best {
			best = rate
		}
		fmt.Fprintf(os.Stderr, "[sdot-peak] pass %d/5 %.2f G SDOT/s (%.2f per cycle at %.1f GHz), sink=%d\n",
			pass+1, rate/1e9, rate/1e9/m1ProPCoreGHz, m1ProPCoreGHz, sink)
	}
	perCycle := best / 1e9 / m1ProPCoreGHz
	t.Logf("SDOT issue peak: %.2f G SDOT/s = %.2f per cycle at %.1f GHz "+
		"(= %.1f GMAC/s of int8, since one SDOT is 16 MACs)",
		best/1e9, perCycle, m1ProPCoreGHz, best*16/1e9)
	for _, p := range []int{2, 3, 4} {
		t.Logf("  %d-pipe ceiling: %.1f G SDOT/s = %.1f GMAC/s — %s",
			p, float64(p)*m1ProPCoreGHz, float64(p)*m1ProPCoreGHz*16,
			map[bool]string{true: "EXCEEDED, so SDOT uses more than this", false: "not reached"}[best/1e9 > float64(p)*m1ProPCoreGHz*1.02])
	}
	if perCycle > 4.3 {
		t.Fatalf("%.2f SDOT/cycle exceeds the 4-pipe ceiling — the probe is measuring "+
			"something other than SDOT issue (folded loop, wrong clock constant, or a "+
			"miscounted sdotPerIter). A rate above the hardware ceiling is never a result.", perCycle)
	}
}
