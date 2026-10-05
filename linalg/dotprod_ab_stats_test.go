package linalg

import (
	"sort"
)

// The statistics and the pre-registered verdict rule for the DotProd-versus-base A/B (dotprod_ab_arm64_test.go), kept in a file with no build tag so the RULE is unit-tested on every platform
// and cannot drift from what the pre-registration says (docs/measurements/dotprod-windows-arm-2026-10-04.md in goinfer). A cell is one (quant, model, projection, M, workers). Its input is the per-round
// speedup of the DotProd path over the base kernels, t_base / t_dot, from rounds that alternate which arm runs first.

type abCell struct {
	Median, Q1, Q3 float64 // of the per-round speedups
	FracFaster     float64 // share of rounds with speedup > 1
	MedianFwd      float64 // median over rounds where the base arm ran first
	MedianRev      float64 // median over rounds where the DotProd arm ran first
	Rounds         int
}

type abVerdict string

const (
	abFaster    abVerdict = "FASTER"
	abSlower    abVerdict = "SLOWER"
	abAmbiguous abVerdict = "AMBIGUOUS"
	abNoisy     abVerdict = "AMBIGUOUS (noisy)"
)

// Pre-registered bands, fixed before any run.
const (
	abFasterMedian   = 1.05 // a cell is FASTER only at a median speedup of at least this ...
	abFasterFrac     = 0.80 // ... with at least this share of rounds above 1 ...
	abFasterOrder    = 1.03 // ... and BOTH order-halves' medians at least this (an arm that only wins going second is a cache effect, not a kernel)
	abSlowerMedian   = 0.95 // SLOWER mirrors it
	abSlowerFrac     = 0.20
	abSlowerOrder    = 0.97
	abNoisyIQRRatio  = 0.15 // (Q3-Q1)/median above this: the cell is too noisy to call either way
	abClaimFasterMin = 0.75 // the overall claim "faster on this core" needs this share of the M=1 cells FASTER and no cell SLOWER
)

func abMedian(xs []float64) float64 { return abQuantile(xs, 0.5) }

func abQuantile(xs []float64, q float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	pos := q * float64(len(s)-1)
	lo := int(pos)
	if lo+1 >= len(s) {
		return s[len(s)-1]
	}
	frac := pos - float64(lo)
	return s[lo]*(1-frac) + s[lo+1]*frac
}

// abSummarize turns the per-round speedups into a cell. baseFirst[i] says whether round i ran the base arm first.
func abSummarize(speedups []float64, baseFirst []bool) abCell {
	var fwd, rev []float64
	faster := 0
	for i, s := range speedups {
		if s > 1 {
			faster++
		}
		if baseFirst[i] {
			fwd = append(fwd, s)
		} else {
			rev = append(rev, s)
		}
	}
	c := abCell{Rounds: len(speedups), Median: abMedian(speedups), Q1: abQuantile(speedups, 0.25), Q3: abQuantile(speedups, 0.75), MedianFwd: abMedian(fwd), MedianRev: abMedian(rev)}
	if len(speedups) > 0 {
		c.FracFaster = float64(faster) / float64(len(speedups))
	}
	return c
}

func abClassify(c abCell) abVerdict {
	if c.Median > 0 && (c.Q3-c.Q1)/c.Median > abNoisyIQRRatio {
		return abNoisy
	}
	switch {
	case c.Median >= abFasterMedian && c.FracFaster >= abFasterFrac && c.MedianFwd >= abFasterOrder && c.MedianRev >= abFasterOrder:
		return abFaster
	case c.Median <= abSlowerMedian && c.FracFaster <= abSlowerFrac && c.MedianFwd <= abSlowerOrder && c.MedianRev <= abSlowerOrder:
		return abSlower
	}
	return abAmbiguous
}

// abClaim applies the overall rule to the M=1 (decode) cells' verdicts: "DotProd is faster on this core" holds only if enough cells are FASTER and none is SLOWER. A SLOWER cell is reported by name either way.
func abClaim(verdicts []abVerdict) (claim bool, faster, slower int) {
	for _, v := range verdicts {
		switch v {
		case abFaster:
			faster++
		case abSlower:
			slower++
		}
	}
	return len(verdicts) > 0 && slower == 0 && float64(faster)/float64(len(verdicts)) >= abClaimFasterMin, faster, slower
}
