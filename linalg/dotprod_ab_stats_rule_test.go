package linalg

import "testing"

func TestDotProdAB_verdictRule(t *testing.T) {
	mk := func(med, iqr, frac, fwd, rev float64) abCell {
		return abCell{Median: med, Q1: med - iqr/2, Q3: med + iqr/2, FracFaster: frac, MedianFwd: fwd, MedianRev: rev, Rounds: 21}
	}
	for _, tc := range []struct {
		name string
		c    abCell
		want abVerdict
	}{
		{"clearly faster", mk(1.8, 0.05, 1, 1.8, 1.8), abFaster},
		{"just inside the faster band", mk(1.05, 0.02, 0.81, 1.03, 1.03), abFaster},
		{"median just under the band", mk(1.049, 0.02, 1, 1.05, 1.05), abAmbiguous},
		{"fraction just under", mk(1.5, 0.05, 0.79, 1.5, 1.5), abAmbiguous},
		{"wins only going second (a cache effect)", mk(1.2, 0.05, 0.9, 0.99, 1.4), abAmbiguous},
		{"clearly slower", mk(0.7, 0.05, 0, 0.7, 0.7), abSlower},
		{"slower only going second is not slower", mk(0.9, 0.05, 0.1, 1.0, 0.8), abAmbiguous},
		{"flat", mk(1.0, 0.03, 0.5, 1.0, 1.0), abAmbiguous},
		{"too noisy to call, even with a big median", mk(1.8, 0.4, 1, 1.8, 1.8), abNoisy},
	} {
		if got := abClassify(tc.c); got != tc.want {
			t.Errorf("%s: %v classified %s, want %s", tc.name, tc.c, got, tc.want)
		}
	}
	// The summary: alternating order halves are split correctly and the fraction counts rounds above 1.
	sp := []float64{2, 1.9, 2.1, 0.9}
	bf := []bool{true, false, true, false}
	c := abSummarize(sp, bf)
	if c.MedianFwd != 2.05 || c.MedianRev != 1.4 || c.FracFaster != 0.75 || c.Rounds != 4 {
		t.Errorf("abSummarize = %+v", c)
	}
}

func TestDotProdAB_claimRule(t *testing.T) {
	f, s, a := abFaster, abSlower, abAmbiguous
	for _, tc := range []struct {
		name string
		vs   []abVerdict
		want bool
	}{
		{"all faster", []abVerdict{f, f, f, f}, true},
		{"exactly 75% faster", []abVerdict{f, f, f, a}, true},
		{"under 75%", []abVerdict{f, f, a, a}, false},
		{"one slower vetoes the claim", []abVerdict{f, f, f, f, f, f, f, s}, false},
		{"nothing to judge", nil, false},
	} {
		if got, _, _ := abClaim(tc.vs); got != tc.want {
			t.Errorf("%s: claim = %v, want %v", tc.name, got, tc.want)
		}
	}
}
