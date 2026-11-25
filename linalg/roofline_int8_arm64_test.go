//go:build arm64

package linalg

import "time"

// int8PeakGMACs is the SDOT issue probe as an int8 MAC rate (one SDOT is 16 MACs), the best of
// five passes. ok is false on a core without DotProd.
func int8PeakGMACs() (gmacs float64, ok bool) {
	if !hasDotProd {
		return 0, false
	}
	const iters = 4_000_000
	best := 0.0
	for range 5 {
		t0 := time.Now()
		sdotIssuePeak(iters)
		if rate := float64(iters) * sdotPerIter * 16 / float64(time.Since(t0).Nanoseconds()); rate > best {
			best = rate
		}
	}
	return best, true
}
