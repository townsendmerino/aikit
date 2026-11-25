//go:build !arm64

package linalg

// int8PeakGMACs reports no int8 dot peak: no probe exists for this arch. tools/roofline takes a
// cited figure instead (--int8-roof).
func int8PeakGMACs() (float64, bool) { return 0, false }
