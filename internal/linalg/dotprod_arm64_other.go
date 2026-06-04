//go:build arm64 && !linux && !darwin

package linalg

// detectDotProd: no portable HWCAP probe on this OS, so conservatively assume no
// DotProd and use the base-ISA SMULL/SADALP kernel.
func detectDotProd() bool { return false }
