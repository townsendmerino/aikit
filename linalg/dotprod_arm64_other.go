//go:build arm64 && !linux && !darwin && !windows

package linalg

// detectDotProd: no portable HWCAP probe on this OS (Windows has its own file), so conservatively assume no
// DotProd and use the base-ISA SMULL/SADALP kernel.
func detectDotProd() bool { return false }
