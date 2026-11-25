//go:build arm64 && !linux && !darwin && !windows

package cpufeat

// DetectDotProd: no portable HWCAP probe on this OS (Windows has its own file), so conservatively assume no
// DotProd and use the base-ISA SMULL/SADALP kernel.
func DetectDotProd() bool { return false }
