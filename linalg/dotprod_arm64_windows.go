//go:build arm64 && windows

package linalg

import "syscall"

// pfARMV82DPInstructionsAvailable is PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE, the IsProcessorFeaturePresent feature number Windows documents for the ARMv8.2 Dot Product instructions (SDOT/UDOT).
const pfARMV82DPInstructionsAvailable = 43

var procIsProcessorFeaturePresent = syscall.NewLazyDLL("kernel32.dll").NewProc("IsProcessorFeaturePresent")

// detectDotProd asks Windows whether this CPU implements DotProd. Windows is the one OS where the answer comes from an API rather than HWCAP or the platform: there is no auxv, and Windows on ARM
// ships on cores both with it (Snapdragon X, Azure Cobalt 100) and, in principle, without. An older Windows that does not know feature 43 returns FALSE, which is the safe direction: the base
// SMULL/SADALP kernels. A false "yes" would be the dangerous one (SIGILL on the first SDOT), which is why this trusts the OS's own answer and nothing inferred.
func detectDotProd() bool {
	r, _, _ := procIsProcessorFeaturePresent.Call(uintptr(pfARMV82DPInstructionsAvailable))
	return r != 0
}
