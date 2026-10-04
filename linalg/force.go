package linalg

// Forced CPU-feature fallbacks, for CI and test runs (goinfer's hardware-coverage H1.3).
//
// A dispatcher here picks its kernel from a package-level flag detected once at init (hasAVX2, hasAVX512VNNI, hasDotProd, ...), so on a machine that has the
// feature the portable or narrower-ISA kernels never run. Building with one of these tags makes the flag report false, so the SAME suites execute the
// fallback path on any machine:
//
//	aikit_noavx512   amd64: hasAVX512VNNI and hasAVX512VNNIVL false (the AVX2 kernels on an AVX-512 VNNI machine)
//	aikit_noavx2     amd64: hasAVX2, hasQ4KAVX2 and (implied) both AVX-512 flags false (the pure-Go kernels)
//	aikit_nopopcnt   amd64: hasPOPCNT false (the portable Hamming kernel)
//	aikit_nodotprod  arm64: hasDotProd false (the base SMULL/SADALP kernels instead of SDOT)
//
// None of these tags is for production: they only ever turn a fast path OFF. They exist so the narrower path is executed on every push rather than only on
// hardware nobody here owns. ForcedFallbacks reports which tags are in effect, so a CI job that passes `-tags` can assert the tag was honoured instead of passing
// silently on a tag that matched no file (a green run that forced nothing).

// forcedFallbacks is appended to by the tagged init functions; empty in a normal build.
var forcedFallbacks []string

// ForcedFallbacks returns the CPU-feature fallbacks this build forces, as the tag names without the "aikit_" prefix ("noavx2", "noavx512", ...), or nil in a
// normal build. It never reports a feature the hardware really lacks, only one a build tag turned off.
func ForcedFallbacks() []string {
	if len(forcedFallbacks) == 0 {
		return nil
	}
	return append([]string(nil), forcedFallbacks...)
}
