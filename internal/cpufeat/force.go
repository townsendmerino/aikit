// Package cpufeat holds the CPU-feature flags every aikit kernel package dispatches on, their detection, and the
// build tags that force them off. The flags are shared state: SelfCheck's repair (DisableTopTier) and the aikit_no*
// tags turn a tier off by writing a flag here, so a kernel package that kept its own copy would keep running the tier.
// Each flag is set once at init and written afterwards only by DisableTopTier and by tests.
//
// Forced CPU-feature fallbacks, for CI and test runs. A dispatcher picks its kernel from one of these flags, so on a
// machine that has the feature the portable or narrower-ISA kernels never run. Building with one of these tags makes the
// flag report false, so the same suites execute the fallback path on any machine:
//
//	aikit_noavx512   amd64: HasAVX512VNNI and HasAVX512VNNIVL false (the AVX2 kernels on an AVX-512 VNNI machine)
//	aikit_noavx2     amd64: HasAVX2, HasQ4KAVX2 and (implied) both AVX-512 flags false (the pure-Go kernels)
//	aikit_nopopcnt   amd64: HasPOPCNT false (the portable Hamming kernel)
//	aikit_nodotprod  arm64: HasDotProd false (the base SMULL/SADALP kernels instead of SDOT)
//
// None of these tags is for production: they only ever turn a fast path off. Forced reports which are in effect, so a CI
// job that passes -tags can assert the tag was honoured instead of passing on a tag that matched no file.
package cpufeat

// forced is appended to by the tagged init functions; empty in a normal build.
var forced []string

// Forced returns the fallbacks this build forces, as the tag names without the "aikit_" prefix ("noavx2", "noavx512",
// ...), or nil in a normal build. It never reports a feature the hardware really lacks, only one a build tag turned off.
func Forced() []string {
	if len(forced) == 0 {
		return nil
	}
	return append([]string(nil), forced...)
}
