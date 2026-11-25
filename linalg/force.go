package linalg

import "github.com/townsendmerino/aikit/internal/cpufeat"

// ForcedFallbacks returns the CPU-feature fallbacks this build forces, as the tag names without the "aikit_" prefix ("noavx2", "noavx512", ...), or nil in a
// normal build. It never reports a feature the hardware really lacks, only one a build tag turned off. The tags and the flags they force are in
// internal/cpufeat's package doc.
func ForcedFallbacks() []string { return cpufeat.Forced() }
