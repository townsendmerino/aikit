//go:build arm64

package cpufeat

// HasDotProd records, once at init, whether this CPU implements the ARMv8.2
// DotProd extension — so dotI8 picks the SDOT kernel on hardware that has it
// (Apple Silicon, Graviton2+, Neoverse, recent Cortex-A) and falls back to the
// base SMULL/SADALP kernel everywhere else. See DetectDotProd (per-OS).
var HasDotProd = DetectDotProd()
