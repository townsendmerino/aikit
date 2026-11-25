package linalg

// MeasuredFMAPeakGFLOPS returns this core's achievable single-core f32 FMA ceiling,
// measured on the spot by a register-saturating loop with no memory traffic, and
// reports whether a measurement was possible.
//
// It exists so "fraction of peak" is grounded in what this machine does rather than
// in a constant. BenchmarkGEMMPeakFraction previously divided by a hardcoded M1 Pro
// figure (3.2 GHz × 16 f32-FMA/cyc), which it also applied on amd64 — reporting a
// Zen 2 GEMM at "~50 %peak" when the measured figure is ~38%. A denominator from the
// wrong machine produces a plausible number, and a plausible wrong number is harder
// to catch than an obviously wrong one.
//
// ok is false where no probe exists for the architecture, or where the CPU lacks the
// required ISA. Callers must report GFLOPS alone in that case rather than inventing a
// fraction.
func measuredFMAPeakGFLOPS() (gflops float64, ok bool) { return measureFMAPeak() }
