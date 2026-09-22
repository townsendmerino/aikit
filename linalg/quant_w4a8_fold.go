package linalg

// The S-05 fold kernel's A/B toggle (docs/task-simd-audit.md S-05), declared PORTABLY
// so a consumer's test can flip it from an untagged file on any architecture. The kernel
// it selects is arm64-only (quant_w4a8_fold_arm64.go / matmul_w4a8_row4_arm64.go); on
// every other architecture the toggle is inert — set, readable, consulted by nothing.
// v1.47.0 declared it inside the arm64 file, which made goinfer's untagged A/B test fail
// to compile on linux/amd64 (goinfer run 35763888743): an exported symbol that exists on
// one architecture is a defect even in the Experimental tier.

// w4a8RowFold selects dotW4A8SplitHalf4RowFold for MatmulBTW4A8Row4Into's M=1 path
// (true, the default) or the pre-S-05 dotW4A8SplitHalf4Row (false). The two are
// bit-identical for every input (TestDotW4A8SplitHalf4RowFold_*), so this is a
// performance switch kept for in-process A/B measurement. Not goroutine-safe to flip
// mid-matmul; a benchmark sets it between calls.
var w4a8RowFold = true

// SetW4A8RowFold switches the arm64 M=1 row4 W4A8 path between the S-05 fold kernel
// (true, the default) and the pre-S-05 kernel (false). Numerically inert — both kernels
// produce identical bits — so this exists only for same-process A/B measurement
// (docs/internal/measuring-performance.md rule 3). A no-op on non-arm64.
func SetW4A8RowFold(on bool) { w4a8RowFold = on }

// W4A8RowFold reports whether the S-05 fold kernel is selected (see SetW4A8RowFold).
func W4A8RowFold() bool { return w4a8RowFold }
