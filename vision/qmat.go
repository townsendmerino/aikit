package vision

import "github.com/townsendmerino/aikit/linalg"

// newQMat wraps an f32 [rows, cols] matmul weight as a linalg.WeightMat, holding
// it either as f32 or as per-row int8 (W8A8) when quant is set. Quantizing once at
// load runs the integer W8A8 kernel — ~2–4× the f32 throughput on the matmul-bound
// SigLIP tower, at the cost of cosine ~0.999 vs the f32 path's 1.0.
//
// This function is the tower's storage POLICY — which weights quantize, and that
// f32 is copied. The precision dispatch itself is WeightMat's job: vision used to
// open-code its own `qmat` wrapper (one of the three WeightMat was written to
// consolidate, alongside encoder.LayerWeightsQ8 and goinfer's decoder.weightMat),
// and this is that migration. Weights dispatch through WeightMat.MatmulBT into the
// same linalg kernels as before, so the numerics are unchanged.
//
// Attention/FFN projections quantize under -vision-quant; the patch-embed conv
// deliberately does not — quant error on the input embedding would propagate
// through every layer.
//
// The source slice is never retained: the f32 path copies (WrapF32 aliases, and
// the caller's tensor is an mmap released after load), and the int8 path consumes
// w into fresh codes + scales.
func newQMat(w []float32, rows, cols int, quant bool) linalg.WeightMat {
	if quant {
		// w8a8=true selects MatmulBTW8A8 — the kernel this tower already used.
		return linalg.QuantizeInt8(w, rows, cols, true)
	}
	return linalg.WrapF32(append([]float32(nil), w...), rows, cols)
}
