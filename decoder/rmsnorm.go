package decoder

import "math"

// rmsNorm applies RMSNorm in place over each row of x ([rows, dim]).
//
//	rms      = sqrt(mean(x²) + eps)
//	x[i]     = (x[i] / rms) * scale[i]
//
// addOne selects the scale: Gemma stores weights as deviations from 1.0 and
// scales by (1 + weight); Llama/Qwen scale by weight directly. Using the wrong
// one silently zeroes/shifts every layer — it's a per-family knob
// (Architecture.RMSAddOne), one of the package's carry-over invariants (doc.go).
//
// Accumulate the sum of squares in float64 — the dim can be small (640) but
// the f32 round-off still matters at the ≥1−1e-4 parity bar, mirroring the
// float64-accumulation discipline embed/ and encoder/ already rely on.
func rmsNorm(x, weight []float32, rows, dim int, eps float64, addOne bool) {
	for r := range rows {
		row := x[r*dim : r*dim+dim]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		inv := float32(1.0 / math.Sqrt(ss/float64(dim)+eps))
		if addOne {
			for i, v := range row {
				row[i] = (v * inv) * (1 + weight[i])
			}
		} else {
			for i, v := range row {
				row[i] = (v * inv) * weight[i]
			}
		}
	}
}

// silu is x·sigmoid(x), the SwiGLU activation Llama/Mistral/Qwen use (computed
// in float64 like geluTanh for parity). Gemma uses geluTanh; this is here for
// the SwiGLU families (multi-model-plan G2).
func silu(x float32) float32 {
	x64 := float64(x)
	return float32(x64 / (1 + math.Exp(-x64)))
}

// geluTanh is the tanh-approximate GELU Gemma's GeGLU MLP uses
// ("gelu_pytorch_tanh"). Provided here so mlp.go (stub) and tests have the
// activation ready.
//
//	0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 x³) ))
func geluTanh(x float32) float32 {
	const c = 0.7978845608028654 // sqrt(2/π)
	x64 := float64(x)
	inner := c * (x64 + 0.044715*x64*x64*x64)
	return float32(0.5 * x64 * (1 + math.Tanh(inner)))
}
