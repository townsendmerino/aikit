package decoder

import "math"

// rmsNorm applies Gemma's RMSNorm in place over each row of x ([rows, dim]).
//
// Gemma's variant scales by (1 + weight), NOT weight — the weights are
// stored as deviations from 1.0. Using weight directly silently zeroes the
// activations on a freshly-initialized norm and shifts every layer
// otherwise. This is one of the package's carry-over invariants (doc.go).
//
//	rms      = sqrt(mean(x²) + eps)
//	x[i]     = (x[i] / rms) * (1 + weight[i])
//
// Accumulate the sum of squares in float64 — the dim can be small (640) but
// the f32 round-off still matters at the ≥1−1e-4 parity bar, mirroring the
// float64-accumulation discipline embed/ and encoder/ already rely on.
func rmsNorm(x, weight []float32, rows, dim int, eps float64) {
	for r := 0; r < rows; r++ {
		row := x[r*dim : r*dim+dim]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		inv := float32(1.0 / math.Sqrt(ss/float64(dim)+eps))
		for i, v := range row {
			row[i] = (v * inv) * (1 + weight[i])
		}
	}
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
