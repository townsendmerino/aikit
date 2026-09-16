package encoder

import "math"

// layerNorm applies y = (x - mean) / sqrt(var + eps) * weight + bias
// per row of x in-place, matching torch.nn.LayerNorm semantics over the
// last (hidden) axis. Mean and variance accumulate in float64 — this
// is the single most parity-sensitive op outside the GEMMs and the
// place where float32 accumulation visibly drifts on long sequences.
//
//	x:      [L, D] row-major, modified in place
//	weight: [D] gain (γ)
//	bias:   [D] shift (β)
//	eps:    1e-12 for CodeRankEmbed (from config.layer_norm_epsilon)
//
// The unbiased=False variance (divisor D, not D-1) matches PyTorch's
// default and is the load-bearing choice.
// Split across cores by ROW (audit M-06). This was the last elementwise stage
// in the encoder running on one goroutine while the linears around it fanned
// out — 25 x L*D of it per forward. The split is EXACTLY bit-identical, and the
// distinction from dead-ends §8.4 matters: that entry rejected a SIMD layernorm
// because vectorising the mean/variance reduction re-associates an f64 sum. A
// row split re-associates nothing — each row's two f64 reductions run in the
// same order on one goroutine, and rows never interact. §8.4's "~0.5% of a
// forward" was also measured against a forward that was serial here.
func layerNorm(x, weight, bias []float32, L, D int, eps float64) {
	parallelRows(L, L*D, func(start, end int) {
		layerNormRows(x, weight, bias, start, end, D, eps)
	})
}

func layerNormRows(x, weight, bias []float32, start, end, D int, eps float64) {
	if D <= 0 {
		return
	}
	_ = weight[D-1]
	_ = bias[D-1]
	for i := start; i < end; i++ {
		row := x[i*D : (i+1)*D]
		_ = row[D-1]
		// mean in f64 (strict sequential accumulation preserving exact f64 sum order)
		var mean float64
		j := 0
		for ; j+3 < D; j += 4 {
			mean += float64(row[j+0])
			mean += float64(row[j+1])
			mean += float64(row[j+2])
			mean += float64(row[j+3])
		}
		for ; j < D; j++ {
			mean += float64(row[j])
		}
		mean /= float64(D)

		// variance in f64 (divisor D — PyTorch default unbiased=False, strict sequential sum)
		var variance float64
		j = 0
		for ; j+3 < D; j += 4 {
			d0 := float64(row[j+0]) - mean
			variance += d0 * d0
			d1 := float64(row[j+1]) - mean
			variance += d1 * d1
			d2 := float64(row[j+2]) - mean
			variance += d2 * d2
			d3 := float64(row[j+3]) - mean
			variance += d3 * d3
		}
		for ; j < D; j++ {
			d := float64(row[j]) - mean
			variance += d * d
		}
		variance /= float64(D)

		invStd := 1.0 / math.Sqrt(variance+eps)
		j = 0
		for ; j+3 < D; j += 4 {
			v0 := row[j+0]
			v1 := row[j+1]
			v2 := row[j+2]
			v3 := row[j+3]
			row[j+0] = float32(((float64(v0)-mean)*invStd)*float64(weight[j+0]) + float64(bias[j+0]))
			row[j+1] = float32(((float64(v1)-mean)*invStd)*float64(weight[j+1]) + float64(bias[j+1]))
			row[j+2] = float32(((float64(v2)-mean)*invStd)*float64(weight[j+2]) + float64(bias[j+2]))
			row[j+3] = float32(((float64(v3)-mean)*invStd)*float64(weight[j+3]) + float64(bias[j+3]))
		}
		for ; j < D; j++ {
			v := row[j]
			row[j] = float32(((float64(v)-mean)*invStd)*float64(weight[j]) + float64(bias[j]))
		}
	}
}
