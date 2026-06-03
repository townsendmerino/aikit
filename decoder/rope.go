package decoder

import "math"

// applyRoPE rotates a single position's projected heads in place. vec is
// [heads, headDim] flattened row-major (a query or key vector for ONE token
// at absolute position pos). base selects the frequency table: Gemma 3 uses
// rope_local_base_freq (10000) on the sliding-window layers and rope_theta
// (1e6) on the global layers, so the caller passes the right one per layer.
//
// NeoX / HF convention (rotate_half, non-interleaved), matching
// encoder/rope.go and HF's apply_rotary_pos_emb:
//
//	x1 = x[:half]; x2 = x[half:]
//	out[:half] = x1*cos - x2*sin
//	out[half:] = x2*cos + x1*sin
//
// where cos/sin use θ_d = pos / base^(2d/headDim) for d ∈ [0, headDim/2).
//
// Frequencies are recomputed per call (cos/sin are cheap relative to the
// matmuls); M7 precomputes per-position tables once the perf backend lands.
func applyRoPE(vec []float32, heads, headDim, pos int, base float64) {
	half := headDim / 2
	posF := float64(pos)
	for d := 0; d < half; d++ {
		invFreq := math.Pow(base, -float64(2*d)/float64(headDim))
		theta := posF * invFreq
		c := math.Cos(theta)
		s := math.Sin(theta)
		for h := 0; h < heads; h++ {
			off := h * headDim
			x1 := float64(vec[off+d])
			x2 := float64(vec[off+half+d])
			vec[off+d] = float32(x1*c - x2*s)
			vec[off+half+d] = float32(x2*c + x1*s)
		}
	}
}
