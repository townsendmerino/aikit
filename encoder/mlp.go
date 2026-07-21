package encoder

// swigluMLP runs one block's SwiGLU MLP and adds the output to the
// residual `h` (in place). Caller applies the post-MLP LayerNorm.
//
// SwiGLU is (per the reference NomciBertGatedMLP.forward — the
// variable naming in the reference is confusingly inverted vs. typical
// SwiGLU papers):
//
//	val  = h · Fc11ᵀ                  // [L, intermediate]   <- unmodified
//	gate = h · Fc12ᵀ                  // [L, intermediate]   <- activated
//	mid  = val ⊙ SiLU(gate)           // [L, intermediate]
//	out  = mid · Fc2ᵀ                 // [L, D]
//
// CodeRankEmbed's checkpoint stores fc11 and fc12 as TWO separate
// tensors — the plan §6.4 footnote noted "confirm the split convention
// and fc1 width from the checkpoint shapes"; the M1 dump confirmed
// there is no fused fc1, so this is two matmuls (not one + split), and
// the SiLU goes on fc12 (not fc11). The first cut of this file had the
// gate/value naming swapped — caught by the smoke cosine going to
// -0.06; the reference's NomciBertGatedMLP.forward is the source of
// truth, not the SwiGLU paper.
//
// No biases on any of fc11/fc12/fc2 (config: mlp_fc1_bias=false,
// mlp_fc2_bias=false).
//
// M8c: takes a *scratch arena so val / gate / mid buffers are reused
// across the 12 layers per forward.
func swigluMLP(h []float32, Fc11, Fc12, Fc2 []float32, D, intermediate, L int, s *scratch) {
	val := s.val[:L*intermediate]
	gate := s.gate[:L*intermediate]
	matmulBTInto(h, Fc11, val, L, D, intermediate)
	matmulBTInto(h, Fc12, gate, L, D, intermediate)
	// mid = val ⊙ SiLU(gate), reuse val's storage
	for i, v := range val {
		val[i] = v * silu(gate[i])
	}
	mid := s.mid[:L*D]
	matmulBTInto(val, Fc2, mid, L, intermediate, D)
	for i := range h {
		h[i] += mid[i]
	}
}

// addRowBias adds a [N] bias row to every row of a [M, N] matrix, in place.
// A nil bias is a no-op, which is how the bias-free Nomic checkpoints
// (CodeRankEmbed, nomic-embed-text-v1.5) keep their exact previous arithmetic.
func addRowBias(dst, bias []float32, M, N int) {
	if bias == nil {
		return
	}
	for i := range M {
		row := dst[i*N : (i+1)*N]
		for j := range N {
			row[j] += bias[j]
		}
	}
}

// geluMLP runs one block's plain two-matrix GELU MLP and adds the output to the
// residual `h` (in place); the caller applies the post-MLP LayerNorm. This is the
// NomicBertMLP form used by the DENSE layers of nomic-embed-text-v2-moe:
//
//	y = GELU(h · Fc1ᵀ + b1) · Fc2ᵀ + b2
//
// (vs. the gated SwiGLU form the v1.5/CodeRankEmbed checkpoints use). GELU is the
// exact erf variant — the reference builds nn.GELU(approximate="none") for
// activation_function == "gelu".
func geluMLP(h, fc1, fc1b, fc2, fc2b []float32, D, intermediate, L int, s *scratch) {
	inner := s.val[:L*intermediate] // reuse the SwiGLU value buffer
	matmulBTInto(h, fc1, inner, L, D, intermediate)
	addRowBias(inner, fc1b, L, intermediate)
	gelu(inner)

	out := s.mid[:L*D] // reuse the SwiGLU fc2-output buffer
	matmulBTInto(inner, fc2, out, L, intermediate, D)
	addRowBias(out, fc2b, L, D)
	for i := range h {
		h[i] += out[i]
	}
}

// moeMLP runs one block's mixture-of-experts FFN and adds the output to the
// residual `h` (in place). Mirrors the reference NomicRouter + NomicExperts
// (modeling_hf_nomic_bert.py) exactly:
//
//	scores  = softmax(h · Routerᵀ)         over ALL experts, in float32
//	(w,e)   = topk(scores, topK)           weights NOT renormalized
//	                                       (moe_normalize_expert_weights=false)
//	y_token = Σ_k w_k · ( GELU(h · W1_{e_k}ᵀ) · W2_{e_k} ) + bias
//
// Two details are easy to get subtly wrong and are load-bearing:
//   - W2 is applied WITHOUT transpose (the reference does act_out.matmul(expert_w2)
//     where expert_w2 is [intermediate, D]), unlike every other projection here.
//   - `bias` is a single shared [D] row added once after combining the experts —
//     it is not per-expert.
//
// The softmax runs over all experts (not just the top-k) and the top-k weights are
// taken from it as-is, so they do not sum to 1.
func moeMLP(h, router, w1, w2, bias []float32, numExperts, topK, D, intermediate, L int, s *scratch) {
	scores := make([]float32, numExperts)
	x1 := make([]float32, intermediate)
	out := make([]float32, D)

	for t := range L {
		row := h[t*D : (t+1)*D]

		// Router: scores over all experts, softmaxed in float32.
		matmulBTInto(row, router, scores, 1, D, numExperts)
		softmaxRow(scores)

		clear(out)
		for range topK {
			// Pick the current argmax, then mask it so the next pass finds the
			// runner-up (topK is 2 here; a full sort would be wasted work).
			best, bestIdx := float32(-1), -1
			for e, sc := range scores {
				if sc > best {
					best, bestIdx = sc, e
				}
			}
			if bestIdx < 0 {
				break
			}
			scores[bestIdx] = -1 // consumed

			off := bestIdx * intermediate
			expW1 := w1[off*D : (off+intermediate)*D]
			expW2 := w2[off*D : (off+intermediate)*D]

			matmulBTInto(row, expW1, x1, 1, D, intermediate)
			gelu(x1)
			// W2 is [intermediate, D] applied untransposed: out_j = Σ_i x1_i·W2[i,j].
			for i := range intermediate {
				wRow := expW2[i*D : (i+1)*D]
				xi := x1[i]
				if xi == 0 {
					continue
				}
				for j := range D {
					out[j] += best * xi * wRow[j]
				}
			}
		}
		for j := range D {
			row[j] += out[j] + bias[j]
		}
	}
}

// applyMLP runs whichever FFN variant layer l declares — MoE, gated SwiGLU, or
// dense GELU — adding the result to the residual h in place. Keeping the choice
// in one place means the four forward variants can't drift apart.
func applyMLP(w *Weights, l *LayerWeights, h []float32, D, intermediate, L int, s *scratch) {
	switch {
	case l.IsMoE:
		moeMLP(h, l.Router, l.ExpW1, l.ExpW2, l.ExpBias,
			w.Cfg.NumExperts, w.Cfg.MoETopK, D, intermediate, L, s)
	case l.Fc11 != nil:
		swigluMLP(h, l.Fc11, l.Fc12, l.Fc2, D, intermediate, L, s)
	default:
		geluMLP(h, l.Fc1, l.Fc1B, l.Fc2, l.Fc2B, D, intermediate, L, s)
	}
}

// hasMoE reports whether any layer uses the MoE FFN.
func (w *Weights) hasMoE() bool {
	for i := range w.Layers {
		if w.Layers[i].IsMoE {
			return true
		}
	}
	return false
}
