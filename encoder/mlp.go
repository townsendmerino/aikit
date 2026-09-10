package encoder

import "github.com/townsendmerino/aikit/linalg"

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
	s.mm(h, Fc11, val, L, D, intermediate)
	s.mm(h, Fc12, gate, L, D, intermediate)
	// mid = val ⊙ SiLU(gate), reuse val's storage.
	//
	// Two audit fixes here (M-07, M-06). M-07: this called linalg.SiLUInto,
	// which is vectorized only under GOEXPERIMENT=simd — a build this library
	// cannot require of importers, and one docs/task-archsimd-eval.md ruled out
	// as a shipping path — so in every shipped build it ran the scalar loop. The
	// hand-written NEON/AVX2 kernels live behind the *ContractInto API instead
	// (5.2x/6.6x on SiLU per CHANGELOG v1.37.0), which is what this now calls.
	// M-06: the SiLU and the product were the encoder's only elementwise stages
	// that never got the parallelRows split the linears and GeGLU have, so they
	// ran on one core while everything around them fanned out.
	//
	// Both are numerically inert as parallelism goes — the split is by row and
	// each element is independent. The kernel swap is NOT bit-identical to the
	// old scalar loop: measured max ABSOLUTE deviation 4.8e-07 over 200k normal
	// inputs (2 ULP at unit magnitude). The encoder goldens are cosine >= 0.9999
	// and gate it end to end.
	parallelRows(L, L*intermediate, func(start, end int) {
		lo, hi := start*intermediate, end*intermediate
		g := gate[lo:hi]
		linalg.SiLUContractInto(g, g)
		v := val[lo:hi]
		for i := range v {
			v[i] *= g[i]
		}
	})
	mid := s.mid[:L*D]
	s.mm(val, Fc2, mid, L, intermediate, D)
	for i := range h {
		h[i] += mid[i]
	}
	// Lens §4.2 (column-block gate into val[:, j0:j1], dropping the [L,I] gate buffer
	// to one jb-wide tile) was built and measured out on arm64: latency-neutral on
	// amd64 but +3.5–6.9% to this stage here (the 12× extra parallel fork/joins bite
	// the 6P+2E barrier). A footprint-only win that fails the "free" bar the lens set
	// when it chose column-block over row-tiling — see the perf-campaign note.
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
// (vs. the gated SwiGLU form the v1.5/CodeRankEmbed checkpoints use). tanh
// selects which GELU: false is the exact erf variant (activation_function ==
// "gelu", the reference builds nn.GELU(approximate="none")); true is the tanh
// approximation ("gelu_new"/"gelu_fast"/"gelu_pytorch_tanh", see Config.geluTanh)
// — collapsing the two is a silent model change, not a safe approximation.
func geluMLP(h, fc1, fc1b, fc2, fc2b []float32, D, intermediate, L int, tanh bool, s *scratch) {
	inner := s.val[:L*intermediate] // reuse the SwiGLU value buffer
	s.mm(h, fc1, inner, L, D, intermediate)
	addRowBias(inner, fc1b, L, intermediate)
	if tanh {
		geluTanh(inner)
	} else {
		gelu(inner)
	}

	out := s.mid[:L*D] // reuse the SwiGLU fc2-output buffer
	s.mm(inner, fc2, out, L, intermediate, D)
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
//   - W2 is [intermediate, D] in the reference (act_out.matmul(expert_w2)); the
//     loader stores it TRANSPOSED per-expert to [D, intermediate] (weights.go,
//     audit #10) so the projection runs through the SIMD A·Bᵀ matmul instead of a
//     scalar triple-loop. The per-expert block size is unchanged, so the offset
//     arithmetic below is the same.
//   - `bias` is a single shared [D] row added once after combining the experts —
//     it is not per-expert.
//
// The softmax runs over all experts (not just the top-k) and the top-k weights are
// taken from it as-is, so they do not sum to 1.
//
// All working buffers come from the scratch arena: moeScores/moeOut are dedicated,
// x1 and contrib reuse val/mid (as gelu/swigluMLP do), so the MoE layers allocate
// nothing after warmup.
func moeMLP(h, router, w1, w2t, bias []float32, numExperts, topK, D, intermediate, L int, tanh bool, s *scratch) {
	s.moeAllScores = ensureF32(s.moeAllScores, L*numExperts)
	s.moeWeight = ensureF32(s.moeWeight, L*topK)
	s.moeGathered = ensureF32(s.moeGathered, L*D)
	s.moeAcc = ensureF32(s.moeAcc, L*D)
	s.moeExpert = ensureI32(s.moeExpert, L*topK)
	s.moeTokens = ensureI32(s.moeTokens, L)
	scores := s.moeAllScores[:L*numExperts]
	acc := s.moeAcc[:L*D]
	clear(acc)

	// 1. Route every token. One [L,numExperts] matmul instead of L single-row
	// ones; linalg's M-invariance makes it bit-identical to the per-token form.
	s.mm(h, router, scores, L, D, numExperts)
	for t := range L {
		row := scores[t*numExperts : (t+1)*numExperts]
		softmaxRow(row)
		for r := range topK {
			// Pick the current argmax, then mask it so the next pass finds the
			// runner-up (topK is 2 here; a full sort would be wasted work).
			best, bestIdx := float32(-1), -1
			for e, sc := range row {
				if sc > best {
					best, bestIdx = sc, e
				}
			}
			if bestIdx < 0 {
				s.moeExpert[t*topK+r] = -1
				continue
			}
			row[bestIdx] = -1 // consumed
			s.moeExpert[t*topK+r] = int32(bestIdx)
			s.moeWeight[t*topK+r] = best
		}
	}

	// 2. One pass per RANK, and within it one GEMM per expert over all the
	// tokens that chose it (perf-campaign item 33). Both projections used to run
	// at M=1, once per (token, rank): 2048 single-row GEMM calls per MoE layer at
	// L=512, which profiled at 47% of the whole forward.
	//
	// Bit-identical, on two separate grounds:
	//   - linalg guarantees M-INVARIANCE — dst[i,j] does not depend on M — so
	//     batching a token's projection with others produces the same bits as
	//     computing it alone (TestMatmulBT_MConsistent).
	//   - the rank loop is OUTERMOST, so each token still accumulates its rank-0
	//     contribution before its rank-1 one, in the same order as before.
	//     Grouping by expert across ranks would reorder that sum, and float
	//     addition is not associative.
	for r := range topK {
		for e := range numExperts {
			// Gather this expert's tokens for this rank.
			m := 0
			for t := range L {
				if int(s.moeExpert[t*topK+r]) == e {
					s.moeTokens[m] = int32(t)
					copy(s.moeGathered[m*D:(m+1)*D], h[t*D:(t+1)*D])
					m++
				}
			}
			if m == 0 {
				continue
			}
			in := s.moeGathered[:m*D]
			x1 := s.val[:m*intermediate]
			contrib := s.mid[:m*D]

			off := e * intermediate
			expW1 := w1[off*D : (off+intermediate)*D]
			// w2t holds this expert's W2 transposed to [D, intermediate], so
			// contrib_j = Σ_i x1_i·W2[i,j] = x1·W2ᵀ.
			expW2t := w2t[off*D : (off+intermediate)*D]

			matmulBTBlockedInto(in, expW1, x1, m, D, intermediate)
			if tanh {
				geluTanh(x1)
			} else {
				gelu(x1)
			}
			matmulBTBlockedInto(x1, expW2t, contrib, m, intermediate, D)

			for i := range m {
				t := int(s.moeTokens[i])
				w := s.moeWeight[t*topK+r]
				dst := acc[t*D : (t+1)*D]
				src := contrib[i*D : (i+1)*D]
				for j := range D {
					dst[j] += w * src[j]
				}
			}
		}
	}

	// 3. Residual + the single shared bias, once per token.
	for t := range L {
		row := h[t*D : (t+1)*D]
		a := acc[t*D : (t+1)*D]
		for j := range D {
			row[j] += a[j] + bias[j]
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
			w.Cfg.NumExperts, w.Cfg.MoETopK, D, intermediate, L, w.Cfg.geluTanh(), s)
	case l.Fc11 != nil:
		swigluMLP(h, l.Fc11, l.Fc12, l.Fc2, D, intermediate, L, s)
	default:
		geluMLP(h, l.Fc1, l.Fc1B, l.Fc2, l.Fc2B, D, intermediate, L, w.Cfg.geluTanh(), s)
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
