package encoder

import "math"

// selfAttentionBatched is the M7 batched variant of selfAttention.
// Layout: h is [B, Lmax, D] flattened row-major to [B*Lmax, D]. The
// linear projections (Wqkv, OutProj) run as one big matmul over
// M=B*Lmax — that's the M7 win on the bandwidth-bound kernels.
// Attention itself stays per-sequence-per-head; realLen[b] bounds the
// inner softmax + ctx loops so padded positions never enter the score
// matrix (no mask tensor needed).
//
// M8c: takes a *scratch sized for B*Lmax rows. qkv/Q/K/V/ctx/out
// reuse scratch buffers across the 12 layers. Per-head qH/kH/vH and
// per-(b,head) scores ARE still allocated inside this function — they
// need different sizes per sequence (realLen[b] varies), and the
// scratch buffers are sized for B*Lmax which is too big. The per-
// (b,head) allocations stay; the BIG allocations (qkv at 3*B*Lmax*D,
// Q/K/V/ctx at B*Lmax*D each, out at B*Lmax*D) are eliminated.
//
// In-place on h: writes h += attentionOutput, leaving the caller to
// apply LayerNorm (post-norm structure, plan §6.2).
func selfAttentionBatched(h []float32, Wqkv, OutProj []float32, heads, headDim, D, B, Lmax int, realLen []int, rope *ropeTable, s *scratch) {
	if heads*headDim != D {
		panic("encoder: heads*headDim != D")
	}
	BL := B * Lmax

	// 1) Project: QKV = h · Wqkvᵀ → [BL, 3D] into scratch.
	qkv := s.qkv[:BL*3*D]
	s.mm(h, Wqkv, qkv, BL, D, 3*D)

	// 2) Split into Q, K, V — each [BL, D] in scratch.
	Q := s.Q[:BL*D]
	K := s.K[:BL*D]
	V := s.V[:BL*D]
	if BL > 0 && D > 0 {
		_ = qkv[BL*3*D-1]
		_ = Q[BL*D-1]
		_ = K[BL*D-1]
		_ = V[BL*D-1]
	}
	for i := range BL {
		copy(Q[i*D:(i+1)*D], qkv[i*3*D:i*3*D+D])
		copy(K[i*D:(i+1)*D], qkv[i*3*D+D:i*3*D+2*D])
		copy(V[i*D:(i+1)*D], qkv[i*3*D+2*D:i*3*D+3*D])
	}

	// 3) RoPE per sequence's [Lmax, D] slice.
	for b := range B {
		off := b * Lmax * D
		rope.apply(Q[off:off+Lmax*D], heads)
		rope.apply(K[off:off+Lmax*D], heads)
	}

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	ctx := s.ctx[:BL*D]
	zeroF32Slice(ctx) // ctx accumulates per-head writes; padded positions stay 0

	// 4) Per-sequence, per-head attention. M11 alloc fix: hoist the
	// four per-(b,head) buffers (qH/kH/vH + scores) to per-CALL
	// reusables. Sized once to (Lmax * headDim) and (Lmax * Lmax) —
	// the upper bound across all (b, head) iterations — then sliced
	// down to the (L * headDim) / (L * L) actual shape each iter.
	// Pre-M11 this loop allocated 4 fresh slices per iteration:
	// at heads=12, B=50 cands, 12 layers = 28,800 allocs per forward.
	// Profile-confirmed as the dominant alloc hotspot (~50% of total).
	qH := s.qH[:Lmax*headDim]
	kH := s.kH[:Lmax*headDim]
	vHT := s.vH[:Lmax*headDim]          // V transposed to [headDim, L] per (b,head)
	ctxHead := s.ctxHead[:Lmax*headDim] // scores·V output [L, headDim] per (b,head)
	scores := s.scores[:Lmax*Lmax]
	for b := range B {
		L := realLen[b]
		if L == 0 {
			continue
		}
		seqOff := b * Lmax * D
		for headIdx := range heads {
			qH = qH[:L*headDim]
			kH = kH[:L*headDim]
			vHTl := vHT[:headDim*L]
			headOff := seqOff + headIdx*headDim
			for i := range L {
				src := i*D + headOff
				copy(qH[i*headDim:(i+1)*headDim], Q[src:src+headDim])
				copy(kH[i*headDim:(i+1)*headDim], K[src:src+headDim])
			}
			// V transposed so scores·V can use the A·Bᵀ matmul (needs Vᵀ).
			if L%4 == 0 && headDim%4 == 0 && L > 0 && headDim > 0 {
				_ = V[(L-1)*D+headOff+headDim-1]
				_ = vHTl[(headDim-1)*L+L-1]
				for i0 := 0; i0 < L; i0 += 4 {
					for d0 := 0; d0 < headDim; d0 += 4 {
						s0 := (i0+0)*D + headOff + d0
						s1 := (i0+1)*D + headOff + d0
						s2 := (i0+2)*D + headOff + d0
						s3 := (i0+3)*D + headOff + d0

						v00, v01, v02, v03 := V[s0], V[s0+1], V[s0+2], V[s0+3]
						v10, v11, v12, v13 := V[s1], V[s1+1], V[s1+2], V[s1+3]
						v20, v21, v22, v23 := V[s2], V[s2+1], V[s2+2], V[s2+3]
						v30, v31, v32, v33 := V[s3], V[s3+1], V[s3+2], V[s3+3]

						t0 := (d0+0)*L + i0
						t1 := (d0+1)*L + i0
						t2 := (d0+2)*L + i0
						t3 := (d0+3)*L + i0

						vHTl[t0], vHTl[t0+1], vHTl[t0+2], vHTl[t0+3] = v00, v10, v20, v30
						vHTl[t1], vHTl[t1+1], vHTl[t1+2], vHTl[t1+3] = v01, v11, v21, v31
						vHTl[t2], vHTl[t2+1], vHTl[t2+2], vHTl[t2+3] = v02, v12, v22, v32
						vHTl[t3], vHTl[t3+1], vHTl[t3+2], vHTl[t3+3] = v03, v13, v23, v33
					}
				}
			} else {
				for i := range L {
					src := i*D + headOff
					if headDim > 0 {
						_ = V[src+headDim-1]
						_ = vHTl[(headDim-1)*L+i]
						d := 0
						for ; d+3 < headDim; d += 4 {
							vHTl[(d+0)*L+i] = V[src+d+0]
							vHTl[(d+1)*L+i] = V[src+d+1]
							vHTl[(d+2)*L+i] = V[src+d+2]
							vHTl[(d+3)*L+i] = V[src+d+3]
						}
						for ; d < headDim; d++ {
							vHTl[d*L+i] = V[src+d]
						}
					}
				}
			}

			scores = scores[:L*L]
			s.mm(qH, kH, scores, L, headDim, L)
			softmaxRowsScaled(scores, scale, L, L)
			// ctxHead[L, headDim] = scores[L, L] · V[L, headDim] — was the scalar
			// triple-loop hotspot; now the SIMD A·Bᵀ matmul.
			ctxHeadL := ctxHead[:L*headDim]
			s.mm(scores, vHTl, ctxHeadL, L, L, headDim)
			for i := range L {
				dst := i*D + headOff
				copy(ctx[dst:dst+headDim], ctxHeadL[i*headDim:(i+1)*headDim])
			}
		}
	}

	// 5) Output projection into scratch.
	out := s.out[:BL*D]
	s.mm(ctx, OutProj, out, BL, D, D)
	// 6) Residual.
	nh := len(h)
	if nh > 0 {
		_ = h[nh-1]
		_ = out[nh-1]
		i := 0
		for ; i+3 < nh; i += 4 {
			h[i+0] += out[i+0]
			h[i+1] += out[i+1]
			h[i+2] += out[i+2]
			h[i+3] += out[i+3]
		}
		for ; i < nh; i++ {
			h[i] += out[i]
		}
	}
}
