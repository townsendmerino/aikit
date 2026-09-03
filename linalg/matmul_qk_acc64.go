package linalg

// MatmulQKAcc64 computes dst[M,N] = a[M,K] · b[N,K]ᵀ, f64-accumulated, where b's
// rows are contiguous K-wide slices of a row-major buffer spaced rowStride
// apart (bOff selects a head's K-wide column range within each row) — the
// attention QKᵀ step. This is MatmulBTAcc64Strided's bElemStride=1 case,
// specialized: attention's own QKᵀ call already reads each row contiguously
// (unlike scores·V — see MatmulAVAcc64), so the lever here isn't memory
// layout, it's FMA latency: dotF32Acc64's single sequential accumulator chain
// (deliberately not vectorized, to stay order-preserving) can't issue its next
// add until the previous one resolves, leaving execution ports idle even
// though the outputs (each key's dot) are fully independent of each other.
//
// This interleaves 8 keys' dot products as 8 concurrent accumulator chains in
// one pass over d: each key's own chain is still the exact d=0..K-1 ascending
// sequential fold (bit-identical to computing it alone), but 8 independent
// chains in flight hide each other's FMA latency. 8-wide measured 4.4x on
// apple-m1pro (real 1.5B shape, depth 130 and 8192), vs 3.0x for 4-wide —
// picked over 4-wide per that measurement, not a guess. A scalar tail handles
// N%8 != 0.
func MatmulQKAcc64(a, bMat, dst []float32, M, K, N, bOff, rowStride int) {
	checkMatmulQKAcc64(len(a), len(bMat), len(dst), M, K, N, bOff, rowStride)
	for i := range M {
		arow := a[i*K : i*K+K]
		drow := dst[i*N : i*N+N]
		// arm64: whole 16-key blocks go through the NEON lane-per-key kernel
		// (attn_acc64_arm64.s) — one key per f64 lane, 8 accumulator registers,
		// the same d-ascending chain per key as the 8-chain loop below, which
		// finishes the N%16 tail. qkAcc64Keys returns 0 elsewhere, and for a K
		// that is not a multiple of 4. TestMatmulQKAcc64_neonMatchesGo pins the
		// kernel against this loop key for key.
		j := qkAcc64Keys(arow, bMat, drow, K, N, bOff, rowStride)
		for ; j+8 <= N; j += 8 {
			base := bOff + j*rowStride
			r0 := bMat[base : base+K]
			r1 := bMat[base+rowStride : base+rowStride+K]
			r2 := bMat[base+2*rowStride : base+2*rowStride+K]
			r3 := bMat[base+3*rowStride : base+3*rowStride+K]
			r4 := bMat[base+4*rowStride : base+4*rowStride+K]
			r5 := bMat[base+5*rowStride : base+5*rowStride+K]
			r6 := bMat[base+6*rowStride : base+6*rowStride+K]
			r7 := bMat[base+7*rowStride : base+7*rowStride+K]
			var s0, s1, s2, s3, s4, s5, s6, s7 float64
			for d := range K {
				qd := float64(arow[d])
				s0 += qd * float64(r0[d])
				s1 += qd * float64(r1[d])
				s2 += qd * float64(r2[d])
				s3 += qd * float64(r3[d])
				s4 += qd * float64(r4[d])
				s5 += qd * float64(r5[d])
				s6 += qd * float64(r6[d])
				s7 += qd * float64(r7[d])
			}
			drow[j] = float32(s0)
			drow[j+1] = float32(s1)
			drow[j+2] = float32(s2)
			drow[j+3] = float32(s3)
			drow[j+4] = float32(s4)
			drow[j+5] = float32(s5)
			drow[j+6] = float32(s6)
			drow[j+7] = float32(s7)
		}
		for ; j < N; j++ { // scalar tail, N%8 != 0
			row := bMat[bOff+j*rowStride : bOff+j*rowStride+K]
			var s float64
			for d := range K {
				s += float64(arow[d]) * float64(row[d])
			}
			drow[j] = float32(s)
		}
	}
}

func checkMatmulQKAcc64(aLen, bLen, dstLen, M, K, N, bOff, rowStride int) {
	if M < 0 || K < 0 || N < 0 || bOff < 0 || rowStride < K {
		panic("linalg: MatmulQKAcc64 invalid shape")
	}
	requireExactLen("MatmulQKAcc64", "a", aLen, mul(M, K))
	requireExactLen("MatmulQKAcc64", "dst", dstLen, mul(M, N))
	need := bOff + max(0, N-1)*rowStride + K
	if bLen < need {
		panic("linalg: MatmulQKAcc64 bMat too short for the given shape/strides")
	}
}
