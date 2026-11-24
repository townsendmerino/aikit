package linalg

// MatmulQKAcc64F16 is MatmulQKAcc64 over a key cache stored as IEEE binary16 bits: dst[M,N] = a[M,K] · b[N,K]ᵀ, f64-accumulated,
// b's rows K-wide at bOff within rows rowStride halves apart. Each half is widened exactly to f32, so the result equals
// MatmulQKAcc64 run on the widened cache bit for bit (every key's sum is the same d-ascending fold), while the cache is read at two
// bytes an element and never widened into scratch.
//
// Output contract: overwrites dst; do not pre-zero.
func MatmulQKAcc64F16(a []float32, bMat []uint16, dst []float32, M, K, N, bOff, rowStride int) {
	checkMatmulQKAcc64(len(a), len(bMat), len(dst), M, K, N, bOff, rowStride)
	for i := range M {
		arow := a[i*K : i*K+K]
		drow := dst[i*N : i*N+N]
		j := qkAcc64KeysF16(arow, bMat, drow, K, N, bOff, rowStride)
		for ; j+8 <= N; j += 8 {
			base := bOff + j*rowStride
			var s0, s1, s2, s3, s4, s5, s6, s7 float64
			for d := range K {
				qd := float64(arow[d])
				s0 += qd * float64(f16ToF32(bMat[base+d]))
				s1 += qd * float64(f16ToF32(bMat[base+rowStride+d]))
				s2 += qd * float64(f16ToF32(bMat[base+2*rowStride+d]))
				s3 += qd * float64(f16ToF32(bMat[base+3*rowStride+d]))
				s4 += qd * float64(f16ToF32(bMat[base+4*rowStride+d]))
				s5 += qd * float64(f16ToF32(bMat[base+5*rowStride+d]))
				s6 += qd * float64(f16ToF32(bMat[base+6*rowStride+d]))
				s7 += qd * float64(f16ToF32(bMat[base+7*rowStride+d]))
			}
			drow[j], drow[j+1], drow[j+2], drow[j+3] = float32(s0), float32(s1), float32(s2), float32(s3)
			drow[j+4], drow[j+5], drow[j+6], drow[j+7] = float32(s4), float32(s5), float32(s6), float32(s7)
		}
		for ; j < N; j++ {
			row := bMat[bOff+j*rowStride : bOff+j*rowStride+K]
			var s float64
			for d := range K {
				s += float64(arow[d]) * float64(f16ToF32(row[d]))
			}
			drow[j] = float32(s)
		}
	}
}

// MatmulAVAcc64F16 is MatmulAVAcc64 over a value cache stored as IEEE binary16 bits: dst[M,hd] = scores[M,nKeys] · V_head[nKeys,hd],
// V_head hd-wide at headOff within rows rowStride halves apart, f64-accumulated. Each half is widened exactly, so the result equals
// MatmulAVAcc64 on the widened cache bit for bit (every output dim's sum is the same key-ascending fold).
//
// Output contract: overwrites dst; do not pre-zero. acc is [hd] scratch, ignored on entry.
func MatmulAVAcc64F16(scores []float32, vals []uint16, dst []float32, acc []float64, M, nKeys, hd, headOff, rowStride int) {
	checkMatmulAVAcc64(len(scores), len(vals), len(dst), len(acc), M, nKeys, hd, headOff, rowStride)
	for i := range M {
		srow := scores[i*nKeys : i*nKeys+nKeys]
		drow := dst[i*hd : i*hd+hd]
		d0 := avAcc64BlocksF16(srow, vals, drow, nKeys, hd, headOff, rowStride)
		for ; d0+16 <= hd; d0 += 16 {
			var a [16]float64
			for s := range nKeys {
				w := float64(srow[s])
				v := vals[headOff+s*rowStride+d0 : headOff+s*rowStride+d0+16 : headOff+s*rowStride+d0+16]
				for d := range a {
					a[d] += w * float64(f16ToF32(v[d]))
				}
			}
			for d := range a {
				drow[d0+d] = float32(a[d])
			}
		}
		if d0 < hd {
			for d := d0; d < hd; d++ {
				acc[d] = 0
			}
			for s := range nKeys {
				w := float64(srow[s])
				vrow := vals[headOff+s*rowStride : headOff+s*rowStride+hd]
				for d := d0; d < hd; d++ {
					acc[d] += w * float64(f16ToF32(vrow[d]))
				}
			}
			for d := d0; d < hd; d++ {
				drow[d] = float32(acc[d])
			}
		}
	}
}
