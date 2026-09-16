package encoder

import (
	"testing"
)

func BenchmarkAttentionHeadExtraction(b *testing.B) {
	const L = 512
	const heads = 12
	const headDim = 64
	const D = heads * headDim
	const mOut = L

	Q := make([]float32, L*D)
	K := make([]float32, L*D)
	V := make([]float32, L*D)
	qH := make([]float32, mOut*headDim)
	kH := make([]float32, L*headDim)
	vHT := make([]float32, headDim*L)
	ctxHead := make([]float32, mOut*headDim)
	ctx := make([]float32, mOut*D)

	for i := range Q {
		Q[i] = float32(i % 17)
		K[i] = float32(i % 19)
		V[i] = float32(i % 23)
		ctxHead[i%(mOut*headDim)] = float32(i % 29)
	}

	b.SetBytes(int64((mOut*headDim + L*headDim + headDim*L + mOut*headDim) * 4))
	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		for headIdx := 0; headIdx < heads; headIdx++ {
			headOff := headIdx * headDim
			for i := range L {
				src := i*D + headOff
				if i < mOut {
					copy(qH[i*headDim:(i+1)*headDim], Q[src:src+headDim])
				}
				copy(kH[i*headDim:(i+1)*headDim], K[src:src+headDim])
			}
			if L%4 == 0 && headDim%4 == 0 && L > 0 && headDim > 0 {
				_ = V[(L-1)*D+headOff+headDim-1]
				_ = vHT[(headDim-1)*L+L-1]
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

						vHT[t0], vHT[t0+1], vHT[t0+2], vHT[t0+3] = v00, v10, v20, v30
						vHT[t1], vHT[t1+1], vHT[t1+2], vHT[t1+3] = v01, v11, v21, v31
						vHT[t2], vHT[t2+1], vHT[t2+2], vHT[t2+3] = v02, v12, v22, v32
						vHT[t3], vHT[t3+1], vHT[t3+2], vHT[t3+3] = v03, v13, v23, v33
					}
				}
			} else {
				for i := range L {
					src := i*D + headOff
					if headDim > 0 {
						_ = V[src+headDim-1]
						_ = vHT[(headDim-1)*L+i]
						d := 0
						for ; d+3 < headDim; d += 4 {
							vHT[(d+0)*L+i] = V[src+d+0]
							vHT[(d+1)*L+i] = V[src+d+1]
							vHT[(d+2)*L+i] = V[src+d+2]
							vHT[(d+3)*L+i] = V[src+d+3]
						}
						for ; d < headDim; d++ {
							vHT[d*L+i] = V[src+d]
						}
					}
				}
			}
			// Scatter head
			for i := range mOut {
				dst := i*D + headOff
				copy(ctx[dst:dst+headDim], ctxHead[i*headDim:(i+1)*headDim])
			}
		}
	}
}

func BenchmarkAttentionCore(b *testing.B) {
	const L = 512
	const heads = 12
	const headDim = 64
	const D = heads * headDim
	const mOut = L

	h := make([]float32, L*D)
	Q := make([]float32, L*D)
	K := make([]float32, L*D)
	V := make([]float32, L*D)
	OutProj := make([]float32, D*D)

	for i := range h {
		h[i] = float32(i%13) * 0.1
		Q[i] = float32(i%17) * 0.1
		K[i] = float32(i%19) * 0.1
		V[i] = float32(i%23) * 0.1
	}
	for i := range OutProj {
		OutProj[i] = float32(i%7) * 0.01
	}

	s := new(scratch)
	s.ensureLayer(L, D, D*4, heads, headDim, L)

	b.SetBytes(int64(L * D * 4))
	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		_ = attentionCore(h, Q, K, V, OutProj, nil, heads, headDim, D, L, mOut, s)
	}
}
