//go:build amd64

package linalg

import "fmt"

// The split-half paths over binary16 scales: each span widens its column chunk's scales into a pooled
// buffer and runs the f32-scale span on chunk-shifted views, so the result is bit-identical to the f32
// path fed the widened values. Split-half is opt-in (the canonical fused kernel is the default), so the
// per-chunk widening is not tuned further.

func matmulBTW4A8SplitHalfF16Into(ws *Workspace, a []float32, w4sh []byte, s16 []uint16, dst []float32, M, K, N, group int) {
	checkMatmulW4A8("MatmulBTW4A8SplitHalfF16", len(a), len(w4sh), len(s16), len(dst), M, K, N, group)
	if K%32 != 0 {
		panic(fmt.Sprintf("linalg: matmulBTW4A8SplitHalfF16Into requires K a multiple of 32, got %d", K))
	}
	nGroups, bpr := groupsFor(K, group)
	aq := ws.int8Buf(M * K)
	aScales := ws.f32Buf(M)
	if M == 1 {
		aScales[0] = quantizeRowInt8(a[:K], aq[:K]) // matmulBTW4A8SplitHalfInto's quantization
	} else {
		quantizeRowsInto(aq, aScales, a, M, K, ws.width) // matmulBTW4A8SplitHalfMultiInto's
	}
	span := func(j0, j1 int) {
		if j0 >= j1 {
			return
		}
		p := getScaleBuf((j1 - j0) * nGroups)
		widenF16(*p, s16[j0*nGroups:j1*nGroups])
		if M == 1 {
			w4a8SplitHalfSpan(aq, aScales[0], w4sh[j0*bpr:j1*bpr], *p, dst[j0:], K, N, nGroups, bpr, 0, j1-j0)
		} else {
			w4a8SplitHalfSpanMulti(aq, aScales, w4sh[j0*bpr:j1*bpr], *p, dst[j0:], M, K, N, nGroups, bpr, 0, j1-j0)
		}
		scaleBufPool.Put(p)
	}
	if M*N*K < ws.thr() || N < 2 {
		span(0, N)
		return
	}
	ws.parallel(N, span)
}

// matmulBTW4A8SplitHalfGroupedF16Into is matmulBTW4A8SplitHalfGroupedInto over binary16 scales.
func matmulBTW4A8SplitHalfGroupedF16Into(ws *Workspace, a []float32, w4sh []byte, s16 []uint16, dst []float32, K, N int) {
	const group = 32
	nGroups, bpr := groupsFor(K, group)
	aq := ws.int8Buf(K)
	aS := ws.f32Buf(nGroups)
	QuantizeActivationsGroupedInto(aq, aS, a[:K], 1, K, group)
	span := func(j0, j1 int) {
		if j0 >= j1 {
			return
		}
		p := getScaleBuf((j1 - j0) * nGroups)
		widenF16(*p, s16[j0*nGroups:j1*nGroups])
		buf := *p
		for j := j0; j < j1; j++ {
			dst[j] = dotW4A8SplitHalfScaledAVX2(&aq[0], &w4sh[j*bpr], &buf[(j-j0)*nGroups], &aS[0], nGroups)
		}
		scaleBufPool.Put(p)
	}
	if N*K < ws.thr() || N < 2 {
		span(0, N)
		return
	}
	ws.parallel(N, span)
}

// w4a8BatchSplitHalfSpanF16 is w4a8BatchSplitHalfSpan over binary16 scales.
func w4a8BatchSplitHalfSpanF16(aq []int8, aScale float32, splitHalf []byte, s16 []uint16, dst []float32, K, N, nGroups, bpr, j0, j1 int) {
	if j0 >= j1 {
		return
	}
	p := getScaleBuf((j1 - j0) * nGroups)
	widenF16(*p, s16[j0*nGroups:j1*nGroups])
	w4a8SplitHalfSpan(aq, aScale, splitHalf[j0*bpr:j1*bpr], *p, dst[j0:], K, N, nGroups, bpr, 0, j1-j0)
	scaleBufPool.Put(p)
}
