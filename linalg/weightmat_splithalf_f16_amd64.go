//go:build amd64

package linalg

// The split-half paths over binary16 scales: each span widens its column chunk's scales into a pooled
// buffer and runs the f32-scale span on chunk-shifted views, so the result is bit-identical to the f32
// path fed the widened values. Split-half is opt-in (the canonical fused kernel is the default), so the
// per-chunk widening is not tuned further.
// The activation arrives already quantized per row (R-13).
func matmulBTW4A8SplitHalfF16Q(ws *Workspace, q *ActQ, w4sh []byte, s16 []uint16, dst []float32, M, K, N, group int) {
	if K%32 != 0 { // the span has no ragged-tail path: a partial last group would drop silently
		panic("linalg: matmulBTW4A8SplitHalfF16Q requires K a multiple of 32")
	}
	nGroups, bpr := groupsFor(K, group)
	aq, aScales := q.Q, q.S
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

// matmulBTW4A8SplitHalfGroupedF16Q is matmulBTW4A8SplitHalfGroupedQ over binary16 scales, on one activation row already quantized
// per 32 (R-13).
func matmulBTW4A8SplitHalfGroupedF16Q(ws *Workspace, aq []int8, aS []float32, w4sh []byte, s16 []uint16, dst []float32, K, N int) {
	const group = 32
	nGroups, bpr := groupsFor(K, group)
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
