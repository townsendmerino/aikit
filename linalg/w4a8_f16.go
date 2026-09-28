package linalg

import "sync"

// W4A8 over binary16 group scales (f16scale.go). Every path widens the scales it is about to use — a row
// for M < 4, a column chunk for the M ≥ 4 tiles — into a buffer and runs the existing f32-scale kernel, so
// each is bit-identical to the f32 path fed the widened values. The amd64 M < 4 path does the widening
// inside the kernel call (dotW4A8FoldF16RowAVX2).

// f16RowBufGroups bounds the on-stack buffer one row's widened scales use (K up to 32768 at group 32).
const f16RowBufGroups = 1024

var scaleBufPool = sync.Pool{New: func() any { b := make([]float32, 0, 4096); return &b }}

func getScaleBuf(n int) *[]float32 {
	p := scaleBufPool.Get().(*[]float32)
	if cap(*p) < n {
		*p = make([]float32, n)
	}
	*p = (*p)[:n]
	return p
}

// MatmulBTW4A8F16Into is MatmulBTW4A8Into over binary16 scales (wScales16: N·⌈K/group⌉ bit patterns), for
// every M and activation grouping. Bit-identical to MatmulBTW4A8Into fed the widened scales.
func MatmulBTW4A8F16Into(ws *Workspace, a []float32, w4 []byte, wScales16 []uint16, dst []float32, M, K, N, group int) {
	checkMatmulW4A8("MatmulBTW4A8F16", len(a), len(w4), len(wScales16), len(dst), M, K, N, group)
	if g := actGroupFor(ws); g > 0 {
		matmulW4A8Grouped(ws, g, a, int4Layout{w4: w4, wS16: wScales16, group: group, K: K}, dst, M, N) // actgroup.go
		return
	}
	nGroups, bpr := groupsFor(K, group)
	aq := ws.int8Buf(M * K)
	aScales := ws.f32Buf(M)
	quantizeRowsInto(aq, aScales, a, M, K, ws.width)
	if M*N*K < ws.thr() || N < 2 {
		w4a8SpanF16(aq, aScales, w4, wScales16, dst, M, K, N, group, nGroups, bpr, 0, N)
		return
	}
	ws.parallel(N, func(j0, j1 int) {
		w4a8SpanF16(aq, aScales, w4, wScales16, dst, M, K, N, group, nGroups, bpr, j0, j1)
	})
}

// w4a8SpanF16 is w4a8Span over binary16 scales, output columns [j0, j1).
func w4a8SpanF16(aq []int8, aScales []float32, w4 []byte, wScales16 []uint16, dst []float32, M, K, N, group, nGroups, bpr, j0, j1 int) {
	if j0 >= j1 {
		return
	}
	if M < 4 && nGroups <= f16RowBufGroups {
		var sb [f16RowBufGroups]float32
		buf := sb[:]
		for j := j0; j < j1; j++ {
			prow := w4[j*bpr : j*bpr+bpr]
			srow := wScales16[j*nGroups : j*nGroups+nGroups]
			for i := 0; i < M; i++ {
				if aScales[i] == 0 {
					dst[i*N+j] = 0
					continue
				}
				dst[i*N+j] = dotW4A8F16Row(aq[i*K:i*K+K], prow, srow, group, K, buf) * aScales[i]
			}
		}
		return
	}
	// Widen this chunk's scales once, then the f32 span (tile kernels for M ≥ 4) over shifted views: the
	// span indexes weights and scales by the chunk-local column and dst by i·N + column.
	p := getScaleBuf((j1 - j0) * nGroups)
	widenF16(*p, wScales16[j0*nGroups:j1*nGroups])
	w4a8Span(aq, aScales, w4[j0*bpr:j1*bpr], *p, dst[j0:], M, K, N, group, nGroups, bpr, 0, j1-j0)
	scaleBufPool.Put(p)
}
