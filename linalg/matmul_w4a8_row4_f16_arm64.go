//go:build arm64

package linalg

import "fmt"

// The row4 paths over binary16 scales (f16scale.go). Each quad's interleaved scales are widened into a
// buffer and the existing f32-scale kernel runs on them, so every result is bit-identical to the f32 path
// fed the widened values.

// RepackInt4Row4ScalesQuadF16 is RepackInt4Row4ScalesQuad over binary16 scales.
func RepackInt4Row4ScalesQuadF16(dst, src []uint16, nGroups int) {
	requireLen("RepackInt4Row4ScalesQuadF16", "dst", len(dst), 4*nGroups)
	requireLen("RepackInt4Row4ScalesQuadF16", "src", len(src), 4*nGroups)
	for g := range nGroups {
		for r := range 4 {
			dst[4*g+r] = src[r*nGroups+g]
		}
	}
}

// RepackW4A8Row4ScalesF16 is RepackW4A8Row4Scales over binary16 scales.
func RepackW4A8Row4ScalesF16(scales []uint16, N, K, group int) []uint16 {
	if group != 32 || N%4 != 0 || K%group != 0 {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4ScalesF16 requires group=32, N%%4==0, K%%group==0 (N=%d K=%d group=%d)", N, K, group))
	}
	nGroups, _ := groupsFor(K, group)
	if len(scales) < N*nGroups {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4ScalesF16 scales len %d < N*nGroups = %d", len(scales), N*nGroups))
	}
	out := make([]uint16, N*nGroups)
	for q := 0; q < N/4; q++ {
		base := q * 4 * nGroups
		RepackInt4Row4ScalesQuadF16(out[base:base+4*nGroups], scales[base:base+4*nGroups], nGroups)
	}
	return out
}

// matmulBTW4A8Row4F16Into is MatmulBTW4A8Row4Into (M=1) over binary16 interleaved scales.
func matmulBTW4A8Row4F16Into(ws *Workspace, a []float32, w4Row4 []byte, s4 []uint16, dst []float32, K, N, group int) {
	const M = 1
	checkMatmulW4A8("MatmulBTW4A8Row4F16", len(a), len(w4Row4), len(s4), len(dst), M, K, N, group)
	if group != 32 || N%4 != 0 || K%group != 0 || K < group {
		panic(fmt.Sprintf("linalg: matmulBTW4A8Row4F16Into requires group=32, N%%4==0, K a multiple of 32 (K=%d N=%d group=%d)", K, N, group))
	}
	if g := actGroupFor(ws); g > 0 {
		if g == 32 {
			matmulBTW4A8Row4GroupedF16Into(ws, a, w4Row4, s4, dst, K, N)
			return
		}
		matmulW4A8Grouped(ws, g, a, int4Layout{r4: w4Row4, r4S16: s4, group: group, K: K}, dst, M, N)
		return
	}
	nGroups, bpr := groupsFor(K, group)
	aq := ws.int8Buf(K)
	aScale := quantizeRowInt8(a[:K], aq)
	if aScale == 0 {
		for j := range N {
			dst[j] = 0
		}
		return
	}
	nQuads := N / 4
	var corr []int32
	if w4a8RowFold {
		corr = ws.int32Buf(4 * nGroups)
		w4a8LaneCorrNeg8(&aq[0], &corr[0], nGroups)
	}
	if M*N*K < ws.thr() || nQuads < 2 {
		w4a8Row4SpanF16(aq, corr, aScale, w4Row4, s4, dst, nGroups, bpr, 0, nQuads)
		return
	}
	ws.parallel(nQuads, func(q0, q1 int) {
		w4a8Row4SpanF16(aq, corr, aScale, w4Row4, s4, dst, nGroups, bpr, q0, q1)
	})
}

// w4a8Row4SpanF16 is w4a8Row4Span over binary16 scales: each quad's 4·nGroups scales are widened into one
// buffer (reused across the span) before the same kernel call.
func w4a8Row4SpanF16(aq []int8, corr []int32, aScale float32, w4Row4 []byte, s4 []uint16, dst []float32, nGroups, bpr, q0, q1 int) {
	if q0 >= q1 {
		return
	}
	p := getScaleBuf(4 * nGroups)
	sblk := *p
	var out [4]float32
	for q := q0; q < q1; q++ {
		blk := w4Row4[q*4*bpr : q*4*bpr+4*bpr]
		widenF16(sblk, s4[q*4*nGroups:q*4*nGroups+4*nGroups])
		if corr != nil {
			dotW4A8SplitHalf4RowFold(&aq[0], &corr[0], &blk[0], &sblk[0], &out[0], nGroups)
		} else {
			dotW4A8SplitHalf4Row(&aq[0], &blk[0], &sblk[0], &out[0], nGroups)
		}
		dst[q*4] = out[0] * aScale
		dst[q*4+1] = out[1] * aScale
		dst[q*4+2] = out[2] * aScale
		dst[q*4+3] = out[3] * aScale
	}
	scaleBufPool.Put(p)
}

// matmulBTW4A8Row4GroupedF16Into is matmulBTW4A8Row4GroupedInto over binary16 scales.
func matmulBTW4A8Row4GroupedF16Into(ws *Workspace, a []float32, w4Row4 []byte, s4 []uint16, dst []float32, K, N int) {
	const group = 32
	nGroups, bpr := groupsFor(K, group)
	aq := ws.int8Buf(K)
	aS := ws.f32Buf(nGroups)
	QuantizeActivationsGroupedInto(aq, aS, a[:K], 1, K, group)
	var corr []int32
	if w4a8RowFold {
		corr = ws.int32Buf(4 * nGroups)
		w4a8LaneCorrNeg8(&aq[0], &corr[0], nGroups)
	}
	nQuads := N / 4
	span := func(q0, q1 int) {
		var out [4]float32
		cs := make([]float32, 4*nGroups)
		for q := q0; q < q1; q++ {
			blk := w4Row4[q*4*bpr : q*4*bpr+4*bpr]
			sblk := s4[q*4*nGroups : q*4*nGroups+4*nGroups]
			for g := range nGroups {
				s := aS[g]
				cs[4*g], cs[4*g+1], cs[4*g+2], cs[4*g+3] = f16ToF32(sblk[4*g])*s, f16ToF32(sblk[4*g+1])*s, f16ToF32(sblk[4*g+2])*s, f16ToF32(sblk[4*g+3])*s
			}
			if corr != nil {
				dotW4A8SplitHalf4RowFold(&aq[0], &corr[0], &blk[0], &cs[0], &out[0], nGroups)
			} else {
				dotW4A8SplitHalf4Row(&aq[0], &blk[0], &cs[0], &out[0], nGroups)
			}
			dst[q*4], dst[q*4+1], dst[q*4+2], dst[q*4+3] = out[0], out[1], out[2], out[3]
		}
	}
	if N*K < ws.thr() || nQuads < 2 {
		span(0, nQuads)
		return
	}
	ws.parallel(nQuads, span)
}

// matmulBTW4A8Row4TileF16Into is MatmulBTW4A8Row4TileInto (M>1) over binary16 scales: a quad range's scales
// are widened once, then the f32 tile span runs on quad-shifted views.
func matmulBTW4A8Row4TileF16Into(ws *Workspace, a []float32, w4Row4 []byte, s4 []uint16, dst []float32, M, K, N, group int) {
	checkMatmulW4A8("MatmulBTW4A8Row4TileF16", len(a), len(w4Row4), len(s4), len(dst), M, K, N, group)
	if group != 32 || N%4 != 0 || K%group != 0 || K < group {
		panic(fmt.Sprintf("linalg: matmulBTW4A8Row4TileF16Into requires group=32, N%%4==0, K a multiple of 32 (K=%d N=%d group=%d)", K, N, group))
	}
	nGroups, bpr := groupsFor(K, group)
	aq := ws.int8Buf(M * K)
	aScales := ws.f32Buf(M)
	quantizeRowsInto(aq, aScales, a, M, K, ws.width)
	nQuads := N / 4
	span := func(q0, q1 int) {
		if q0 >= q1 {
			return
		}
		p := getScaleBuf((q1 - q0) * 4 * nGroups)
		widenF16(*p, s4[q0*4*nGroups:q1*4*nGroups])
		w4a8Row4TileSpan(aq, aScales, w4Row4[q0*4*bpr:q1*4*bpr], *p, dst[q0*4:], M, K, N, nGroups, bpr, 0, q1-q0)
		scaleBufPool.Put(p)
	}
	if M*N*K < ws.thr() || nQuads < 2 {
		span(0, nQuads)
		return
	}
	ws.parallel(nQuads, span)
}

// w4a8BatchRow4SpanF16 is w4a8BatchRow4Span over binary16 scales.
func w4a8BatchRow4SpanF16(aq []int8, aScale float32, row4 []byte, s4 []uint16, dst []float32, nGroups, bpr, q0, q1 int) {
	w4a8Row4SpanF16(aq, nil, aScale, row4, s4, dst, nGroups, bpr, q0, q1)
}
