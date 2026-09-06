//go:build arm64

package linalg

import "fmt"

// docs/task-w4a8-neon-bandwidth.md's cold-fix harness pass (goinfer's
// docs/task-zeno-compare.md confirmed the mechanism by PMU counters: row4's
// 4 simultaneous accumulator chains share each cold cache-line region, so
// one miss stalls 4 rows' in-flight work at once). This file is the
// full-tensor exported plumbing for the two mechanism-aimed remedies —
// HARNESS ONLY, mirroring MatmulBTW4A8Row4Into's own shape exactly but never
// wired into WeightMat's dispatch (weightmat_row4_arm64.go) or the .giw
// format. Neither remedy changes any numeric result: both are proven
// bit-identical to MatmulBTW4A8Row4Into by construction (same per-row math,
// only the memory access pattern differs).

// MatmulBTW4A8Row4PrefetchInto is MatmulBTW4A8Row4Into plus a software
// prefetch hint, prefetchDistance BYTES ahead of each quad's own read
// position, threaded down to dotW4A8SplitHalf4RowPrefetch. Same
// w4Row4/wScales4 layout as MatmulBTW4A8Row4Into (RepackW4A8Row4/
// RepackW4A8Row4Scales, or a .giw kind-4 tensor's on-disk bytes) — this is
// the SAME data, just dispatched through a kernel with one added PRFM per
// iteration. Same M/N/K/group contract as MatmulBTW4A8Row4Into.
func MatmulBTW4A8Row4PrefetchInto(ws *Workspace, a []float32, w4Row4 []byte, wScales4 []float32, dst []float32, M, K, N, group, prefetchDistance int) {
	checkMatmulW4A8("MatmulBTW4A8Row4Prefetch", len(a), len(w4Row4), len(wScales4), len(dst), M, K, N, group)
	checkGroupMatmul("MatmulBTW4A8Row4Prefetch", len(a), w4Row4, wScales4, len(dst), M, K, N, group)
	if M != 1 {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4PrefetchInto requires M=1, got M=%d", M))
	}
	if group != 32 {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4PrefetchInto requires group=32, got %d", group))
	}
	if N%4 != 0 {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4PrefetchInto requires N a multiple of 4, got %d", N))
	}
	if K%group != 0 {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4PrefetchInto requires K a multiple of group=%d, got %d", group, K))
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
	var out [4]float32
	nQuads := N / 4
	for q := range nQuads {
		blk := w4Row4[q*4*bpr : q*4*bpr+4*bpr]
		sblk := wScales4[q*4*nGroups : q*4*nGroups+4*nGroups]
		dotW4A8SplitHalf4RowPrefetch(&aq[0], &blk[0], &sblk[0], &out[0], nGroups, prefetchDistance)
		dst[q*4] = out[0] * aScale
		dst[q*4+1] = out[1] * aScale
		dst[q*4+2] = out[2] * aScale
		dst[q*4+3] = out[3] * aScale
	}
}

// RepackW4A8Row4Deshared is RepackW4A8Row4's non-interleaved counterpart:
// same canonical-packed input, same N/K/group contract, but returns 4
// SEPARATE byte arrays instead of one interleaved block — b0[q*bpr:(q+1)*bpr]
// holds quad q's "row-position-0" bytes, b1 quad q's "row-position-1" bytes,
// and so on, each role's bytes contiguous across every quad but the 4 roles
// living in 4 entirely separate allocations (de-sharing the cache line 4
// concurrent chains would otherwise contend on a cold miss).
func RepackW4A8Row4Deshared(packed []byte, N, K, group int) (b0, b1, b2, b3 []byte) {
	if group != 32 {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4Deshared requires group=32, got %d", group))
	}
	if N%4 != 0 {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4Deshared requires N a multiple of 4, got %d", N))
	}
	if K%group != 0 {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4Deshared requires K a multiple of group=%d, got %d", group, K))
	}
	_, bpr := groupsFor(K, group)
	if len(packed) < N*bpr {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4Deshared packed len %d < N*bytesPerRow = %d", len(packed), N*bpr))
	}
	nQuads := N / 4
	b0 = make([]byte, nQuads*bpr)
	b1 = make([]byte, nQuads*bpr)
	b2 = make([]byte, nQuads*bpr)
	b3 = make([]byte, nQuads*bpr)
	for q := range nQuads {
		r0, r1, r2, r3 := q*4*bpr, (q*4+1)*bpr, (q*4+2)*bpr, (q*4+3)*bpr
		sh0, sh1, sh2, sh3 := repackSplitHalf4RowDeshared(packed[r0:r0+bpr], packed[r1:r1+bpr], packed[r2:r2+bpr], packed[r3:r3+bpr], K)
		copy(b0[q*bpr:(q+1)*bpr], sh0)
		copy(b1[q*bpr:(q+1)*bpr], sh1)
		copy(b2[q*bpr:(q+1)*bpr], sh2)
		copy(b3[q*bpr:(q+1)*bpr], sh3)
	}
	return b0, b1, b2, b3
}

// RepackW4A8Row4DesharedScales is RepackW4A8Row4Deshared's counterpart for
// the per-group f32 scales: no repacking needed per row (scales are already
// separable), just split into 4 per-role arrays the same way.
func RepackW4A8Row4DesharedScales(scales []float32, N, K, group int) (s0, s1, s2, s3 []float32) {
	if group != 32 {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4DesharedScales requires group=32, got %d", group))
	}
	if N%4 != 0 {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4DesharedScales requires N a multiple of 4, got %d", N))
	}
	nGroups, _ := groupsFor(K, group)
	if len(scales) < N*nGroups {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4DesharedScales scales len %d < N*nGroups = %d", len(scales), N*nGroups))
	}
	nQuads := N / 4
	s0 = make([]float32, nQuads*nGroups)
	s1 = make([]float32, nQuads*nGroups)
	s2 = make([]float32, nQuads*nGroups)
	s3 = make([]float32, nQuads*nGroups)
	for q := range nQuads {
		r0, r1, r2, r3 := q*4*nGroups, (q*4+1)*nGroups, (q*4+2)*nGroups, (q*4+3)*nGroups
		copy(s0[q*nGroups:(q+1)*nGroups], scales[r0:r0+nGroups])
		copy(s1[q*nGroups:(q+1)*nGroups], scales[r1:r1+nGroups])
		copy(s2[q*nGroups:(q+1)*nGroups], scales[r2:r2+nGroups])
		copy(s3[q*nGroups:(q+1)*nGroups], scales[r3:r3+nGroups])
	}
	return s0, s1, s2, s3
}

// MatmulBTW4A8Row4DesharedInto computes dst[N] = a[K]·bᵀ from
// RepackW4A8Row4Deshared/RepackW4A8Row4DesharedScales's 4-array layout —
// bit-identical to MatmulBTW4A8Row4Into/MatmulBTW4A8Into for the same
// logical weights. Same M/N/K/group contract.
func MatmulBTW4A8Row4DesharedInto(ws *Workspace, a []float32, w0, w1, w2, w3 []byte, s0, s1, s2, s3 []float32, dst []float32, M, K, N, group int) {
	if M != 1 {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4DesharedInto requires M=1, got M=%d", M))
	}
	if group != 32 {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4DesharedInto requires group=32, got %d", group))
	}
	if N%4 != 0 {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4DesharedInto requires N a multiple of 4, got %d", N))
	}
	if K%group != 0 {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4DesharedInto requires K a multiple of group=%d, got %d", group, K))
	}
	nGroups, bpr := groupsFor(K, group)
	nQuads := N / 4
	if len(w0) < nQuads*bpr || len(w1) < nQuads*bpr || len(w2) < nQuads*bpr || len(w3) < nQuads*bpr {
		panic("linalg: MatmulBTW4A8Row4DesharedInto: a per-role weight array is too short")
	}
	if len(s0) < nQuads*nGroups || len(s1) < nQuads*nGroups || len(s2) < nQuads*nGroups || len(s3) < nQuads*nGroups {
		panic("linalg: MatmulBTW4A8Row4DesharedInto: a per-role scale array is too short")
	}
	aq := ws.int8Buf(K)
	aScale := quantizeRowInt8(a[:K], aq)
	if aScale == 0 {
		for j := range N {
			dst[j] = 0
		}
		return
	}
	var out [4]float32
	for q := range nQuads {
		p0 := w0[q*bpr : q*bpr+bpr]
		p1 := w1[q*bpr : q*bpr+bpr]
		p2 := w2[q*bpr : q*bpr+bpr]
		p3 := w3[q*bpr : q*bpr+bpr]
		sc0 := s0[q*nGroups : q*nGroups+nGroups]
		sc1 := s1[q*nGroups : q*nGroups+nGroups]
		sc2 := s2[q*nGroups : q*nGroups+nGroups]
		sc3 := s3[q*nGroups : q*nGroups+nGroups]
		dotW4A8SplitHalf4RowDeshared(&aq[0], &p0[0], &p1[0], &p2[0], &p3[0], &sc0[0], &sc1[0], &sc2[0], &sc3[0], &out[0], nGroups)
		dst[q*4] = out[0] * aScale
		dst[q*4+1] = out[1] * aScale
		dst[q*4+2] = out[2] * aScale
		dst[q*4+3] = out[3] * aScale
	}
}
