package linalg

import "fmt"

// PER-GROUP ACTIVATION QUANTIZATION — the Go reference (goinfer
// docs/tasks/task-actquant-pergroup-2026-09.md, goinfer queue H2).
//
// Every W8A8 and W4A8 kernel here quantizes an activation row to int8 with ONE max/127 scale for
// the whole row. A model whose projection inputs carry massive outliers loses most of each row
// to rounding. Measured on Phi-3-mini: max/rms ~80-90 at the down_proj input, so a per-row
// scale rounds 99.9% of the row to zero. Scaling each group of G consecutive inputs separately
// keeps an outlier's damage inside its own group, which is what llama.cpp's Q8_0/Q8_K activation
// blocks do.
//
// This file is the REFERENCE: plain Go, architecture-independent, exact integer arithmetic within
// each group, behind SetActQuantGroup. It exists to answer the quality question before any
// SIMD/PTX kernel is written, and to be the oracle those kernels are tested against. It is not
// fast (scalar Go, the weight codes re-gathered per call) and does not try to be. With the group
// at 0, the default, nothing here runs and every existing kernel is untouched.

// actQuantGroup is the activation group size the W8A8/W4A8 entry points use: 0 = one scale per
// row (the kernels' existing behaviour); > 0 = one scale per that many inputs, via the reference
// below. Not goroutine-safe to change mid-matmul; set it before loading/running a model.
var actQuantGroup int

// SetActQuantGroup selects per-group activation scales for every W8A8/W4A8 matmul entry point
// (MatmulBTW8A8Into/Batch, MatmulBTW4A8Into/Batch, MatmulBTW4A8Row4Into and
// WeightMat.MatmulBTW4A8Into): 0 restores the per-row kernels, g > 0 routes those calls through
// the Go reference with one int8 scale per g inputs. For W4A8, g and the weight group must divide
// one another.
func SetActQuantGroup(g int) {
	if g < 0 {
		panic(fmt.Sprintf("linalg: SetActQuantGroup(%d): group must be >= 0", g))
	}
	actQuantGroup = g
}

// ActQuantGroup reports the activation group size SetActQuantGroup selected (0 = per-row).
func ActQuantGroup() int { return actQuantGroup }

// SetActQuantGroup sets this workspace's activation group size: every W8A8/W4A8 matmul run through
// it uses per-g activation scales (g > 0) regardless of the process-wide SetActQuantGroup, or defers
// to the process-wide setting (g == 0). A consumer with one workspace per model or per call — goinfer
// sets it from the model's options before each matmul — gets a per-model choice with no global state.
func (w *Workspace) SetActQuantGroup(g int) {
	if g < 0 {
		panic(fmt.Sprintf("linalg: Workspace.SetActQuantGroup(%d): group must be >= 0", g))
	}
	w.actGroup = g
}

// ActQuantGroup reports this workspace's own activation group size (0 = defer to the process-wide one).
func (w *Workspace) ActQuantGroup() int { return w.actGroup }

// actGroupFor is the activation group size a matmul through ws uses: the workspace's own when set,
// else the process-wide one.
func actGroupFor(ws *Workspace) int {
	if ws != nil && ws.actGroup > 0 {
		return ws.actGroup
	}
	return actQuantGroup
}

// QuantizeActivationsGroupedInto quantizes a's M rows of K floats to int8 with one symmetric
// max/127 scale per group of `group` consecutive elements (the last group of a row ragged when
// group does not divide K). scales[m*nG+g] is row m's group g, nG = ceil(K/group). Each group
// uses exactly the per-row quantizer's rounding, clamp and zero convention (an all-zero group gets
// scale 0 and zero codes), so per-row quantization is the special case group >= K.
//
// Output contract: overwrites aq and scales; do not pre-zero. Cover aq[:M*K] and scales[:M*nG] (nG = ceil(K/group)).
func QuantizeActivationsGroupedInto(aq []int8, scales []float32, a []float32, M, K, group int) {
	nG := (K + group - 1) / group
	if group <= 0 || len(aq) < M*K || len(scales) < M*nG || len(a) < M*K {
		panic("linalg: QuantizeActivationsGroupedInto: bad group or short buffer")
	}
	for m := range M {
		for g := range nG {
			lo, hi := m*K+g*group, m*K+min((g+1)*group, K)
			scales[m*nG+g] = quantizeRowInt8Core(a[lo:hi], aq[lo:hi], 0)
		}
	}
}

// groupedDot is Σ_k aq·w over one row, scaled per segment: segments are the intersection of the
// activation groups (ag, scales aS) and the weight groups (wg, scales wS, nil for a per-row weight
// scale applied by the caller). Within a segment the dot is exact int32; each segment's partial
// is scaled in f32 and accumulated in f32. ag and wg must divide one another.
func groupedDot(aq []int8, aS []float32, ag int, w []int8, wS []float32, wg int, K int) float32 {
	seg := ag
	if wS != nil && wg < seg {
		seg = wg
	}
	var acc float32
	for k0 := 0; k0 < K; k0 += seg {
		k1 := min(k0+seg, K)
		var isum int32
		for k := k0; k < k1; k++ {
			isum += int32(aq[k]) * int32(w[k])
		}
		s := aS[k0/ag]
		if wS != nil {
			s *= wS[k0/wg]
		}
		acc += float32(isum) * s
	}
	return acc
}

func checkGroups(ag, wg int) {
	if ag%wg != 0 && wg%ag != 0 {
		panic(fmt.Sprintf("linalg: activation group %d and weight group %d must divide one another", ag, wg))
	}
}

// refParallel runs fn over [0, N) row spans on the workspace's workers (the reference's only
// concession to speed: a 7B decode step at one core would take seconds).
func refParallel(ws *Workspace, N int, fn func(n0, n1 int)) {
	width := 0
	if ws != nil {
		width = ws.width
	}
	if N < 64 {
		fn(0, N)
		return
	}
	parallelSpawnCols(N, resolveWidth(width), fn)
}

// quantizeActGrouped quantizes a into workspace scratch with group g and returns the codes,
// scales and group count per row.
func quantizeActGrouped(ws *Workspace, g int, a []float32, M, K int) ([]int8, []float32, int) {
	if ws == nil {
		ws = new(Workspace)
	}
	nG := (K + g - 1) / g
	aq := ws.int8Buf(M * K)
	aS := ws.f32Buf(M * nG)
	QuantizeActivationsGroupedInto(aq, aS, a, M, K, g)
	return aq, aS, nG
}

// matmulW8A8GroupedRef is MatmulBTW8A8Into with per-group activation scales:
// dst[m,n] = bScales[n] · Σ_g aS[m,g] · Σ_{k∈g} aq[m,k]·bQ[n,k].
func matmulW8A8GroupedRef(ws *Workspace, g int, a []float32, bQ []int8, bScales, dst []float32, M, K, N int) {
	aq, aS, nG := quantizeActGrouped(ws, g, a, M, K)
	refParallel(ws, N, func(n0, n1 int) {
		for n := n0; n < n1; n++ {
			w := bQ[n*K : n*K+K]
			for m := range M {
				dst[m*N+n] = groupedDot(aq[m*K:m*K+K], aS[m*nG:m*nG+nG], g, w, nil, 0, K) * bScales[n]
			}
		}
	})
}

// int4Layout reads row n's int4 codes (nibble-8, in [-8, 7]) and its per-group scales from one of
// the three int4 layouts: canonical (w4, wS, any group), split-half (sh, wS, group 32) or row4
// (r4, r4S, group 32). Each branch is the same gather as that layout's dequantizer
// (DequantizeRowInt4, dequantizeRowFromSplitHalf, dequantizeRowFromRow4), emitting codes instead
// of products.
type int4Layout struct {
	w4, sh, r4  []byte
	wS, r4S     []float32 // f32 scales (the f32 free functions' callers)
	wS16, r4S16 []uint16  // binary16 scales (a WeightMat, MatmulBTW4A8F16Into); used when wS/r4S are nil
	group, K    int
}

// scaleAt is the f32 value of canonical-order scale i.
func (l int4Layout) scaleAt(i int) float32 {
	if l.wS != nil {
		return l.wS[i]
	}
	return f16ToF32(l.wS16[i])
}

// row4ScaleAt is the f32 value of row4-interleaved scale i.
func (l int4Layout) row4ScaleAt(i int) float32 {
	if l.r4S != nil {
		return l.r4S[i]
	}
	return f16ToF32(l.r4S16[i])
}

func (l int4Layout) row(n int, codes []int8, scales []float32) {
	K, group := l.K, l.group
	nG := (K + group - 1) / group
	bpr := (K + 1) / 2
	switch {
	case l.w4 != nil:
		row := l.w4[n*bpr : (n+1)*bpr]
		for k := range K {
			b := row[k/2]
			if k&1 == 1 {
				b >>= 4
			}
			codes[k] = int8(b&0x0F) - 8
		}
		if l.wS != nil {
			copy(scales, l.wS[n*nG:(n+1)*nG])
		} else {
			widenF16(scales[:nG], l.wS16[n*nG:(n+1)*nG])
		}
	case l.sh != nil, l.r4 != nil:
		var q, r int
		if l.r4 != nil {
			q, r = n/4, n%4
		}
		for g := range nG {
			var chunk []byte
			if l.sh != nil {
				chunk = l.sh[n*bpr+g*16 : n*bpr+g*16+16]
				scales[g] = l.scaleAt(n*nG + g)
			} else {
				base := q*4*bpr + g*64 + r*16
				chunk = l.r4[base : base+16]
				scales[g] = l.row4ScaleAt(q*4*nG + 4*g + r)
			}
			gk := g * 32
			for k := gk; k < min(gk+32, K); k++ {
				kl := k - gk
				var nib byte
				if kl < 16 {
					nib = chunk[kl] & 0x0F
				} else {
					nib = chunk[kl-16] >> 4
				}
				codes[k] = int8(nib) - 8
			}
		}
	default:
		panic("linalg: int4Layout: no layout")
	}
}

// matmulW4A8GroupedRef is the W4A8 matmul with per-group activation scales over any int4 layout:
// dst[m,n] = Σ_seg aS[m,·]·wS[n,·]·Σ_{k∈seg} aq[m,k]·code[n,k]. Mathematically the same product the
// per-row kernels compute, with the activation scale moved inside the sum.
func matmulW4A8GroupedRef(ws *Workspace, g int, a []float32, l int4Layout, dst []float32, M, N int) {
	K := l.K
	if l.sh != nil || l.r4 != nil {
		l.group = 32
	}
	checkGroups(g, l.group)
	aq, aS, nG := quantizeActGrouped(ws, g, a, M, K)
	wG := (K + l.group - 1) / l.group
	refParallel(ws, N, func(n0, n1 int) {
		codes := make([]int8, K)
		wS := make([]float32, wG)
		for n := n0; n < n1; n++ {
			l.row(n, codes, wS)
			for m := range M {
				dst[m*N+n] = groupedDot(aq[m*K:m*K+K], aS[m*nG:m*nG+nG], g, codes, wS, l.group, K)
			}
		}
	})
}

// int4Layout describes whichever int4 layout w holds, canonical first (matching Row's order).
func (w *WeightMat) int4Layout() int4Layout {
	return int4Layout{w4: w.q4, sh: w.q4SplitHalf, r4: w.q4Row4, wS16: w.q4s16, r4S16: w.q4Row4Scales16, group: w.group, K: w.cols}
}

// matmulW4A8Grouped is the per-group W4A8 matmul every W4A8 hook calls. With per-32 activation scales
// over 32-wide weight groups it runs the existing SIMD kernels fed combined scales wS[j,g]*aS[i,g]:
// the arch's M=1 repacked kernel per activation row where the layout is present (amd64 split-half,
// arm64 row4), else the canonical dotW4A8. Anything else (other group sizes, ragged K) takes the
// Go reference.
func matmulW4A8Grouped(ws *Workspace, g int, a []float32, l int4Layout, dst []float32, M, N int) {
	K := l.K
	if g == 32 && K%32 == 0 && K >= 32 {
		if (l.sh != nil || l.r4 != nil) && w4a8GroupedFastRows(ws, a, l, dst, M, N) {
			return
		}
		if l.w4 != nil && l.group == 32 {
			matmulW4A8CanonicalGrouped(ws, a, l, dst, M, K, N)
			return
		}
	}
	matmulW4A8GroupedRef(ws, g, a, l, dst, M, N)
}

// matmulW4A8CanonicalGrouped runs the canonical-layout dotW4A8 kernel (every arch) with combined
// per-group scales, for per-32 activations over 32-wide weight groups.
func matmulW4A8CanonicalGrouped(ws *Workspace, a []float32, l int4Layout, dst []float32, M, K, N int) {
	aq, aS, nG := quantizeActGrouped(ws, 32, a, M, K)
	bpr := (K + 1) / 2
	w4 := l.w4
	parallelFor(ws, M*N*K, N, func(n0, n1 int) {
		cs := make([]float32, nG)
		srow := make([]float32, nG)
		for j := n0; j < n1; j++ {
			prow := w4[j*bpr : j*bpr+bpr]
			if l.wS != nil {
				copy(srow, l.wS[j*nG:j*nG+nG])
			} else {
				widenF16(srow, l.wS16[j*nG:j*nG+nG])
			}
			for i := range M {
				as := aS[i*nG : i*nG+nG]
				for g := range cs {
					cs[g] = srow[g] * as[g]
				}
				dst[i*N+j] = dotW4A8(aq[i*K:i*K+K], prow, cs, 32, K)
			}
		}
	})
}

// dotI8Scaled32Go is dotI8Scaled32's portable form and its test oracle.
func dotI8Scaled32Go(a, b []int8, aS []float32) float32 {
	var acc float32
	for g := range len(a) / 32 {
		var isum int32
		for k := g * 32; k < g*32+32; k++ {
			isum += int32(a[k]) * int32(b[k])
		}
		acc += float32(isum) * aS[g]
	}
	return acc
}

// matmulW8A8Grouped is the per-group W8A8 matmul the W8A8 hooks call: dotI8Scaled32 (AVX2 on amd64)
// for per-32 activations over a K that is a multiple of 32, the Go reference otherwise.
func matmulW8A8Grouped(ws *Workspace, g int, a []float32, bQ []int8, bScales, dst []float32, M, K, N int) {
	if g != 32 || K%32 != 0 {
		matmulW8A8GroupedRef(ws, g, a, bQ, bScales, dst, M, K, N)
		return
	}
	aq, aS, nG := quantizeActGrouped(ws, 32, a, M, K)
	span := func(n0, n1 int) {
		w8a8GroupedSpan(aq, aS, bQ, bScales, dst, M, K, N, nG, n0, n1)
	}
	// The workspace's own fan-out and threshold, as the per-row W8A8 path uses.
	parallelFor(ws, M*N*K, N, span)
}

// matmulW8A8GroupedBatch is MatmulBTW8A8Batch with per-group activation scales: quantizes activations
// ONCE for the whole batch and executes all ops under ONE parallel fan-out across totalN columns.
func matmulW8A8GroupedBatch(ws *Workspace, g int, a []float32, ops []W8A8Op, M, K, totalN int) {
	if g != 32 || K%32 != 0 {
		for _, op := range ops {
			matmulW8A8GroupedRef(ws, g, a, op.BQ, op.Scales, op.Dst, M, K, op.N)
		}
		return
	}
	aq, aS, nG := quantizeActGrouped(ws, 32, a, M, K)
	span := func(g0, g1 int) {
		base := 0
		for _, op := range ops {
			lo, hi := max(g0, base), min(g1, base+op.N)
			if lo < hi {
				w8a8GroupedSpan(aq, aS, op.BQ, op.Scales, op.Dst, M, K, op.N, nG, lo-base, hi-base)
			}
			base += op.N
		}
	}
	parallelFor(ws, M*totalN*K, totalN, span)
}

// SetActQuantGroup stamps this weight's activation group size: WeightMat's own matmul methods use it
// when the workspace sets none, and a consumer calling the free functions with this weight's raw
// arrays reads it back (ActQuantGroup) to set its workspace. Per weight, so per model.
func (w *WeightMat) SetActQuantGroup(g int) {
	if g < 0 {
		panic(fmt.Sprintf("linalg: WeightMat.SetActQuantGroup(%d): group must be >= 0", g))
	}
	w.actGroup = g
}

// ActQuantGroup reports the group SetActQuantGroup stamped on this weight (0 = none).
func (w *WeightMat) ActQuantGroup() int { return w.actGroup }

// groupFor is the activation group a matmul of w through ws uses: the workspace's own, else the
// weight's, else the process-wide one.
func (w *WeightMat) groupFor(ws *Workspace) int {
	if ws != nil && ws.actGroup > 0 {
		return ws.actGroup
	}
	if w.actGroup > 0 {
		return w.actGroup
	}
	return actQuantGroup
}

// withWeightGroup runs fn with ws carrying w's group when ws sets none, so a WeightMat method that
// calls a free function (which reads only the workspace) honours the weight's stamp.
func (w *WeightMat) withWeightGroup(ws *Workspace, fn func(*Workspace)) {
	if w.actGroup > 0 && (ws == nil || ws.actGroup == 0) {
		if ws == nil {
			ws = new(Workspace)
		}
		ws.actGroup = w.actGroup
		defer func() { ws.actGroup = 0 }()
	}
	fn(ws)
}

// parallelFor is Workspace.parallelCols (the per-row kernels' fan-out and threshold), tolerating a
// nil workspace.
func parallelFor(ws *Workspace, work, N int, fn func(j0, j1 int)) {
	if ws == nil {
		ws = new(Workspace)
	}
	ws.parallelCols(work, N, fn)
}
