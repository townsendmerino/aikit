package linalg

import "fmt"

// Q4_K weights, kept in GGUF's own super-block layout (goinfer
// docs/tasks/task-int4-weight-quality-2026-09.md). A model file's Q4_K tensor is already 4-bit and
// asymmetric: per 32-element sub-block, w = d·sc·q − dmin·m with q ∈ [0, 15]. Re-quantizing it to the
// symmetric int4 WeightMat kind rounds a second time, and that measured as the int4 quality loss.
// This kind stores the file's bytes verbatim and multiplies them exactly.
//
// Layout per 256 weights (144 bytes): d (f16), dmin (f16), scales[12] (eight 6-bit scales and eight
// 6-bit mins, packed as ggml's get_scale_min_k4 reads them), qs[128]. Byte qs[32j+l] holds element
// 64j+l in its low nibble and element 64j+32+l in its high nibble.
//
// Activations are always quantized per 32 (one int8 scale per group), whatever the workspace's
// activation group says: a sub-block is 32 weights, so each activation group meets exactly one
// weight scale and one minimum. Per sub-block g of row n:
//
//	Σ_k w·a ≈ aS_g · (d·sc_g · Σ q·aq − dmin·m_g · Σ aq)
//
// with Σ aq computed once per activation row.
//
// PRIOR ART: kquant.go holds a native Q4_K/Q6_K × Q8_K matmul, kept as a negative result and never wired
// into WeightMat. Its bar was SPEED (≥ 1.3× over the int8 W8A8 path), which Q6_K cannot reach by bytes
// alone. This kind exists for QUALITY, and it is measured against the symmetric int4 kind, which reads
// the same bytes. It also differs in activations: per-32 groups here, one scale per 256 (Q8_K) there,
// because per-32 is what H2's outlier fix and the Phase 0 quality measurement used. It reuses kquant.go's
// block helpers (f16ToF32, q4kScaleMin, u16le). The scalar unpack there measured 20-45× slower than W8A8,
// so dotQ4KGo is an oracle, and speed needs the arch kernel.

// q4kGroup is the activation group, and the Q4_K sub-block: 32 weights share one scale and minimum.
const q4kGroup = 32

// Q4KRowBytes is the storage one row of cols Q4_K weights occupies (cols must be a multiple of 256).
func Q4KRowBytes(cols int) int { return cols / qkK * q4kBlockB }

// WrapQ4K wraps rows×cols weights already in GGUF Q4_K layout WITHOUT copying: raw may point into an
// mmap'd file, and the caller keeps it alive. cols must be a multiple of 256 (GGUF requires it of a
// Q4_K tensor's rows) and len(raw) must be exactly rows·cols/256·144.
func WrapQ4K(raw []byte, rows, cols int) (WeightMat, error) {
	if rows <= 0 || cols <= 0 || cols%qkK != 0 {
		return WeightMat{}, fmt.Errorf("linalg: WrapQ4K: shape %dx%d (cols must be a positive multiple of 256)", rows, cols)
	}
	if want := mul(rows, Q4KRowBytes(cols)); len(raw) != want {
		return WeightMat{}, fmt.Errorf("linalg: WrapQ4K: %d bytes for %dx%d, want %d", len(raw), rows, cols, want)
	}
	return WeightMat{q4k: raw, rows: rows, cols: cols}, nil
}

// Q4K returns the raw Q4_K super-blocks (ok=false unless this is a Q4_K WeightMat), for a consumer's
// GPU export.
func (w *WeightMat) Q4K() (raw []byte, ok bool) { return w.q4k, w.q4k != nil }

// dequantQ4KRow writes one row's weights (cols of them) into dst.
func dequantQ4KRow(row []byte, cols int, dst []float32) {
	for b := range cols / qkK {
		blk := row[b*q4kBlockB : (b+1)*q4kBlockB]
		d := f16ToF32(u16le(blk[0:]))
		dmin := f16ToF32(u16le(blk[2:]))
		scales, qs := blk[4:16], blk[16:144]
		out := dst[b*qkK : (b+1)*qkK]
		for j := range 4 {
			sc1, m1 := q4kScaleMin(2*j, scales)
			sc2, m2 := q4kScaleMin(2*j+1, scales)
			d1, o1 := d*float32(sc1), dmin*float32(m1)
			d2, o2 := d*float32(sc2), dmin*float32(m2)
			q := qs[32*j : 32*j+32]
			for l := range 32 {
				out[64*j+l] = d1*float32(q[l]&0x0F) - o1
				out[64*j+32+l] = d2*float32(q[l]>>4) - o2
			}
		}
	}
}

// q4kActSums fills sums[m*nG+g] = Σ aq over row m's group g (the per-group minimum's multiplier).
func q4kActSums(aq []int8, sums []int32, M, K int) {
	nG := K / q4kGroup
	for m := range M {
		for g := range nG {
			var s int32
			for _, v := range aq[m*K+g*q4kGroup : m*K+(g+1)*q4kGroup] {
				s += int32(v)
			}
			sums[m*nG+g] = s
		}
	}
}

// dotQ4KGo is one row's Q4_K × per-32-int8 dot, the portable form and the kernels' oracle.
func dotQ4KGo(row []byte, aq []int8, aS []float32, sums []int32, K int) float32 {
	var acc float32
	for b := range K / qkK {
		blk := row[b*q4kBlockB : (b+1)*q4kBlockB]
		d := f16ToF32(u16le(blk[0:]))
		dmin := f16ToF32(u16le(blk[2:]))
		scales, qs := blk[4:16], blk[16:144]
		for j := range 4 {
			q := qs[32*j : 32*j+32]
			lo := aq[b*qkK+64*j : b*qkK+64*j+32]
			hi := aq[b*qkK+64*j+32 : b*qkK+64*j+64]
			var dl, dh int32
			for l := range 32 {
				dl += int32(q[l]&0x0F) * int32(lo[l])
				dh += int32(q[l]>>4) * int32(hi[l])
			}
			g := b*8 + 2*j
			sc1, m1 := q4kScaleMin(2*j, scales)
			sc2, m2 := q4kScaleMin(2*j+1, scales)
			acc += aS[g] * (d*float32(sc1)*float32(dl) - dmin*float32(m1)*float32(sums[g]))
			acc += aS[g+1] * (d*float32(sc2)*float32(dh) - dmin*float32(m2)*float32(sums[g+1]))
		}
	}
	return acc
}

// matmulQ4K computes dst[M, N] = a[M, K] · W[N, K]ᵀ for Q4_K weights (raw: N rows of
// Q4KRowBytes(K)), with per-32 int8 activations. Columns fan out over the workspace.
func matmulQ4K(ws *Workspace, a []float32, raw []byte, dst []float32, M, K, N int) {
	if ws == nil {
		ws = new(Workspace)
	}
	aq, aS, nG := quantizeActGrouped(ws, q4kGroup, a, M, K)
	sums := ws.int32Buf(M * nG)
	q4kActSums(aq, sums, M, K)
	asumf := ws.q4kSumBuf(M * nG)
	for i := range asumf {
		asumf[i] = aS[i] * float32(sums[i])
	}
	rb := Q4KRowBytes(K)
	parallelFor(ws, M*N*K, N, func(n0, n1 int) {
		for n := n0; n < n1; n++ {
			row := raw[n*rb : (n+1)*rb]
			for m := range M {
				g := m * nG
				dst[m*N+n] = dotQ4K(row, aq[m*K:m*K+K], aS[g:g+nG], sums[g:g+nG], asumf[g:g+nG], K)
			}
		}
	})
}
