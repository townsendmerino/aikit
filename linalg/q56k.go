package linalg

import (
	"fmt"

	"github.com/townsendmerino/aikit/internal/half"
	"github.com/townsendmerino/aikit/linalg/internal/shape"
)

// Q5_K and Q6_K weights, kept in GGUF's own super-block layout: the Q4_K kind's siblings (q4k.go). A
// Q4_K_M file stores some tensors at these types; widening them to int8 reads more bytes than the
// file holds and rounds a second time. Activations are per-32 int8, as for Q4_K.
//
// Q5_K, 176 bytes per 256 weights: d, dmin (f16), scales[12] (Q4_K's packing), qh[32], qs[128].
// Element 64j+l is the low nibble of qs[32j+l] plus 16 when bit 2j of qh[l] is set; element
// 64j+32+l is the high nibble plus 16 for bit 2j+1. w = d·sc·q − dmin·m with q ∈ [0, 31], one scale
// and minimum per 32-element sub-block, so the dot is Q4_K's:
//
//	Σ_k w·a ≈ aS_g · (d·sc_g · Σ q·aq − dmin·m_g · Σ aq)
//
// Q6_K, 210 bytes per 256 weights: ql[128], qh[64], scales[16] (int8), d (f16, last). For half h and
// l < 32, with L = ql[64h:] and H = qh[32h:], the codes of elements 128h + {l, 32+l, 64+l, 96+l} are
// L[l]&15, L[l+32]&15, L[l]>>4 and L[l+32]>>4, each OR'd with the next two bits of H[l] (bits 0-1,
// 2-3, 4-5, 6-7) shifted to bits 4-5; w = d·sc[k/16]·(q − 32). A 32-element activation group meets
// two scales, so each group's dot is the integer Σ_s sc_s · Σ (q−32)·aq over its two halves, exact
// in int32, and then aS_g · d · that.
//
// The arch kernels are on amd64 (AVX2 + F16C, q56k_amd64.s); elsewhere the Go oracle runs, which is
// far slower than the int8 kinds (KQuantNative).

// q5kBlockB is a Q5_K super-block's size in bytes.
const q5kBlockB = 176

// Q5KRowBytes is the storage one row of cols Q5_K weights occupies (cols must be a multiple of 256).
func Q5KRowBytes(cols int) int { return cols / qkK * q5kBlockB }

// Q6KRowBytes is the storage one row of cols Q6_K weights occupies (cols must be a multiple of 256).
func Q6KRowBytes(cols int) int { return cols / qkK * q6kBlockB }

// KQuantNative reports whether the Q4_K, Q5_K and Q6_K kinds run on an arch kernel on this CPU. When
// false they run the portable Go dot, many times slower than the int8 kinds, and a consumer choosing
// a kind for speed should widen instead.
func KQuantNative() bool { return kquantNative() }

func wrapKQuant(fn string, raw []byte, rows, cols, rowBytes int) error {
	if rows <= 0 || cols <= 0 || cols%qkK != 0 {
		return fmt.Errorf("linalg: %s: shape %dx%d (cols must be a positive multiple of 256)", fn, rows, cols)
	}
	if want := shape.Mul(rows, rowBytes); len(raw) != want {
		return fmt.Errorf("linalg: %s: %d bytes for %dx%d, want %d", fn, len(raw), rows, cols, want)
	}
	return nil
}

// WrapQ5K wraps rows×cols weights already in GGUF Q5_K layout WITHOUT copying (raw may point into an
// mmap'd file, and the caller keeps it alive). cols must be a multiple of 256 and len(raw) exactly
// rows·cols/256·176.
func WrapQ5K(raw []byte, rows, cols int) (WeightMat, error) {
	if err := wrapKQuant("WrapQ5K", raw, rows, cols, Q5KRowBytes(cols)); err != nil {
		return WeightMat{}, err
	}
	return WeightMat{kq: raw, kqKind: kqQ5K, rows: rows, cols: cols}, nil
}

// WrapQ6K wraps rows×cols weights already in GGUF Q6_K layout WITHOUT copying (raw may point into an
// mmap'd file, and the caller keeps it alive). cols must be a multiple of 256 and len(raw) exactly
// rows·cols/256·210.
func WrapQ6K(raw []byte, rows, cols int) (WeightMat, error) {
	if err := wrapKQuant("WrapQ6K", raw, rows, cols, Q6KRowBytes(cols)); err != nil {
		return WeightMat{}, err
	}
	return WeightMat{kq: raw, kqKind: kqQ6K, rows: rows, cols: cols}, nil
}

// Q5K returns the raw Q5_K super-blocks (ok=false unless this is a Q5_K WeightMat).
func (w *WeightMat) Q5K() (raw []byte, ok bool) { return w.kquantRaw(kqQ5K) }

// Q6K returns the raw Q6_K super-blocks (ok=false unless this is a Q6_K WeightMat).
func (w *WeightMat) Q6K() (raw []byte, ok bool) { return w.kquantRaw(kqQ6K) }

// The K-quant kinds a WeightMat's kq holds (kqKind). Zero is none of them.
const (
	kqQ4K uint8 = 4
	kqQ5K uint8 = 5
	kqQ6K uint8 = 6
)

// kquantRaw returns kq when it holds kind k.
func (w *WeightMat) kquantRaw(k uint8) ([]byte, bool) {
	if w.kq == nil || w.kqKind != k {
		return nil, false
	}
	return w.kq, true
}

// kquantKind is Kind's label for the K-quant kinds.
func (w *WeightMat) kquantKind() Kind {
	switch w.kqKind {
	case kqQ5K:
		return KindQ5K
	case kqQ6K:
		return KindQ6K
	}
	return KindQ4K
}

// matmulKQuant is MatmulBT / MatmulBTInto for the K-quant kinds; ws may be nil.
func (w *WeightMat) matmulKQuant(ws *Workspace, a, dst []float32, M int) {
	switch w.kqKind {
	case kqQ5K:
		matmulQ5K(ws, a, w.kq, dst, M, w.cols, w.rows)
	case kqQ6K:
		matmulQ6K(ws, a, w.kq, dst, M, w.cols, w.rows)
	default:
		matmulQ4K(ws, a, w.kq, dst, M, w.cols, w.rows)
	}
}

// kquantRow is Row for the K-quant kinds.
func (w *WeightMat) kquantRow(i int, dst []float32) {
	switch w.kqKind {
	case kqQ5K:
		rb := Q5KRowBytes(w.cols)
		dequantQ5KRow(w.kq[i*rb:(i+1)*rb], w.cols, dst)
	case kqQ6K:
		rb := Q6KRowBytes(w.cols)
		dequantQ6KRow(w.kq[i*rb:(i+1)*rb], w.cols, dst)
	default:
		rb := Q4KRowBytes(w.cols)
		dequantQ4KRow(w.kq[i*rb:(i+1)*rb], w.cols, dst)
	}
}

// q5kCode is element 64j+l (hi=false) or 64j+32+l (hi=true) of a Q5_K block's codes.
func q5kCode(qh, qs []byte, j, l int, hi bool) uint8 {
	q := qs[32*j+l]
	if hi {
		return q>>4 | (qh[l]>>(2*j+1)&1)<<4
	}
	return q&0x0F | (qh[l]>>(2*j)&1)<<4
}

// q6kCode is the unsigned code (q+32 ∈ [0, 63]) of element 128h + 32c + l of a Q6_K block.
func q6kCode(blk []byte, h, c, l int) uint8 {
	L, H := blk[64*h:], blk[128+32*h:]
	var lo uint8
	switch c {
	case 0:
		lo = L[l] & 0x0F
	case 1:
		lo = L[l+32] & 0x0F
	case 2:
		lo = L[l] >> 4
	default:
		lo = L[l+32] >> 4
	}
	return lo | (H[l]>>(2*c)&3)<<4
}

// dequantQ5KRow writes one row's weights (cols of them) into dst.
func dequantQ5KRow(row []byte, cols int, dst []float32) {
	for b := range cols / qkK {
		blk := row[b*q5kBlockB : (b+1)*q5kBlockB]
		d := half.ToF32(u16le(blk[0:]))
		dmin := half.ToF32(u16le(blk[2:]))
		scales, qh, qs := blk[4:16], blk[16:48], blk[48:176]
		out := dst[b*qkK : (b+1)*qkK]
		for j := range 4 {
			sc1, m1 := q4kScaleMin(2*j, scales)
			sc2, m2 := q4kScaleMin(2*j+1, scales)
			d1, o1 := d*float32(sc1), dmin*float32(m1)
			d2, o2 := d*float32(sc2), dmin*float32(m2)
			for l := range 32 {
				out[64*j+l] = d1*float32(q5kCode(qh, qs, j, l, false)) - o1
				out[64*j+32+l] = d2*float32(q5kCode(qh, qs, j, l, true)) - o2
			}
		}
	}
}

// dequantQ6KRow writes one row's weights (cols of them) into dst.
func dequantQ6KRow(row []byte, cols int, dst []float32) {
	for b := range cols / qkK {
		blk := row[b*q6kBlockB : (b+1)*q6kBlockB]
		d := half.ToF32(u16le(blk[208:]))
		out := dst[b*qkK : (b+1)*qkK]
		for h := range 2 {
			for c := range 4 {
				for l := range 32 {
					k := 128*h + 32*c + l
					out[k] = d * float32(int8(blk[192+k/16])) * float32(int32(q6kCode(blk, h, c, l))-32)
				}
			}
		}
	}
}

// dotQ5KGo is one row's Q5_K × per-32-int8 dot, the portable form and the kernel's oracle.
func dotQ5KGo(row []byte, aq []int8, aS []float32, sums []int32, K int) float32 {
	var acc float32
	for b := range K / qkK {
		blk := row[b*q5kBlockB : (b+1)*q5kBlockB]
		d := half.ToF32(u16le(blk[0:]))
		dmin := half.ToF32(u16le(blk[2:]))
		scales, qh, qs := blk[4:16], blk[16:48], blk[48:176]
		for j := range 4 {
			lo := aq[b*qkK+64*j : b*qkK+64*j+32]
			hi := aq[b*qkK+64*j+32 : b*qkK+64*j+64]
			var dl, dh int32
			for l := range 32 {
				dl += int32(q5kCode(qh, qs, j, l, false)) * int32(lo[l])
				dh += int32(q5kCode(qh, qs, j, l, true)) * int32(hi[l])
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

// dotQ6KGo is one row's Q6_K × per-32-int8 dot, the portable form and the kernel's oracle.
func dotQ6KGo(row []byte, aq []int8, aS []float32, K int) float32 {
	var acc float32
	for b := range K / qkK {
		blk := row[b*q6kBlockB : (b+1)*q6kBlockB]
		d := half.ToF32(u16le(blk[208:]))
		for h := range 2 {
			for c := range 4 {
				a := aq[b*qkK+128*h+32*c:]
				var s0, s1 int32
				for l := range 16 {
					s0 += (int32(q6kCode(blk, h, c, l)) - 32) * int32(a[l])
					s1 += (int32(q6kCode(blk, h, c, l+16)) - 32) * int32(a[l+16])
				}
				sc := blk[192+8*h+2*c:]
				isum := int32(int8(sc[0]))*s0 + int32(int8(sc[1]))*s1
				acc += aS[b*8+4*h+c] * d * float32(isum)
			}
		}
	}
	return acc
}

// matmulQ5K computes dst[M, N] = a[M, K] · W[N, K]ᵀ for Q5_K weights (raw: N rows of
// Q5KRowBytes(K)), with per-32 int8 activations. Columns fan out over the workspace.
func matmulQ5K(ws *Workspace, a []float32, raw []byte, dst []float32, M, K, N int) {
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
	rb := Q5KRowBytes(K)
	parallelFor(ws, M*N*K, N, func(n0, n1 int) {
		for n := n0; n < n1; n++ {
			row := raw[n*rb : (n+1)*rb]
			for m := range M {
				g := m * nG
				dst[m*N+n] = dotQ5K(row, aq[m*K:m*K+K], aS[g:g+nG], sums[g:g+nG], asumf[g:g+nG], K)
			}
		}
	})
}

// matmulQ6K computes dst[M, N] = a[M, K] · W[N, K]ᵀ for Q6_K weights (raw: N rows of
// Q6KRowBytes(K)), with per-32 int8 activations. Columns fan out over the workspace.
func matmulQ6K(ws *Workspace, a []float32, raw []byte, dst []float32, M, K, N int) {
	if ws == nil {
		ws = new(Workspace)
	}
	aq, aS, nG := quantizeActGrouped(ws, q4kGroup, a, M, K)
	rb := Q6KRowBytes(K)
	parallelFor(ws, M*N*K, N, func(n0, n1 int) {
		for n := n0; n < n1; n++ {
			row := raw[n*rb : (n+1)*rb]
			for m := range M {
				g := m * nG
				dst[m*N+n] = dotQ6K(row, aq[m*K:m*K+K], aS[g:g+nG], K)
			}
		}
	})
}
