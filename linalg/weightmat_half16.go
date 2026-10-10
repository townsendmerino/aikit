package linalg

import (
	"fmt"
	"math"
	"sync"
)

// 16-bit weight storage: a WeightMat holding its [rows, cols] weights as bf16 or IEEE binary16 bit patterns, half the bytes of f32.
//
// The matmul widens a tile of half16Tile weight rows to f32 in a scratch buffer and runs the SAME blocked f32 kernel over the tile, so
// each output is bit-identical to MatmulBT over the widened weights (widening is exact, and the f32 kernel's per-output result does not
// depend on which 8-aligned column group of the tile it sits in: the width contract in MatmulBT's comment, TestParallelWidth_bitIdentical).
// The cost over f32 is the widen, which the f32 path does not pay; the saving is resident bytes and memory traffic.
//
// CPU only: a consumer with an accelerator dispatches through the raw accessors and F32() reports !ok for this kind.

const (
	half16None uint8 = iota
	half16BF16
	half16F16
)

// half16Tile is the weight rows widened at a time. A multiple of 8 so every tile but a shard's last is a whole number of the f32
// kernel's 8-column groups (the last tile can be shorter only where the matrix itself ends or the shard does, as in the f32 path).
const half16Tile = 64

type half16Scratch struct{ w, out []float32 }

var half16Pool = sync.Pool{New: func() any { return new(half16Scratch) }}

func growF32(s []float32, n int) []float32 {
	if cap(s) < n {
		return make([]float32, n)
	}
	return s[:n]
}

// F32ToBF16 rounds f to bfloat16, nearest-even; a NaN stays a NaN.
func F32ToBF16(f float32) uint16 {
	b := math.Float32bits(f)
	if b&0x7fffffff > 0x7f800000 { // NaN: keep it a NaN with the top mantissa bit set
		return uint16(b>>16) | 0x0040
	}
	b += 0x7fff + (b>>16)&1
	return uint16(b >> 16)
}

// BF16ToF32 widens a bfloat16 bit pattern to f32. Exact.
func BF16ToF32(u uint16) float32 { return math.Float32frombits(uint32(u) << 16) }

func (w *WeightMat) widenHalf16(dst []float32, src []uint16) {
	if w.h16Kind == half16BF16 {
		widenBF16(dst, src)
		return
	}
	F16ToF32Slice(dst, src)
}

// WrapBF16 wraps an existing [rows, cols] bf16 weight (raw bit patterns) WITHOUT copying; the caller keeps u alive.
// Panics if a dimension is negative or len(u) != rows*cols.
func WrapBF16(u []uint16, rows, cols int) WeightMat {
	return wrapHalf16("WrapBF16", half16BF16, u, rows, cols)
}

// WrapF16 is WrapBF16 for IEEE binary16 bit patterns.
func WrapF16(u []uint16, rows, cols int) WeightMat {
	return wrapHalf16("WrapF16", half16F16, u, rows, cols)
}

func wrapHalf16(name string, kind uint8, u []uint16, rows, cols int) WeightMat {
	if rows < 0 || cols < 0 {
		panic(fmt.Sprintf("linalg: %s negative dim (rows=%d cols=%d)", name, rows, cols))
	}
	requireExactLen(name, "u", len(u), mul(rows, cols))
	return WeightMat{h16: u, h16Kind: kind, rows: rows, cols: cols}
}

// QuantizeBF16 rounds a [rows, cols] f32 weight to bf16 (nearest-even), not retaining the source. Exact for a weight that came from bf16.
func QuantizeBF16(w []float32, rows, cols int) WeightMat {
	requireExactLen("QuantizeBF16", "w", len(w), mul(rows, cols))
	u := make([]uint16, len(w))
	for i, f := range w {
		u[i] = F32ToBF16(f)
	}
	return WeightMat{h16: u, h16Kind: half16BF16, rows: rows, cols: cols}
}

// QuantizeF16 rounds a [rows, cols] f32 weight to binary16 (F32ToF16), not retaining the source.
func QuantizeF16(w []float32, rows, cols int) WeightMat {
	requireExactLen("QuantizeF16", "w", len(w), mul(rows, cols))
	u := make([]uint16, len(w))
	F32ToF16Slice(u, w)
	return WeightMat{h16: u, h16Kind: half16F16, rows: rows, cols: cols}
}

// Half16 returns the stored bit patterns and whether they are bf16 (else binary16); ok is false for every other kind.
func (w *WeightMat) Half16() (u []uint16, bf16, ok bool) {
	return w.h16, w.h16Kind == half16BF16, w.h16 != nil
}

// matmulHalf16 is MatmulBT / Workspace.MatmulBT for a 16-bit weight: dst[M, rows] = a[M, cols] · weightᵀ. ws may be nil (the process-wide
// fan-out). Overwrites dst.
func (w *WeightMat) matmulHalf16(ws *Workspace, a, dst []float32, M int) {
	K, N := w.cols, w.rows
	checkMatmulBT("MatmulBT", len(a), len(w.h16), len(dst), M, K, N)
	span := func(j0, j1 int) {
		sc := half16Pool.Get().(*half16Scratch)
		for t0 := j0; t0 < j1; t0 += half16Tile {
			t1 := min(t0+half16Tile, j1)
			n := t1 - t0
			sc.w = growF32(sc.w, n*K)
			sc.out = growF32(sc.out, M*n)
			w.widenHalf16(sc.w, w.h16[t0*K:t1*K])
			zeroSpanF32(sc.out)
			blockedFill(a, sc.w, sc.out, M, K, n, 0, n, mBlockDefault, nBlockDefault, kBlockDefault)
			for i := range M {
				copy(dst[i*N+t0:i*N+t1], sc.out[i*n:(i+1)*n])
			}
		}
		half16Pool.Put(sc)
	}
	if ws != nil {
		ws.parallelCols(M*N*K, N, span)
		return
	}
	parallelCols(M*N*K, N, span)
}
