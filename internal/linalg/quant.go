package linalg

import "math"

// Per-row symmetric int8 weight quantization (gemma-decoder-plan §8 / M8).
// Each output row (channel) of a [rows, cols] weight matrix gets its own f32
// scale; the symmetric [-127,127] range keeps zero at zero with no zero-point
// bookkeeping. This is the standard per-channel scheme bitsandbytes/GPTQ use.
// Halves memory vs the bf16 checkpoint (and quarters it vs the f32 we widen to
// on load), at a per-row reconstruction error the model tolerates.
//
// The decoder uses weight-only int8: weights are int8, activations stay f32,
// and the int8→f32 widen happens inside the matmul (MatmulBTQ8) — the win is
// the 2–4× smaller weight footprint and the memory bandwidth it saves.

// QuantizeRowsInt8 quantizes a [rows, cols] f32 matrix (row-major) to int8
// weights + per-row f32 scales. Reconstruct: W[i,j] ≈ float32(q[i*cols+j]) *
// scales[i]. An all-zero row gets scale 1 (its codes are all zero anyway).
func QuantizeRowsInt8(w []float32, rows, cols int) (q []int8, scales []float32) {
	if rows*cols != len(w) {
		panic("linalg: QuantizeRowsInt8 shape mismatch")
	}
	q = make([]int8, rows*cols)
	scales = make([]float32, rows)
	for i := range rows {
		row := w[i*cols : (i+1)*cols]
		var maxAbs float32
		for _, v := range row {
			if v < 0 {
				v = -v
			}
			if v > maxAbs {
				maxAbs = v
			}
		}
		if maxAbs == 0 {
			scales[i] = 1
			continue
		}
		s := maxAbs / 127.0
		scales[i] = s
		inv := 1.0 / s
		off := i * cols
		for j, v := range row {
			x := math.Round(float64(v * inv))
			if x > 127 {
				x = 127
			} else if x < -127 {
				x = -127
			}
			q[off+j] = int8(x)
		}
	}
	return q, scales
}

// DequantizeRowInt8 reconstructs one row into dst: dst[j] = float32(q[j])*scale.
// Used for the tied embedding lookup when the table is stored int8.
func DequantizeRowInt8(q []int8, scale float32, dst []float32) {
	for j, c := range q {
		dst[j] = float32(c) * scale
	}
}

// MatmulBTQ8 computes dst[M,N] = a[M,K] · bᵀ where b is the [N,K] matrix stored
// as int8 rows bQ + per-row f32 scales bScales (b[j,k] ≈ float32(bQ[j,k]) *
// bScales[j]). Mirrors MatmulBT: a-row · b-row dot, widened int8→f32 in the
// inner loop, scaled per row at write-back, parallelized over the N columns.
func MatmulBTQ8(a []float32, bQ []int8, bScales []float32, dst []float32, M, K, N int) {
	parallelCols(M*N*K, N, func(j0, j1 int) {
		for i := range M {
			arow := a[i*K : i*K+K]
			drow := dst[i*N : i*N+N]
			for j := j0; j < j1; j++ {
				bq := bQ[j*K : j*K+K]
				var s float32
				for k := range K {
					s += arow[k] * float32(bq[k])
				}
				drow[j] = s * bScales[j]
			}
		}
	})
}
