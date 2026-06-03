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

// Group-wise symmetric int4 weight quantization (gemma-decoder-plan §8 / M8
// int4). Per-ROW int8 is too coarse at 4 bits, so each row is split into groups
// of `group` consecutive input features (along K), and each group gets its own
// f32 scale: W[i, g*group+e] ≈ (nibble-8) * scale[i,g], with the nibble a 4-bit
// code in [1,15] (8 = zero, symmetric range [-7,7]). Two nibbles pack per byte
// (even k = low nibble, odd k = high). At group 32 this is ~0.625 byte/element
// (4-bit code + the per-group scale amortized), ≈ 6.4× smaller than f32 and
// ~1.6× smaller than per-row int8 — the footprint that fits a 7B-class model in
// laptop RAM. The matmul (MatmulBTQ4) dequantizes per group inside the inner
// loop; activations stay f32.

// groupsFor returns the number of groups a K-wide row splits into (the final
// group is ragged when group does not divide K) and the packed bytes per row.
func groupsFor(cols, group int) (nGroups, bytesPerRow int) {
	return (cols + group - 1) / group, (cols + 1) / 2
}

// QuantizeGroupsInt4 quantizes a [rows, cols] f32 matrix (row-major) to packed
// 4-bit codes + per-group f32 scales. Reconstruct: W[i, k] ≈ (nibble(i,k)-8) *
// scales[i*nGroups + k/group]. An all-zero group gets scale 1 (codes all 8).
func QuantizeGroupsInt4(w []float32, rows, cols, group int) (packed []byte, scales []float32) {
	if rows*cols != len(w) {
		panic("linalg: QuantizeGroupsInt4 shape mismatch")
	}
	nGroups, bpr := groupsFor(cols, group)
	packed = make([]byte, rows*bpr)
	scales = make([]float32, rows*nGroups)
	for i := range rows {
		row := w[i*cols : (i+1)*cols]
		for g := range nGroups {
			ks := g * group
			ke := min(ks+group, cols)
			var maxAbs float32
			for k := ks; k < ke; k++ {
				if v := row[k]; v > maxAbs {
					maxAbs = v
				} else if -v > maxAbs {
					maxAbs = -v
				}
			}
			s := float32(1)
			if maxAbs > 0 {
				s = maxAbs / 7
			}
			scales[i*nGroups+g] = s
			inv := 1.0 / s
			for k := ks; k < ke; k++ {
				q := int(math.Round(float64(row[k] * inv)))
				if q > 7 {
					q = 7
				} else if q < -7 {
					q = -7
				}
				nib := byte(q + 8) // [1,15]; 8 = zero
				bi := i*bpr + k/2
				if k&1 == 0 {
					packed[bi] = (packed[bi] &^ 0x0F) | (nib & 0x0F)
				} else {
					packed[bi] = (packed[bi] &^ 0xF0) | (nib << 4)
				}
			}
		}
	}
	return packed, scales
}

// DequantizeRowInt4 reconstructs one row into dst[:cols] from its packed nibbles
// and per-group scales (both already sliced to the row). Used for the tied
// embedding lookup when the table is stored int4.
func DequantizeRowInt4(packed []byte, scales []float32, group, cols int, dst []float32) {
	for k := range cols {
		b := packed[k/2]
		var nib byte
		if k&1 == 0 {
			nib = b & 0x0F
		} else {
			nib = b >> 4
		}
		dst[k] = float32(int(nib)-8) * scales[k/group]
	}
}

// MatmulBTQ4 computes dst[M,N] = a[M,K] · bᵀ where b is the [N,K] matrix stored
// as group-wise int4 (bPacked nibbles + bScales per group; see
// QuantizeGroupsInt4). For each group it unpacks the 4-bit codes into a small
// reused scratch buffer as centered floats (nibble-8, no scale), runs the SIMD
// dotF32 kernel (AVX2/NEON — the same primitive MatmulBT uses) over the group,
// then applies that group's f32 scale and accumulates. So the float-heavy
// multiply-accumulate is vectorized; only the cheap nibble unpack stays scalar.
// Parallelized over the N columns; activations stay f32. The scratch is one
// group wide and allocated once per worker.
func MatmulBTQ4(a []float32, bPacked []byte, bScales []float32, dst []float32, M, K, N, group int) {
	nGroups, bpr := groupsFor(K, group)
	parallelCols(M*N*K, N, func(j0, j1 int) {
		deq := make([]float32, group) // per-worker scratch: one dequantized group
		for i := range M {
			arow := a[i*K : i*K+K]
			drow := dst[i*N : i*N+N]
			for j := j0; j < j1; j++ {
				prow := bPacked[j*bpr : j*bpr+bpr]
				srow := bScales[j*nGroups : j*nGroups+nGroups]
				var total float32
				for g := range nGroups {
					ks := g * group
					ke := min(ks+group, K)
					gsz := ke - ks
					for e := range gsz {
						k := ks + e
						b := prow[k>>1]
						nib := b & 0x0F
						if k&1 == 1 {
							nib = b >> 4
						}
						deq[e] = float32(int(nib) - 8)
					}
					total += dotF32(arow[ks:ke], deq[:gsz]) * srow[g]
				}
				drow[j] = total
			}
		}
	})
}
