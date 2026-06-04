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
		scales[i] = QuantizeRowInt8(w[i*cols:(i+1)*cols], q[i*cols:(i+1)*cols])
	}
	return q, scales
}

// QuantizeRowInt8 quantizes one f32 row into q (len cols) and returns its scale —
// the single-row core of QuantizeRowsInt8 (bit-identical), exposed so a loader can
// quantize each row as it is dequantized, without buffering the whole f32 matrix.
// An all-zero row gets scale 1.
func QuantizeRowInt8(row []float32, q []int8) (scale float32) {
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
		for j := range q {
			q[j] = 0
		}
		return 1
	}
	s := maxAbs / 127.0
	inv := 1.0 / s
	for j, v := range row {
		x := math.Round(float64(v * inv))
		if x > 127 {
			x = 127
		} else if x < -127 {
			x = -127
		}
		q[j] = int8(x)
	}
	return s
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
// bScales[j]). Each output row is widened int8→f32 into a reused scratch buffer,
// then the SIMD dotF32 kernel (AVX2/NEON — the primitive MatmulBT uses) runs over
// the whole row and the per-row scale is applied at write-back. Only the cheap
// int8→f32 widen stays scalar; the multiply-accumulate is vectorized. The scratch
// is one row wide and allocated once per worker. Parallelized over the N columns.
func MatmulBTQ8(a []float32, bQ []int8, bScales []float32, dst []float32, M, K, N int) {
	parallelCols(M*N*K, N, func(j0, j1 int) {
		deq := make([]float32, K) // per-worker scratch: one widened b-row
		for i := range M {
			arow := a[i*K : i*K+K]
			drow := dst[i*N : i*N+N]
			for j := j0; j < j1; j++ {
				bq := bQ[j*K : j*K+K]
				for k := range K {
					deq[k] = float32(bq[k])
				}
				drow[j] = dotF32(arow, deq) * bScales[j]
			}
		}
	})
}

// dotI8Scalar returns Σ a[i]*b[i] as an int32 over two int8 vectors (the products
// fit: 127*127*K stays well within int32 for transformer K). It is the portable
// reference and the tail/fallback for the SIMD dotI8 dispatcher (see the per-arch
// quant_i8_*.go).
func dotI8Scalar(a, b []int8) int32 {
	var s int32
	for k := range a {
		s += int32(a[k]) * int32(b[k])
	}
	return s
}

// quantizeRowInt8 dynamically quantizes one f32 activation row to int8 with a
// single symmetric scale (maxabs/127). Returns the codes (into dst) and the
// scale; an all-zero row gets scale 0 and zero codes.
func quantizeRowInt8(a []float32, dst []int8) (scale float32) {
	var maxAbs float32
	for _, v := range a {
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs == 0 {
		for k := range dst {
			dst[k] = 0
		}
		return 0
	}
	scale = maxAbs / 127
	inv := 1.0 / scale
	for k, v := range a {
		x := math.Round(float64(v * inv))
		if x > 127 {
			x = 127
		} else if x < -127 {
			x = -127
		}
		dst[k] = int8(x)
	}
	return scale
}

// MatmulBTW8A8 computes dst[M,N] = a[M,K] · bᵀ as full int8×int8→int32 (W8A8):
// the f32 activation row is quantized to int8 on the fly (dynamic per-row scale),
// the integer dot accumulates in int32, and the result is rescaled by the
// activation scale × the per-row weight scale. Unlike MatmulBTQ8 (weight-only
// int8, f32 activations) this also quantizes the activations, so it is lossier —
// the tradeoff for an integer kernel. Parallelized over the N columns.
func MatmulBTW8A8(a []float32, bQ []int8, bScales []float32, dst []float32, M, K, N int) {
	parallelCols(M*N*K, N, func(j0, j1 int) {
		aq := make([]int8, K) // per-worker scratch: the quantized activation row
		for i := range M {
			aScale := quantizeRowInt8(a[i*K:i*K+K], aq)
			drow := dst[i*N : i*N+N]
			if aScale == 0 {
				for j := j0; j < j1; j++ {
					drow[j] = 0
				}
				continue
			}
			for j := j0; j < j1; j++ {
				acc := dotI8(aq, bQ[j*K:j*K+K])
				drow[j] = float32(acc) * aScale * bScales[j]
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
		QuantizeGroupInt4Row(w[i*cols:(i+1)*cols], cols, group, packed[i*bpr:(i+1)*bpr], scales[i*nGroups:(i+1)*nGroups])
	}
	return packed, scales
}

// QuantizeGroupInt4Row quantizes one f32 row into packed (len (cols+1)/2) +
// per-group scales (len ⌈cols/group⌉) — the single-row core of QuantizeGroupsInt4
// (bit-identical), exposed so a loader can quantize each row as it is dequantized,
// without buffering the whole f32 matrix. packed is assumed zeroed on entry (a
// fresh per-row slice).
func QuantizeGroupInt4Row(row []float32, cols, group int, packed []byte, scales []float32) {
	nGroups := (cols + group - 1) / group
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
		scales[g] = s
		inv := 1.0 / s
		for k := ks; k < ke; k++ {
			q := int(math.Round(float64(row[k] * inv)))
			if q > 7 {
				q = 7
			} else if q < -7 {
				q = -7
			}
			nib := byte(q + 8) // [1,15]; 8 = zero
			bi := k / 2
			if k&1 == 0 {
				packed[bi] = (packed[bi] &^ 0x0F) | (nib & 0x0F)
			} else {
				packed[bi] = (packed[bi] &^ 0xF0) | (nib << 4)
			}
		}
	}
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
