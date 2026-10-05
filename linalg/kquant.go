package linalg

// Native K-quant integer-accumulation matmul: Q4_K/Q6_K weights × Q8_K activations.
//
// NEGATIVE RESULT — kept as a tested reference, NOT wired into WeightMat. The ≥1.3× decode gate is
// provably unreachable for Q6_K (byte-ratio ceiling 256/210 = 1.22× vs the int8 W8A8 path,
// regardless of unpack quality; case analysis in docs/internal/cpu-acceleration.md → "Native
// K-quant matmul — evaluated, NOT shipped"). These kernels are correct and bit-exact; they exist so
// the decision is reproducible (BenchmarkGEMV_*) and so a future revisit (e.g. a bandwidth-bound
// Q4_K variant, ceiling 1.78×) starts from a validated base rather than scratch.
//
// EXPERIMENTAL tier (docs/internal/archive/task-q8k-integer-accum.md). This is the algorithm cpubrrr took
// from llama.cpp's ARM path to beat it on Q4_K: quantize activations to Q8_K, accumulate the
// sub-block integer dot products weighted by the integer sub-scales in int32, and convert to float
// ONCE per 256-value superblock. The inner loop never leaves the integer domain, so it maps onto
// SDOT + integer-add throughput instead of the FP pipeline.
//
// Standalone by design: nothing here is wired into WeightMat's dispatch. The productionization
// (a native-GGUF WeightMat kind) is the step AFTER the throughput go/no-go clears — see the task
// doc §4. This file is scalar; the SDOT/SIMD kernel and quad-interleaved layout are later stages
// that must stay bit-exact against these references.
//
// linalg is stdlib-only and does not import embed, so the block layouts are mirrored from
// embed/gguf.go (dequantQ6KBlock / dequantQ4KBlock); TestKQuantUnpackMatchesGolden pins them so the
// two copies cannot drift.

import (
	"fmt"
	"math"
)

// u16le reads a little-endian uint16 (linalg avoids encoding/binary for this one spot).
func u16le(b []byte) uint16 { return uint16(b[0]) | uint16(b[1])<<8 }

const (
	qkK       = 256 // K-quant superblock size (weights per superblock)
	q6kBlockB = 210 // Q6_K bytes per superblock: ql[128] qh[64] sc[16] d(f16)
	q4kBlockB = 144 // Q4_K bytes per superblock: d(f16) dmin(f16) scales[12] qs[128]
)

// f16ToF32 decodes an IEEE-754 half. Mirrors embed.halfBitsToF32 (linalg cannot import embed).
func f16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := int32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	var bits uint32
	switch {
	case exp == 0:
		if mant == 0 {
			bits = sign
			break
		}
		exp = 1
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		mant &= 0x3ff
		bits = sign | uint32(exp+(127-15))<<23 | mant<<13
	case exp == 0x1f:
		bits = sign | 0x7f800000 | mant<<13
	default:
		bits = sign | uint32(exp+(127-15))<<23 | mant<<13
	}
	return math.Float32frombits(bits)
}

// Q8K is a Q8_K-quantized activation matrix (mirrors llama.cpp block_q8_K): each 256-value block
// carries a single f32 scale, 256 int8 values, and 16 int16 bsums (the sum of each consecutive-16
// run). The bsums are computed here, during quantization, because Q4_K's min-offset term needs
// Σq8 per sub-block and recomputing it in the kernel is a wasted second pass. Rows are independent;
// K must be a multiple of 256.
type Q8K struct {
	M, K  int
	D     []float32 // [M * K/256]  per-block scale
	Qs    []int8    // [M * K]      quantized values
	Bsums []int16   // [M * K/16]   per-consecutive-16 sums
}

// QuantizeActQ8K quantizes a[M,K] (row-major f32) to Q8_K. Panics if K is not a multiple of 256.
// Per-block symmetric scale d = maxAbs/127 (matching quantizeRowInt8's rounding), value =
// round(x/d) clamped to ±127; an all-zero block yields d=0 and zero values.
func QuantizeActQ8K(a []float32, M, K int) Q8K {
	if M < 0 || K < 0 {
		panic(fmt.Sprintf("QuantizeActQ8K: negative dim (M=%d K=%d)", M, K))
	}
	if K%qkK != 0 {
		panic(fmt.Sprintf("QuantizeActQ8K: K=%d not a multiple of %d", K, qkK))
	}
	requireExactLen("QuantizeActQ8K", "a", len(a), mul(M, K))
	nb := K / qkK
	q := Q8K{
		M: M, K: K,
		D:     make([]float32, mul(M, nb)),
		Qs:    make([]int8, mul(M, K)),
		Bsums: make([]int16, mul(M, K/16)),
	}
	for m := range M {
		for b := range nb {
			blk := a[m*K+b*qkK : m*K+b*qkK+qkK]
			var maxAbs float32
			for _, v := range blk {
				if v < 0 {
					v = -v
				}
				if v > maxAbs {
					maxAbs = v
				}
			}
			bi := m*nb + b
			qOff := m*K + b*qkK
			sOff := (m*K + b*qkK) / 16
			if maxAbs == 0 {
				q.D[bi] = 0
				continue // Qs/Bsums already zero
			}
			d := maxAbs / 127
			inv := 1.0 / d
			q.D[bi] = d
			for sub := range 16 {
				var bsum int32
				for i := range 16 {
					x := math.Round(float64(blk[sub*16+i] * inv))
					if x > 127 {
						x = 127
					} else if x < -127 {
						x = -127
					}
					qi := int8(x)
					q.Qs[qOff+sub*16+i] = qi
					bsum += int32(qi)
				}
				q.Bsums[sOff+sub] = int16(bsum)
			}
		}
	}
	return q
}

// unpackQ6K unpacks superblock sb of a Q6_K weight row into recentered int8 codes in logical order
// 0..255 (q−32 ∈ [−32,31]), the 16 int8 sub-scales (scale[j] applies to codes[j*16 : j*16+16]),
// and the f16 super-scale. Mirrors embed.dequantQ6KBlock's bit extraction, storing the recentered
// code instead of d·sc·q.
func unpackQ6K(raw []byte, sb int, codes *[qkK]int8) (scales [16]int8, d float32) {
	base := sb * q6kBlockB
	ql := raw[base : base+128]
	qh := raw[base+128 : base+192]
	sc := raw[base+192 : base+208]
	d = f16ToF32(u16le(raw[base+208:]))
	for i := range 16 {
		scales[i] = int8(sc[i])
	}
	for chunk := range 2 {
		n0 := chunk * 128
		qlo := ql[chunk*64:]
		qho := qh[chunk*32:]
		for l := range 32 {
			q1 := int8((qlo[l]&0x0F)|(((qho[l]>>0)&3)<<4)) - 32
			q2 := int8((qlo[l+32]&0x0F)|(((qho[l]>>2)&3)<<4)) - 32
			q3 := int8((qlo[l]>>4)|(((qho[l]>>4)&3)<<4)) - 32
			q4 := int8((qlo[l+32]>>4)|(((qho[l]>>6)&3)<<4)) - 32
			codes[n0+l+0] = q1
			codes[n0+l+32] = q2
			codes[n0+l+64] = q3
			codes[n0+l+96] = q4
		}
	}
	return scales, d
}

// unpackQ4K unpacks superblock sb of a Q4_K weight row into raw int8 codes in logical order 0..255
// (q ∈ [0,15]), the 8 sub-block 6-bit scales and mins (scales[s]/mins[s] apply to codes[s*32 :
// s*32+32]), and the f16 super-scales d, dmin. Mirrors embed.dequantQ4KBlock.
func unpackQ4K(raw []byte, sb int, codes *[qkK]int8) (scales, mins [8]uint8, d, dmin float32) {
	base := sb * q4kBlockB
	d = f16ToF32(u16le(raw[base:]))
	dmin = f16ToF32(u16le(raw[base+2:]))
	scaleBytes := raw[base+4 : base+16]
	qs := raw[base+16 : base+144]
	for s := range 8 {
		scales[s], mins[s] = q4kScaleMin(s, scaleBytes)
	}
	for j := range 4 { // four 64-element groups → sub-blocks 2j (low nibble) and 2j+1 (high)
		q := qs[j*32 : j*32+32]
		for l := range 32 {
			codes[j*64+l] = int8(q[l] & 0x0F)
			codes[j*64+32+l] = int8(q[l] >> 4)
		}
	}
	return scales, mins, d, dmin
}

// q4kScaleMin unpacks the s-th 6-bit scale and min from a Q4_K super-block's 12-byte scales array
// (ggml get_scale_min_k4). Mirrors embed.q4kScaleMin.
func q4kScaleMin(j int, q []byte) (scale, min uint8) {
	if j < 4 {
		scale = q[j] & 63
		min = q[j+4] & 63
	} else {
		scale = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4)
		min = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
	}
	return scale, min
}

// dotQ6KQ8K is the integer-accumulation dot of one Q6_K weight row (nb superblocks, nb*210 bytes)
// with one Q8_K activation row. Per superblock: 16 sub-block int dot products, weighted by the
// int8 sub-scales in int32, converted to float once. The recentered codes (q−32) fold Q6_K's
// offset so no bsums term is needed.
func dotQ6KQ8K(w []byte, q8 *Q8K, row, nb int) float32 {
	var out float32
	var codes [qkK]int8
	var partials [16]int32
	qsBase := row * q8.K
	dBase := row * nb
	for sb := range nb {
		scales, dw := unpackQ6K(w, sb, &codes)
		qs := q8.Qs[qsBase+sb*qkK : qsBase+sb*qkK+qkK]
		dotPartials16(codes[:], qs, partials[:]) // 16 sub-block int dots (SDOT if available)
		var acc int32
		for j := range 16 {
			acc += int32(scales[j]) * partials[j]
		}
		out += dw * q8.D[dBase+sb] * float32(acc)
	}
	return out
}

// dotPartials16Scalar is the arch-neutral reference for dotPartials16: out[j] = Σ_{i<16}
// codes[j*16+i]·qs[j*16+i]. The DotProd (SDOT) path in kquant_dp_arm64.s must match it bit-for-bit
// (integer arithmetic — exact regardless of path).
func dotPartials16Scalar(codes, qs []int8, out []int32) {
	for j := range out {
		c := codes[j*16 : j*16+16]
		a := qs[j*16 : j*16+16]
		var p int32
		for i := range 16 {
			p += int32(c[i]) * int32(a[i])
		}
		out[j] = p
	}
}

// dotQ4KQ8K is the integer-accumulation dot of one Q4_K weight row with one Q8_K activation row.
// Per superblock: 8 sub-block int dot products weighted by the 6-bit scales, plus the min-offset
// term −dmin·Σ_s min[s]·(Σ_32 q8) folded from the bsums (block_q8_K stores bsums per-16, so a
// 32-wide sub-block sums an adjacent pair). Two float converts per superblock (scale and min terms).
func dotQ4KQ8K(w []byte, q8 *Q8K, row, nb int) float32 {
	var out float32
	var codes [qkK]int8
	var partials [16]int32
	qsBase := row * q8.K
	dBase := row * nb
	bBase := row * (q8.K / 16)
	for sb := range nb {
		scales, mins, dw, dmin := unpackQ4K(w, sb, &codes)
		qs := q8.Qs[qsBase+sb*qkK : qsBase+sb*qkK+qkK]
		bsums := q8.Bsums[bBase+sb*16:]
		dotPartials16(codes[:], qs, partials[:]) // 16 partials; a 32-wide sub-block sums an adjacent pair
		var accScale, accMin int32
		for s := range 8 {
			p := partials[2*s] + partials[2*s+1]
			accScale += int32(scales[s]) * p
			accMin += int32(mins[s]) * (int32(bsums[2*s]) + int32(bsums[2*s+1]))
		}
		da := q8.D[dBase+sb]
		out += da * (dw*float32(accScale) - dmin*float32(accMin))
	}
	return out
}

// MatmulBTGGUFQ6K computes dst[M,N] = a[M,K] · Wᵀ where W is N rows of Q6_K weights (row n at
// w[n*rowBytes : (n+1)*rowBytes], rowBytes = K/256*210). Activations are quantized to Q8_K once,
// then integer-accum dotted. Standalone (not a WeightMat kind); scalar; the go/no-go benchmark and
// the SIMD kernel call the per-row dot directly.
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:M*N].
func MatmulBTGGUFQ6K(a []float32, w []byte, dst []float32, M, K, N int) {
	if K%qkK != 0 {
		panic(fmt.Sprintf("MatmulBTGGUFQ6K: K=%d not a multiple of %d", K, qkK))
	}
	nb := K / qkK
	rowBytes := nb * q6kBlockB
	requireExactLen("MatmulBTGGUFQ6K", "w", len(w), mul(N, rowBytes))
	requireExactLen("MatmulBTGGUFQ6K", "dst", len(dst), mul(M, N))
	q8 := QuantizeActQ8K(a, M, K)
	for m := range M {
		for n := range N {
			dst[m*N+n] = dotQ6KQ8K(w[n*rowBytes:(n+1)*rowBytes], &q8, m, nb)
		}
	}
}

// MatmulBTGGUFQ4K is MatmulBTGGUFQ6K for Q4_K weights (rowBytes = K/256*144).
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:M*N].
func MatmulBTGGUFQ4K(a []float32, w []byte, dst []float32, M, K, N int) {
	if K%qkK != 0 {
		panic(fmt.Sprintf("MatmulBTGGUFQ4K: K=%d not a multiple of %d", K, qkK))
	}
	nb := K / qkK
	rowBytes := nb * q4kBlockB
	requireExactLen("MatmulBTGGUFQ4K", "w", len(w), mul(N, rowBytes))
	requireExactLen("MatmulBTGGUFQ4K", "dst", len(dst), mul(M, N))
	q8 := QuantizeActQ8K(a, M, K)
	for m := range M {
		for n := range N {
			dst[m*N+n] = dotQ4KQ8K(w[n*rowBytes:(n+1)*rowBytes], &q8, m, nb)
		}
	}
}
