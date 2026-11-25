package linalg

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/internal/half"
)

// Correctness for the native K-quant integer-accumulation matmul (kquant.go). Order of trust:
//  1. unpackQ6K/unpackQ4K reproduce embed/gguf.go's dequant EXACTLY (refDequant* below are
//     line-for-line copies of that formula, so this is the anti-drift guard §4 asks for — a bug in
//     the linalg unpack shows up as a mismatch against the independent embed-shaped reference).
//  2. QuantizeActQ8K round-trips within quant error and its bsums are exact.
//  3. The integer-accum dot equals a dequant-then-f32-dot over the SAME quantized activations —
//     algebraically identical, so any structural bug (wrong scale index, bad bsums pairing, missing
//     min term) dwarfs f32 rounding.

// --- Independent references, copied from embed/gguf.go (do NOT call unpack*). ---

func refDequantQ6K(raw []byte, sb int, out []float32) {
	base := sb * 210
	ql := raw[base : base+128]
	qh := raw[base+128 : base+192]
	sc := raw[base+192 : base+208]
	d := half.ToF32(u16le(raw[base+208:]))
	for chunk := range 2 {
		n0 := chunk * 128
		qlo := ql[chunk*64:]
		qho := qh[chunk*32:]
		sco := sc[chunk*8:]
		for l := range 32 {
			is := l / 16
			q1 := int8((qlo[l]&0x0F)|(((qho[l]>>0)&3)<<4)) - 32
			q2 := int8((qlo[l+32]&0x0F)|(((qho[l]>>2)&3)<<4)) - 32
			q3 := int8((qlo[l]>>4)|(((qho[l]>>4)&3)<<4)) - 32
			q4 := int8((qlo[l+32]>>4)|(((qho[l]>>6)&3)<<4)) - 32
			out[n0+l+0] = d * float32(int8(sco[is+0])) * float32(q1)
			out[n0+l+32] = d * float32(int8(sco[is+2])) * float32(q2)
			out[n0+l+64] = d * float32(int8(sco[is+4])) * float32(q3)
			out[n0+l+96] = d * float32(int8(sco[is+6])) * float32(q4)
		}
	}
}

func refDequantQ4K(raw []byte, sb int, out []float32) {
	base := sb * 144
	d := half.ToF32(u16le(raw[base:]))
	dmin := half.ToF32(u16le(raw[base+2:]))
	scales := raw[base+4 : base+16]
	qs := raw[base+16 : base+144]
	yi := 0
	for j := range 4 {
		is := 2 * j
		sc1, m1 := q4kScaleMin(is+0, scales)
		sc2, m2 := q4kScaleMin(is+1, scales)
		d1, off1 := d*float32(sc1), dmin*float32(m1)
		d2, off2 := d*float32(sc2), dmin*float32(m2)
		q := qs[j*32 : j*32+32]
		for l := range 32 {
			out[yi] = d1*float32(q[l]&0x0F) - off1
			yi++
		}
		for l := range 32 {
			out[yi] = d2*float32(q[l]>>4) - off2
			yi++
		}
	}
}

// reconstructQ6K / reconstructQ4K rebuild f32 from the linalg unpack, in the same multiply order as
// the embed dequant, so a correct unpack is BIT-identical to refDequant.
func reconstructQ6K(raw []byte, sb int, out []float32) {
	var codes [qkK]int8
	scales, d := unpackQ6K(raw, sb, &codes)
	for p := range qkK {
		out[p] = d * float32(scales[p/16]) * float32(codes[p])
	}
}

func reconstructQ4K(raw []byte, sb int, out []float32) {
	var codes [qkK]int8
	scales, mins, d, dmin := unpackQ4K(raw, sb, &codes)
	for p := range qkK {
		s := p / 32
		out[p] = d*float32(scales[s])*float32(codes[p]) - dmin*float32(mins[s])
	}
}

func randBytes(rng *rand.Rand, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(rng.Intn(256))
	}
	return b
}

// randF16FiniteBits returns f16 bits with exponent in [0,30] — never 31, so never Inf/NaN. Real
// K-quant super-scales are always finite; random 2-byte patterns are not, and NaN·x yields NaNs
// whose bit patterns differ between the two dequant formulations, which is not a drift.
func randF16FiniteBits(rng *rand.Rand) uint16 {
	return (uint16(rng.Intn(0x10000)) &^ 0x7C00) | uint16(rng.Intn(31))<<10
}

func putU16(b []byte, v uint16) { b[0], b[1] = byte(v), byte(v>>8) }

// randQ6KRow / randQ4KRow build nb-superblock weight rows with random quant bytes but FINITE f16
// super-scales (Q6_K: d at +208; Q4_K: d at +0, dmin at +2).
func randQ6KRow(rng *rand.Rand, nb int) []byte {
	b := randBytes(rng, nb*210)
	for sb := range nb {
		putU16(b[sb*210+208:], randF16FiniteBits(rng))
	}
	return b
}

func randQ4KRow(rng *rand.Rand, nb int) []byte {
	b := randBytes(rng, nb*144)
	for sb := range nb {
		putU16(b[sb*144+0:], randF16FiniteBits(rng))
		putU16(b[sb*144+2:], randF16FiniteBits(rng))
	}
	return b
}

// TestKQuantUnpackMatchesEmbed is the drift guard: the linalg unpack, reconstructed to f32, must be
// bit-identical to the embed-shaped reference dequant over random blocks (every byte pattern is a
// valid block).
func TestKQuantUnpackMatchesEmbed(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	var got, want [qkK]float32
	for iter := range 200 {
		q6 := randQ6KRow(rng, 1)
		reconstructQ6K(q6, 0, got[:])
		refDequantQ6K(q6, 0, want[:])
		for p := range qkK {
			if math.Float32bits(got[p]) != math.Float32bits(want[p]) {
				t.Fatalf("Q6_K unpack drift at iter %d pos %d: got %v want %v", iter, p, got[p], want[p])
			}
		}
		q4 := randQ4KRow(rng, 1)
		reconstructQ4K(q4, 0, got[:])
		refDequantQ4K(q4, 0, want[:])
		for p := range qkK {
			if math.Float32bits(got[p]) != math.Float32bits(want[p]) {
				t.Fatalf("Q4_K unpack drift at iter %d pos %d: got %v want %v", iter, p, got[p], want[p])
			}
		}
	}
}

// TestKQuantDotPartials_matchesScalar pins the DotProd (SDOT) partials to the scalar reference
// bit-for-bit over full int8 range. On arm64+DotProd (every Apple-silicon Mac) this exercises the
// asm in kquant_dp_arm64.s; elsewhere both sides are scalar. Integer arithmetic — must be exact.
func TestKQuantDotPartials_matchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	for _, nsub := range []int{1, 16, 16 * 8} { // one sub-block, one superblock, eight superblocks
		codes := make([]int8, nsub*16)
		qs := make([]int8, nsub*16)
		for i := range codes {
			codes[i] = int8(rng.Intn(256) - 128) // full [-128,127]
			qs[i] = int8(rng.Intn(255) - 127)    // [-127,127]
		}
		got := make([]int32, nsub)
		want := make([]int32, nsub)
		dotPartials16(codes, qs, got)
		dotPartials16Scalar(codes, qs, want)
		for j := range nsub {
			if got[j] != want[j] {
				t.Fatalf("partial[%d] (nsub=%d): dotPartials16 %d != scalar %d", j, nsub, got[j], want[j])
			}
		}
	}
}

// TestQuantizeActQ8K checks round-trip accuracy and exact bsums (and the all-zero block → d=0).
func TestQuantizeActQ8K(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const M, K = 3, 512
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	// One block deliberately all-zero to exercise d=0.
	for i := range qkK {
		a[qkK:][i] = 0 // row 0, block 1
	}
	q := QuantizeActQ8K(a, M, K)
	nb := K / qkK
	if len(q.D) != M*nb || len(q.Qs) != M*K || len(q.Bsums) != M*K/16 {
		t.Fatalf("shape: D=%d Qs=%d Bsums=%d", len(q.D), len(q.Qs), len(q.Bsums))
	}
	if q.D[0*nb+1] != 0 {
		t.Errorf("all-zero block should have d=0, got %v", q.D[1])
	}
	for m := range M {
		for b := range nb {
			d := q.D[m*nb+b]
			// bsums exact.
			for sub := range 16 {
				var want int32
				for i := range 16 {
					want += int32(q.Qs[m*K+b*qkK+sub*16+i])
				}
				if got := int32(q.Bsums[(m*K+b*qkK)/16+sub]); got != want {
					t.Fatalf("bsums[m=%d b=%d sub=%d] = %d want %d", m, b, sub, got, want)
				}
			}
			// round-trip: dequant q8 ≈ original within one quant step (d).
			if d == 0 {
				continue
			}
			for i := range qkK {
				orig := a[m*K+b*qkK+i]
				deq := d * float32(q.Qs[m*K+b*qkK+i])
				if math.Abs(float64(orig-deq)) > float64(d)+1e-6 {
					t.Fatalf("round-trip m=%d b=%d i=%d: orig %v deq %v (d=%v)", m, b, i, orig, deq, d)
				}
			}
		}
	}
}

// oracleDot dequants a weight row and the quantized activation row to f32 and dots them — the
// dequant-then-f32-dot reference the integer kernel must match.
func oracleDotQ6K(w []byte, q8 *Q8K, row, nb int) float64 {
	var wf [qkK]float32
	var sum float64
	for sb := range nb {
		refDequantQ6K(w, sb, wf[:])
		da := float64(q8.D[row*nb+sb])
		qs := q8.Qs[row*q8.K+sb*qkK:]
		for p := range qkK {
			sum += float64(wf[p]) * da * float64(qs[p])
		}
	}
	return sum
}

func oracleDotQ4K(w []byte, q8 *Q8K, row, nb int) float64 {
	var wf [qkK]float32
	var sum float64
	for sb := range nb {
		refDequantQ4K(w, sb, wf[:])
		da := float64(q8.D[row*nb+sb])
		qs := q8.Qs[row*q8.K+sb*qkK:]
		for p := range qkK {
			sum += float64(wf[p]) * da * float64(qs[p])
		}
	}
	return sum
}

// TestKQuantDot_vsOracle: the integer-accum dot equals the dequant-then-dot oracle over the same
// quantized activations, across several K. Relative tolerance absorbs f32 rounding; any structural
// bug is orders of magnitude larger.
func TestKQuantDot_vsOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, K := range []int{256, 512, 2048} {
		nb := K / qkK
		a := make([]float32, K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		q8 := QuantizeActQ8K(a, 1, K)

		w6 := randQ6KRow(rng, nb)
		got6 := float64(dotQ6KQ8K(w6, &q8, 0, nb))
		want6 := oracleDotQ6K(w6, &q8, 0, nb)
		if math.Abs(got6-want6) > 1e-3*(math.Abs(want6)+1) {
			t.Errorf("Q6_K K=%d: dot %v vs oracle %v", K, got6, want6)
		}

		w4 := randQ4KRow(rng, nb)
		got4 := float64(dotQ4KQ8K(w4, &q8, 0, nb))
		want4 := oracleDotQ4K(w4, &q8, 0, nb)
		if math.Abs(got4-want4) > 1e-3*(math.Abs(want4)+1) {
			t.Errorf("Q4_K K=%d: dot %v vs oracle %v", K, got4, want4)
		}
	}
}

// TestKQuant_edgeCases: all-zero weight block, all-zero activation block (d_a=0), and scale/min
// extremes — each must still match the oracle (integer path degenerates cleanly).
func TestKQuant_edgeCases(t *testing.T) {
	const K = 256
	nb := 1
	mk := func(seedVals func(a []float32)) Q8K {
		a := make([]float32, K)
		seedVals(a)
		return QuantizeActQ8K(a, 1, K)
	}
	rng := rand.New(rand.NewSource(4))

	// activation all-zero → d_a=0 → dot must be exactly 0 for any weights.
	q8zero := mk(func(a []float32) {})
	w6 := randQ6KRow(rng, 1)
	if v := dotQ6KQ8K(w6, &q8zero, 0, nb); v != 0 {
		t.Errorf("zero-activation Q6_K dot = %v, want 0", v)
	}
	w4 := randQ4KRow(rng, 1)
	if v := dotQ4KQ8K(w4, &q8zero, 0, nb); v != 0 {
		t.Errorf("zero-activation Q4_K dot = %v, want 0", v)
	}

	q8 := mk(func(a []float32) {
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
	})

	// Q6_K all-zero weight block (all bytes 0 → codes all −32, scales 0, d=0) → dot 0.
	zero6 := make([]byte, 210)
	if v := dotQ6KQ8K(zero6, &q8, 0, nb); v != 0 {
		t.Errorf("zero-weight Q6_K dot = %v, want 0 (d=0)", v)
	}

	// Scale extremes: Q6_K int8 sub-scales at ±127, d large; must still match the oracle.
	ext6 := randQ6KRow(rng, 1)
	for i := range 16 {
		if i%2 == 0 {
			ext6[192+i] = 127
		} else {
			ext6[192+i] = 129 // int8 −127
		}
	}
	if got, want := float64(dotQ6KQ8K(ext6, &q8, 0, nb)), oracleDotQ6K(ext6, &q8, 0, nb); math.Abs(got-want) > 1e-3*(math.Abs(want)+1) {
		t.Errorf("Q6_K scale-extreme: %v vs oracle %v", got, want)
	}

	// Q4_K with 6-bit scales and mins at their max (63) — exercises the min-offset term hard.
	ext4 := randQ4KRow(rng, 1)
	for i := range 12 {
		ext4[4+i] = 0xFF // all 6-bit fields → 63
	}
	if got, want := float64(dotQ4KQ8K(ext4, &q8, 0, nb)), oracleDotQ4K(ext4, &q8, 0, nb); math.Abs(got-want) > 1e-3*(math.Abs(want)+1) {
		t.Errorf("Q4_K scale/min-extreme: %v vs oracle %v", got, want)
	}
}

// TestKQuantMatmulBT: the standalone GEMV wrappers match per-row oracle dots.
func TestKQuantMatmulBT(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	const M, K, N = 2, 512, 5
	nb := K / qkK
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	q8 := QuantizeActQ8K(a, M, K)

	w6 := randQ6KRow(rng, N*nb)
	dst6 := make([]float32, M*N)
	MatmulBTGGUFQ6K(a, w6, dst6, M, K, N)
	for m := range M {
		for n := range N {
			want := oracleDotQ6K(w6[n*nb*210:(n+1)*nb*210], &q8, m, nb)
			if got := float64(dst6[m*N+n]); math.Abs(got-want) > 1e-3*(math.Abs(want)+1) {
				t.Errorf("Q6_K GEMV [%d,%d] = %v vs oracle %v", m, n, got, want)
			}
		}
	}

	w4 := randQ4KRow(rng, N*nb)
	dst4 := make([]float32, M*N)
	MatmulBTGGUFQ4K(a, w4, dst4, M, K, N)
	for m := range M {
		for n := range N {
			want := oracleDotQ4K(w4[n*nb*144:(n+1)*nb*144], &q8, m, nb)
			if got := float64(dst4[m*N+n]); math.Abs(got-want) > 1e-3*(math.Abs(want)+1) {
				t.Errorf("Q4_K GEMV [%d,%d] = %v vs oracle %v", m, n, got, want)
			}
		}
	}
}
