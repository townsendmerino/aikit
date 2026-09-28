package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestF32ToF16_rounding checks the encoder against an independent rule: the result is a nearest binary16
// value, ties resolved away from zero (round-half-up on the magnitude, goinfer's F16Bits), every binary16
// value round-trips, overflow saturates to ±Inf, and NaN stays NaN.
func TestF32ToF16_rounding(t *testing.T) {
	for h := 0; h < 1<<16; h++ {
		f := F16ToF32(uint16(h))
		if f != f {
			if g := F16ToF32(F32ToF16(f)); g == g {
				t.Fatalf("NaN %#04x did not stay NaN", h)
			}
			continue
		}
		if got := F32ToF16(f); got != uint16(h) && !(f == 0 && got&0x7fff == 0) {
			t.Fatalf("binary16 %#04x (%g) round-trips to %#04x", h, f, got)
		}
	}
	rng := rand.New(rand.NewPCG(5, 6))
	for n := 0; n < 200000; n++ {
		x := float32(math.Ldexp(rng.Float64()*2-1, rng.IntN(40)-26)) // magnitudes across subnormal..overflow
		got := F32ToF16(x)
		if math.Abs(float64(x)) >= 65520 { // at or past the half-way point above 65504
			if got&0x7fff != 0x7c00 {
				t.Fatalf("%g: want ±Inf, got %#04x", x, got)
			}
			continue
		}
		gv := float64(F16ToF32(got))
		best := math.Inf(1)
		for d := -1; d <= 1; d++ { // the nearest candidate among the neighbours of the result
			c := int(got&0x7fff) + d
			if c < 0 || c > 0x7bff {
				continue
			}
			cv := float64(F16ToF32(uint16(c) | got&0x8000))
			best = math.Min(best, math.Abs(cv-float64(x)))
		}
		if e := math.Abs(gv - float64(x)); e > best {
			t.Fatalf("%g -> %g (%#04x) is not a nearest binary16 value (error %g > %g)", x, gv, got, e, best)
		}
		if e := math.Abs(gv - float64(x)); e == best && math.Abs(gv) < math.Abs(float64(x)) {
			// a tie resolved toward zero: only wrong if the other neighbour is equally near
			up := F16ToF32((got&0x7fff + 1) | got&0x8000)
			if math.Abs(float64(up)-float64(x)) == e {
				t.Fatalf("%g: tie resolved toward zero (%g), want away (%g)", x, gv, up)
			}
		}
	}
}

// f16Fixture returns nibbles, binary16 scales (quantizer output converted with F32ToF16) and their exact
// f32 widening — the values any f32-scale path must be fed for the comparison to be bit-for-bit.
func f16Fixture(rng *rand.Rand, N, K int) (q4 []byte, s16 []uint16, s32 []float32) {
	w := make([]float32, N*K)
	for i := range w {
		w[i] = float32(rng.NormFloat64()) * 0.02
	}
	q4, sq := QuantizeGroupsInt4(w, N, K, 32)
	s16 = F32ToF16Scales(sq)
	s32 = make([]float32, len(s16))
	F16ToF32Slice(s32, s16)
	return
}

func f16Acts(rng *rand.Rand, n int) []float32 {
	a := make([]float32, n)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	return a
}

func eqF32(t *testing.T, what string, got, want []float32) {
	t.Helper()
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: [%d] f16 %v != f32 %v", what, i, got[i], want[i])
		}
	}
}

func differsF32(a, b []float32) bool {
	for i := range a {
		if a[i] != b[i] {
			return true
		}
	}
	return false
}

// TestW4A8F16_everyPathMatchesF32 is gate 1 of the L1 build (goinfer docs/tasks/task-cpu-decode-peer-gap-2026-09.md):
// every W4A8 path fed binary16 scales equals its f32 path fed the widened values, bit for bit, and a
// one-ulp change to one scale moves each path's output.
func TestW4A8F16_everyPathMatchesF32(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	for _, K := range []int{896, 1536} { // 28 groups (not a multiple of 8: the overlapping widen block) and 48
		const N = 64
		q4, s16, s32 := f16Fixture(rng, N, K)
		for _, M := range []int{1, 2, 3, 4, 7} {
			a := f16Acts(rng, M*K)
			for _, ag := range []int{0, 32} {
				var ws Workspace
				ws.SetActQuantGroup(ag)
				want, got := make([]float32, M*N), make([]float32, M*N)
				MatmulBTW4A8Into(&ws, a, q4, s32, want, M, K, N, 32)
				MatmulBTW4A8F16Into(&ws, a, q4, s16, got, M, K, N, 32)
				eqF32(t, "MatmulBTW4A8F16Into", got, want)

				// Batch ops carrying only binary16 scales.
				gb := make([]float32, M*N)
				MatmulBTW4A8Batch(&ws, a, M, K, 32, []W4A8Op{{W4: q4, ScalesF16: s16, Dst: gb, N: N}})
				eqF32(t, "MatmulBTW4A8Batch(ScalesF16)", gb, want)

				// WeightMat: canonical, then whichever repacked layout this arch has.
				for _, layout := range []string{"canonical", "repacked"} {
					wm := WrapInt4F16(append([]byte(nil), q4...), append([]uint16(nil), s16...), N, K, 32)
					if layout == "repacked" && !wm.RepackInt4Row4() && !wm.RepackInt4SplitHalf() {
						continue
					}
					gw := make([]float32, M*N)
					wm.MatmulBTInto(&ws, a, gw, M)
					eqF32(t, "WeightMat.MatmulBTInto "+layout, gw, want)
					// The layout-preferring entry point (split-half / row4 when present), against that layout's own
					// f32 path: repacked kernels are not all bit-identical to canonical (the split-half per-32
					// kernel forms its combined scale in-register), so canonical is not their reference.
					ref := want
					if layout == "repacked" {
						ref = layoutRefF32(&ws, a, &wm, q4, s32, M, K, N)
					}
					gl := make([]float32, M*N)
					wm.MatmulBTW4A8Into(&ws, a, gl, M)
					eqF32(t, "WeightMat.MatmulBTW4A8Into "+layout, gl, ref)
					if layout == "repacked" {
						if r4, r4s, ok := wm.Int4Row4F16(); ok {
							gr := make([]float32, M*N)
							MatmulBTW4A8Batch(&ws, a, M, K, 32, []W4A8Op{{W4: q4, ScalesF16: s16, Row4: r4, Row4ScalesF16: r4s, Dst: gr, N: N}})
							eqF32(t, "MatmulBTW4A8Batch(Row4ScalesF16)", gr, batchRow4RefF32(&ws, a, q4, s32, &wm, M, K, N))
						}
					}
					// Mutation: one scale one ulp up must move this path's output.
					mut := wm
					m16 := append([]uint16(nil), s16...)
					m16[5]++
					if layout == "canonical" {
						mut = WrapInt4F16(q4, m16, N, K, 32)
					} else if mut.q4Row4Scales16 != nil {
						mut.q4Row4Scales16 = append([]uint16(nil), wm.q4Row4Scales16...)
						mut.q4Row4Scales16[5]++
					} else {
						mut.q4s16 = m16
					}
					gm := make([]float32, M*N)
					mut.MatmulBTW4A8Into(&ws, a, gm, M)
					if !differsF32(gm, ref) {
						t.Fatalf("WeightMat %s K=%d M=%d ag=%d: a changed scale left the output unchanged", layout, K, M, ag)
					}
				}
			}
		}
		// Row(): every layout dequantizes to DequantizeRowInt4 over the widened scales.
		nG := K / 32
		for _, layout := range []string{"canonical", "repacked"} {
			wm := WrapInt4F16(append([]byte(nil), q4...), append([]uint16(nil), s16...), N, K, 32)
			if layout == "repacked" && !wm.RepackInt4Row4() && !wm.RepackInt4SplitHalf() {
				continue
			}
			for _, i := range []int{0, 5, N - 1} {
				got, want := make([]float32, K), make([]float32, K)
				wm.Row(i, got)
				DequantizeRowInt4(q4[i*K/2:(i+1)*K/2], s32[i*nG:(i+1)*nG], 32, K, want)
				eqF32(t, "Row "+layout, got, want)
			}
		}
	}
	t.Logf("fused amd64 row kernel active: %v; row4 usable: %v; split-half usable: %v", f16FusedKernelActive(), row4Usable(), splitHalfUsable())
}
