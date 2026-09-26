package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

// q4kTestRows builds rows×cols of valid Q4_K super-blocks: d and dmin are normal f16 values in
// [2^-6, 2^1), scale/min bytes and codes random. sat sets every code to 15 and every 6-bit scale and
// min to 63, the largest integer dot the kernel can see.
func q4kTestRows(r *rand.Rand, rows, cols int, sat bool) []byte {
	raw := make([]byte, rows*Q4KRowBytes(cols))
	for b := 0; b < len(raw); b += q4kBlockB {
		blk := raw[b : b+q4kBlockB]
		for _, off := range []int{0, 2} {
			h := uint16(9+r.IntN(7))<<10 | uint16(r.IntN(1024)) // exponent 9..15 → [2^-6, 2^1)
			blk[off], blk[off+1] = byte(h), byte(h>>8)
		}
		for i := 4; i < q4kBlockB; i++ {
			blk[i] = byte(r.IntN(256))
			if sat {
				blk[i] = 0xFF
			}
		}
	}
	return raw
}

// q4kTruth is dst[m,n] = Σ_k w[n,k]·(aq[m,k]·aS[m,k/32]) in float64, w dequantized in float64 from
// the same blocks: the independent evaluation the kernels are held to.
func q4kTruth(raw []byte, a []float32, M, K, N int) []float64 {
	nG := K / q4kGroup
	aq := make([]int8, M*K)
	aS := make([]float32, M*nG)
	QuantizeActivationsGroupedInto(aq, aS, a, M, K, q4kGroup)
	rb := Q4KRowBytes(K)
	out := make([]float64, M*N)
	w := make([]float64, K)
	for n := range N {
		row := raw[n*rb : (n+1)*rb]
		for b := range K / qkK {
			blk := row[b*q4kBlockB : (b+1)*q4kBlockB]
			d, dmin := float64(f16ToF32(u16le(blk[0:]))), float64(f16ToF32(u16le(blk[2:])))
			for j := range 4 {
				sc1, m1 := q4kScaleMin(2*j, blk[4:16])
				sc2, m2 := q4kScaleMin(2*j+1, blk[4:16])
				for l := range 32 {
					q := blk[16+32*j+l]
					w[b*qkK+64*j+l] = d*float64(sc1)*float64(q&0x0F) - dmin*float64(m1)
					w[b*qkK+64*j+32+l] = d*float64(sc2)*float64(q>>4) - dmin*float64(m2)
				}
			}
		}
		for m := range M {
			var s float64
			for k := range K {
				s += w[k] * float64(aq[m*K+k]) * float64(aS[m*nG+k/q4kGroup])
			}
			out[m*N+n] = s
		}
	}
	return out
}

func q4kRelErr(got []float32, want []float64) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - want[i]
		num += d * d
		den += want[i] * want[i]
	}
	return math.Sqrt(num / den)
}

// TestQ4K_matchesFloat64: WeightMat's Q4_K matmul (MatmulBT and MatmulBTInto, serial and fanned out)
// against the float64 evaluation, relative error ≤ 1e-6, at M = 1 and 3, an odd N, random and
// saturated blocks, activations with a massive outlier.
func TestQ4K_matchesFloat64(t *testing.T) {
	r := rand.New(rand.NewPCG(51, 52))
	for _, K := range []int{256, 2048} {
		for _, M := range []int{1, 3} {
			for _, sat := range []bool{false, true} {
				const N = 37
				raw := q4kTestRows(r, N, K, sat)
				a := make([]float32, M*K)
				for i := range a {
					a[i] = float32(r.NormFloat64())
					if sat {
						a[i] = 1
					}
				}
				a[K/3] = 400
				want := q4kTruth(raw, a, M, K, N)
				wm, err := WrapQ4K(raw, N, K)
				if err != nil {
					t.Fatal(err)
				}
				got := make([]float32, M*N)
				wm.MatmulBT(a, got, M)
				if e := q4kRelErr(got, want); e > 1e-6 {
					t.Errorf("K=%d M=%d sat=%v MatmulBT: rel err %.3g", K, M, sat, e)
				}
				for _, par := range []bool{false, true} {
					ws := new(Workspace)
					if par {
						ws.SetThreshold(1)
						ws.SetWorkers(5)
					} else {
						ws.SetThreshold(math.MaxInt)
					}
					clear(got)
					wm.MatmulBTInto(ws, a, got, M)
					if e := q4kRelErr(got, want); e > 1e-6 {
						t.Errorf("K=%d M=%d sat=%v par=%v MatmulBTInto: rel err %.3g", K, M, sat, par, e)
					}
				}
			}
		}
	}
}

// TestQ4K_rowAndKind: Row dequantizes exactly as the float64 evaluation's weights (f32-rounded), Kind
// and Q4K report the kind, and WrapQ4K rejects bad shapes.
func TestQ4K_rowAndKind(t *testing.T) {
	r := rand.New(rand.NewPCG(53, 54))
	const N, K = 3, 512
	raw := q4kTestRows(r, N, K, false)
	wm, err := WrapQ4K(raw, N, K)
	if err != nil {
		t.Fatal(err)
	}
	if wm.Kind() != "q4k" {
		t.Errorf("Kind = %q", wm.Kind())
	}
	if got, ok := wm.Q4K(); !ok || &got[0] != &raw[0] {
		t.Errorf("Q4K() does not alias the wrapped bytes")
	}
	var codes [qkK]int8
	row := make([]float32, K)
	for n := range N {
		wm.Row(n, row)
		for b := range K / qkK {
			sc, mn, d, dmin := unpackQ4K(raw[n*Q4KRowBytes(K):], b, &codes)
			for i := range qkK {
				want := d*float32(sc[i/32])*float32(codes[i]) - dmin*float32(mn[i/32])
				if row[b*qkK+i] != want {
					t.Fatalf("Row(%d)[%d] = %v, want %v", n, b*qkK+i, row[b*qkK+i], want)
				}
			}
		}
	}
	for _, c := range []struct{ rows, cols, n int }{{N, 300, 0}, {N, K, len(raw) - 1}, {0, K, 0}} {
		n := c.n
		if n == 0 {
			n = c.rows * Q4KRowBytes(c.cols)
		}
		if _, err := WrapQ4K(make([]byte, n), c.rows, c.cols); err == nil {
			t.Errorf("WrapQ4K(%d bytes, %dx%d) accepted", n, c.rows, c.cols)
		}
	}
}
