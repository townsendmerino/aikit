package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/internal/half"
)

// kqF16 is a random normal f16 in [2^-6, 2^1), as two little-endian bytes.
func kqF16(r *rand.Rand, dst []byte) {
	h := uint16(9+r.IntN(7))<<10 | uint16(r.IntN(1024))
	dst[0], dst[1] = byte(h), byte(h>>8)
}

// q5kTestRows builds rows×cols of valid Q5_K super-blocks: d and dmin normal f16, every other byte
// random, or 0xFF when sat (codes 31, 6-bit scales and mins 63: the largest dot).
func q5kTestRows(r *rand.Rand, rows, cols int, sat bool) []byte {
	raw := make([]byte, rows*Q5KRowBytes(cols))
	for b := 0; b < len(raw); b += q5kBlockB {
		blk := raw[b : b+q5kBlockB]
		kqF16(r, blk[0:])
		kqF16(r, blk[2:])
		for i := 4; i < q5kBlockB; i++ {
			blk[i] = byte(r.IntN(256))
			if sat {
				blk[i] = 0xFF
			}
		}
	}
	return raw
}

// q6kTestRows builds rows×cols of valid Q6_K super-blocks: d normal f16, every other byte random.
// sat=1 sets codes to 63 (q−32 = 31) and scales to 127; sat=2 codes to 0 (q−32 = −32) and scales
// to −128: the two ends of the integer dot.
func q6kTestRows(r *rand.Rand, rows, cols int, sat int) []byte {
	raw := make([]byte, rows*Q6KRowBytes(cols))
	for b := 0; b < len(raw); b += q6kBlockB {
		blk := raw[b : b+q6kBlockB]
		for i := range 208 {
			blk[i] = byte(r.IntN(256))
			switch {
			case sat == 1 && i < 192:
				blk[i] = 0xFF
			case sat == 1:
				blk[i] = 0x7F
			case sat == 2 && i < 192:
				blk[i] = 0
			case sat == 2:
				blk[i] = 0x80
			}
		}
		kqF16(r, blk[208:])
	}
	return raw
}

// q5kTruthWeights dequantizes one Q5_K row in float64, written from embed.dequantQ5KBlock's form
// (u1/u2 bit masks) rather than q5kCode, so the two decodings are independent.
func q5kTruthWeights(row []byte, K int) []float64 {
	w := make([]float64, K)
	for b := range K / qkK {
		blk := row[b*q5kBlockB : (b+1)*q5kBlockB]
		d, dmin := float64(half.ToF32(u16le(blk[0:]))), float64(half.ToF32(u16le(blk[2:])))
		scales, qh, qs := blk[4:16], blk[16:48], blk[48:176]
		u1, u2 := byte(1), byte(2)
		for j := range 4 {
			sc1, m1 := q4kScaleMin(2*j, scales)
			sc2, m2 := q4kScaleMin(2*j+1, scales)
			for l := range 32 {
				lo, hi := float64(qs[32*j+l]&0x0F), float64(qs[32*j+l]>>4)
				if qh[l]&u1 != 0 {
					lo += 16
				}
				if qh[l]&u2 != 0 {
					hi += 16
				}
				w[b*qkK+64*j+l] = d*float64(sc1)*lo - dmin*float64(m1)
				w[b*qkK+64*j+32+l] = d*float64(sc2)*hi - dmin*float64(m2)
			}
			u1 <<= 2
			u2 <<= 2
		}
	}
	return w
}

// q6kTruthWeights dequantizes one Q6_K row in float64 through kquant.go's unpackQ6K (pinned against
// embed's decoder by TestKQuantUnpackMatchesGolden), independent of q6kCode.
func q6kTruthWeights(row []byte, K int) []float64 {
	w := make([]float64, K)
	var codes [qkK]int8
	for b := range K / qkK {
		sc, d := unpackQ6K(row, b, &codes)
		for i := range qkK {
			w[b*qkK+i] = float64(d) * float64(sc[i/16]) * float64(codes[i])
		}
	}
	return w
}

// kqTruth is dst[m,n] = Σ_k w[n,k]·(aq[m,k]·aS[m,k/32]) in float64.
func kqTruth(weights func(row []byte, K int) []float64, raw []byte, rb int, a []float32, M, K, N int) []float64 {
	nG := K / q4kGroup
	aq := make([]int8, M*K)
	aS := make([]float32, M*nG)
	QuantizeActivationsGroupedInto(aq, aS, a, M, K, q4kGroup)
	out := make([]float64, M*N)
	for n := range N {
		w := weights(raw[n*rb:(n+1)*rb], K)
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

type kqCase struct {
	name    string
	rows    func(r *rand.Rand, rows, cols int, sat int) []byte
	rb      func(cols int) int
	wrap    func(raw []byte, rows, cols int) (WeightMat, error)
	truth   func(row []byte, K int) []float64
	kind    string
	sats    []int
	rawOf   func(w *WeightMat) ([]byte, bool)
	kernel  func(row []byte, aq []int8, aS []float32, sums []int32, asumf []float32, K int) float32
	oracle  func(row []byte, aq []int8, aS []float32, sums []int32, K int) float32
	rowFill func(row []byte, cols int, dst []float32)
}

var kqCases = []kqCase{
	{
		name: "Q5K", kind: "q5k", sats: []int{0, 1},
		rows:    func(r *rand.Rand, rows, cols int, sat int) []byte { return q5kTestRows(r, rows, cols, sat == 1) },
		rb:      Q5KRowBytes,
		wrap:    WrapQ5K,
		truth:   q5kTruthWeights,
		rawOf:   (*WeightMat).Q5K,
		kernel:  dotQ5K,
		oracle:  dotQ5KGo,
		rowFill: dequantQ5KRow,
	},
	{
		name: "Q6K", kind: "q6k", sats: []int{0, 1, 2},
		rows:  q6kTestRows,
		rb:    Q6KRowBytes,
		wrap:  WrapQ6K,
		truth: q6kTruthWeights,
		rawOf: (*WeightMat).Q6K,
		kernel: func(row []byte, aq []int8, aS []float32, _ []int32, _ []float32, K int) float32 {
			return dotQ6K(row, aq, aS, K)
		},
		oracle: func(row []byte, aq []int8, aS []float32, _ []int32, K int) float32 {
			return dotQ6KGo(row, aq, aS, K)
		},
		rowFill: dequantQ6KRow,
	},
}

// TestQ56K_matchesFloat64: WeightMat's Q5_K and Q6_K matmuls (MatmulBT and MatmulBTInto, serial and
// fanned out) against the float64 evaluation, relative error ≤ 1e-6, at M = 1 and 3, an odd N,
// random and saturated blocks, activations with a massive outlier.
func TestQ56K_matchesFloat64(t *testing.T) {
	for _, c := range kqCases {
		r := rand.New(rand.NewPCG(61, 62))
		for _, K := range []int{256, 3072} {
			for _, M := range []int{1, 3} {
				for _, sat := range c.sats {
					const N = 37
					raw := c.rows(r, N, K, sat)
					a := make([]float32, M*K)
					for i := range a {
						a[i] = float32(r.NormFloat64())
						if sat == 1 {
							a[i] = 1
						} else if sat == 2 {
							a[i] = -1
						}
					}
					a[K/3] = 400
					want := kqTruth(c.truth, raw, c.rb(K), a, M, K, N)
					wm, err := c.wrap(raw, N, K)
					if err != nil {
						t.Fatal(err)
					}
					got := make([]float32, M*N)
					wm.MatmulBT(a, got, M)
					if e := q4kRelErr(got, want); e > 1e-6 {
						t.Errorf("%s K=%d M=%d sat=%d MatmulBT: rel err %.3g", c.name, K, M, sat, e)
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
							t.Errorf("%s K=%d M=%d sat=%d par=%v MatmulBTInto: rel err %.3g", c.name, K, M, sat, par, e)
						}
					}
				}
			}
		}
	}
}

// TestQ56K_rowAndKind: Row matches the float64 dequant (f32-rounded, within one f32 rounding of the
// product), Kind and the raw accessor report the kind, and the Wrap functions reject bad shapes.
func TestQ56K_rowAndKind(t *testing.T) {
	for _, c := range kqCases {
		r := rand.New(rand.NewPCG(63, 64))
		const N, K = 3, 512
		raw := c.rows(r, N, K, 0)
		wm, err := c.wrap(raw, N, K)
		if err != nil {
			t.Fatal(err)
		}
		if wm.Kind() != c.kind {
			t.Errorf("%s: Kind = %q", c.name, wm.Kind())
		}
		if got, ok := c.rawOf(&wm); !ok || &got[0] != &raw[0] {
			t.Errorf("%s: raw accessor does not alias the wrapped bytes", c.name)
		}
		row := make([]float32, K)
		for n := range N {
			wm.Row(n, row)
			want := c.truth(raw[n*c.rb(K):(n+1)*c.rb(K)], K)
			for k := range K {
				if d := math.Abs(float64(row[k]) - want[k]); d > 1e-6*math.Abs(want[k])+1e-30 {
					t.Fatalf("%s Row(%d)[%d] = %v, want %v", c.name, n, k, row[k], want[k])
				}
			}
		}
		for _, sh := range []struct{ rows, cols, n int }{{N, 300, 0}, {N, K, len(raw) - 1}, {0, K, 0}} {
			n := sh.n
			if n == 0 {
				n = sh.rows * c.rb(sh.cols)
			}
			if _, err := c.wrap(make([]byte, n), sh.rows, sh.cols); err == nil {
				t.Errorf("%s: Wrap(%d bytes, %dx%d) accepted", c.name, n, sh.rows, sh.cols)
			}
		}
	}
}

// TestQ56K_kernelMatchesGo: the arch kernels (AVX2 on amd64) against the Go oracles on random and
// saturated rows of 1..12 super-blocks: relative error ≤ 1e-6. On a build with no arch kernel the
// two are the same function and this checks nothing, which it logs.
func TestQ56K_kernelMatchesGo(t *testing.T) {
	if !KQuantNative() {
		t.Log("no K-quant arch kernel on this build: kernel and oracle are the same function")
	}
	for _, c := range kqCases {
		r := rand.New(rand.NewPCG(65, 66))
		for _, nSB := range []int{1, 2, 3, 12} {
			for _, sat := range c.sats {
				K := nSB * qkK
				raw := c.rows(r, 1, K, sat)
				a := make([]float32, K)
				for i := range a {
					a[i] = float32(r.NormFloat64())
					if sat == 1 {
						a[i] = 1
					} else if sat == 2 {
						a[i] = -1
					}
				}
				a[K/2] = 300
				nG := K / q4kGroup
				aq, aS := make([]int8, K), make([]float32, nG)
				QuantizeActivationsGroupedInto(aq, aS, a, 1, K, q4kGroup)
				sums := make([]int32, nG)
				q4kActSums(aq, sums, 1, K)
				asumf := make([]float32, nG)
				for i := range asumf {
					asumf[i] = aS[i] * float32(sums[i])
				}
				want := float64(c.oracle(raw, aq, aS, sums, K))
				got := float64(c.kernel(raw, aq, aS, sums, asumf, K))
				if e := math.Abs(got-want) / math.Max(math.Abs(want), 1e-30); e > 1e-6 {
					t.Errorf("%s nSB=%d sat=%d: kernel %v, Go %v (rel %.3g)", c.name, nSB, sat, got, want, e)
				}
			}
		}
	}
}

// TestWeightMat_kquantTagInPadding: the K-quant kind tag sits in WeightMat's padding. Consumers count WeightMat headers in their
// memory accounting (goinfer's decision census), so a field that grows the struct moves their numbers and wants its own change.
func TestWeightMat_kquantTagInPadding(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("the size is pinned for 64-bit builds")
	}
	if got := unsafe.Sizeof(WeightMat{}); got != 288 {
		t.Errorf("unsafe.Sizeof(WeightMat{}) = %d, want 288", got)
	}
}
