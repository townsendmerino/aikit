package linalg

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/linalg/internal/outputcontract"
)

// The output-contract gate over linalg's writers (the harness and its rules are linalg/internal/outputcontract's package
// doc). Run it per build:
//
//	go test ./linalg/ -run TestOutputContract                       # the host's default kernels (AVX2/VNNI on amd64, NEON on arm64)
//	GOARCH=386 go test ./linalg/ -run TestOutputContract            # the generic (!amd64 && !arm64) kernels, natively on an amd64 Linux box
//	go test -tags aikit_noavx2 ./linalg/ -run TestOutputContract    # the narrower amd64 tiers; likewise aikit_noavx512, aikit_nodotprod (arm64)

func rf32(seed int64, n int) []float32 {
	r := rand.New(rand.NewSource(seed))
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

func ri8(seed int64, n int) []int8 {
	r := rand.New(rand.NewSource(seed))
	v := make([]int8, n)
	for i := range v {
		v[i] = int8(r.Intn(255) - 127)
	}
	return v
}

func TestOutputContract_overwriteOrZeroInternally(t *testing.T) {
	for _, c := range outputContractCases() {
		t.Run(c.Name, func(t *testing.T) { outputcontract.Run(t, c) })
	}
}

// The one writer with a precondition: QuantizeGroupInt4Row never writes the PAD nibble (the high nibble of the last byte when cols is odd), so that nibble must be zero on entry; everything else it
// covers is overwritten. Even cols: fully independent of the buffer. Odd cols: independent except that single nibble, which is carried through unchanged.
func TestOutputContract_quantizeGroupInt4RowPadNibble(t *testing.T) {
	for _, tc := range []struct{ cols, group int }{{64, 32}, {37, 16}, {37, 7}, {33, 32}, {5, 3}, {1, 32}} {
		row := rf32(int64(tc.cols*100+tc.group), tc.cols)
		nGroups := (tc.cols + tc.group - 1) / tc.group
		bpr := (tc.cols + 1) / 2
		run := func(fill byte) ([]byte, []float32) {
			packed := make([]byte, bpr)
			scales := make([]float32, nGroups)
			for i := range packed {
				packed[i] = fill
			}
			for i := range scales {
				scales[i] = math.Float32frombits(0xFFFFFFFF) // NaN
			}
			QuantizeGroupInt4Row(row, tc.cols, tc.group, packed, scales)
			return packed, scales
		}
		pFF, sFF := run(0xFF)
		p00, s00 := run(0x00)
		for i := range sFF {
			if math.Float32bits(sFF[i]) != math.Float32bits(s00[i]) || math.IsNaN(float64(sFF[i])) {
				t.Errorf("cols=%d group=%d: scales[%d] depends on its prior contents or is NaN", tc.cols, tc.group, i)
			}
		}
		for i := range pFF {
			a, b := pFF[i], p00[i]
			if tc.cols%2 == 1 && i == bpr-1 { // the pad nibble: carried through, and only it
				if a&0x0F != b&0x0F {
					t.Errorf("cols=%d group=%d: the last byte's low (real) nibble depends on its prior contents", tc.cols, tc.group)
				}
				if a>>4 != 0xF || b>>4 != 0x0 {
					t.Errorf("cols=%d group=%d: the pad nibble is not carried through unchanged (got %#x from 0xF, %#x from 0x0)", tc.cols, tc.group, a>>4, b>>4)
				}
				continue
			}
			if a != b {
				t.Errorf("cols=%d group=%d: packed[%d] depends on its prior contents (%#x vs %#x)", tc.cols, tc.group, i, a, b)
			}
		}
	}
}

func outputContractCases() []outputcontract.Case {
	var cs []outputcontract.Case
	add := func(name string, setup func() (func(), []*outputcontract.OutBuf)) {
		cs = append(cs, outputcontract.Case{Name: name, Setup: setup})
	}

	// ---- conversions and quantizers
	for _, n := range []int{5, 37, 130} {
		add(fmt.Sprintf("F32ToF16Slice/n=%d", n), func() (func(), []*outputcontract.OutBuf) {
			src := rf32(7, n)
			dst, o := outputcontract.MkOut[uint16]("dst", n)
			return func() { F32ToF16Slice(dst, src) }, []*outputcontract.OutBuf{o}
		})
		add(fmt.Sprintf("F16ToF32Slice/n=%d", n), func() (func(), []*outputcontract.OutBuf) {
			src := F32ToF16Scales(rf32(8, n))
			dst, o := outputcontract.MkOut[float32]("dst", n)
			return func() { F16ToF32Slice(dst, src) }, []*outputcontract.OutBuf{o}
		})
		add(fmt.Sprintf("DequantizeRowInt8/n=%d", n), func() (func(), []*outputcontract.OutBuf) {
			q := ri8(9, n)
			dst, o := outputcontract.MkOut[float32]("dst", n)
			return func() { DequantizeRowInt8(q, 0.07, dst) }, []*outputcontract.OutBuf{o}
		})
		add(fmt.Sprintf("QuantizeRowInt8/n=%d", n), func() (func(), []*outputcontract.OutBuf) {
			row := rf32(10, n)
			q, o := outputcontract.MkOut[int8]("q", n)
			return func() { QuantizeRowInt8(row, q) }, []*outputcontract.OutBuf{o}
		})
		add(fmt.Sprintf("DequantizeRowInt4/cols=%d", n), func() (func(), []*outputcontract.OutBuf) {
			packed, scales := QuantizeGroupsInt4(rf32(11, n), 1, n, 16)
			dst, o := outputcontract.MkOut[float32]("dst", n)
			return func() { DequantizeRowInt4(packed, scales, 16, n, dst) }, []*outputcontract.OutBuf{o}
		})
	}
	add("DequantizeRowsInt8Into", func() (func(), []*outputcontract.OutBuf) {
		const rows, cols = 3, 37
		q, scales := QuantizeRowsInt8(rf32(12, rows*cols), rows, cols)
		dst, o := outputcontract.MkOut[float32]("dst", rows*cols)
		return func() { DequantizeRowsInt8Into(dst, q, scales, rows, cols) }, []*outputcontract.OutBuf{o}
	})
	for _, kind := range []string{"f32", "int8", "int8-weightonly", "int4"} {
		add("WeightMat.Row/"+kind, func() (func(), []*outputcontract.OutBuf) {
			const rows, cols = 5, 64
			w := rf32(13, rows*cols)
			var wm WeightMat
			switch kind {
			case "f32":
				wm = WrapF32(w, rows, cols)
			case "int8":
				wm = QuantizeInt8(w, rows, cols, true)
			case "int8-weightonly":
				wm = QuantizeInt8(w, rows, cols, false)
			default:
				wm = QuantizeInt4(w, rows, cols, 32)
			}
			dst, o := outputcontract.MkOut[float32]("dst", cols)
			return func() { wm.Row(3, dst) }, []*outputcontract.OutBuf{o}
		})
	}
	for _, mk := range [][2]int{{1, 64}, {3, 96}, {5, 37}} {
		M, K := mk[0], mk[1]
		add(fmt.Sprintf("QuantizeActivationsInto/M=%d,K=%d", M, K), func() (func(), []*outputcontract.OutBuf) {
			a := rf32(14, M*K)
			aq, o1 := outputcontract.MkOut[int8]("aq", M*K)
			sc, o2 := outputcontract.MkOut[float32]("scales", M)
			return func() { quantizeActivationsInto(aq, sc, a, M, K) }, []*outputcontract.OutBuf{o1, o2}
		})
		for _, g := range []int{16, 32} {
			ng := (K + g - 1) / g
			add(fmt.Sprintf("QuantizeActivationsGroupedInto/M=%d,K=%d,group=%d", M, K, g), func() (func(), []*outputcontract.OutBuf) {
				a := rf32(15, M*K)
				aq, o1 := outputcontract.MkOut[int8]("aq", M*K)
				sc, o2 := outputcontract.MkOut[float32]("scales", M*ng)
				return func() { QuantizeActivationsGroupedInto(aq, sc, a, M, K, g) }, []*outputcontract.OutBuf{o1, o2}
			})
			add(fmt.Sprintf("SumActGroupsInto/M=%d,K=%d,group=%d", M, K, g), func() (func(), []*outputcontract.OutBuf) {
				aq0 := ri8(16, M*K)
				sum, o := outputcontract.MkOut[int32]("sumAct", M*ng)
				return func() { sumActGroupsInto(sum, aq0, M, K, g) }, []*outputcontract.OutBuf{o}
			})
		}
	}

	// ---- dense and quantized matmuls
	type shape struct{ M, K, N int }
	shapes := []shape{{1, 64, 10}, {3, 96, 19}, {8, 64, 33}}
	for _, s := range shapes {
		nm := fmt.Sprintf("M=%d,K=%d,N=%d", s.M, s.K, s.N)
		a := func() []float32 { return rf32(20, s.M*s.K) }
		b := func() []float32 { return rf32(21, s.N*s.K) }
		f32mm := map[string]func(dst, a, b []float32){
			"MatmulBT":                        func(d, a, b []float32) { MatmulBT(a, b, d, s.M, s.K, s.N) },
			"Workspace.MatmulBT":              func(d, a, b []float32) { (&Workspace{}).MatmulBT(a, b, d, s.M, s.K, s.N) },
			"MatmulBTInto":                    func(d, a, b []float32) { MatmulBTInto(d, a, b, s.M, s.K, s.N) },
			"MatmulBTAcc64":                   func(d, a, b []float32) { MatmulBTAcc64(a, b, d, s.M, s.K, s.N) },
			"Workspace.MatmulBTAcc64":         func(d, a, b []float32) { (&Workspace{}).matmulBTAcc64(a, b, d, s.M, s.K, s.N) },
			"MatmulBTAcc64Strided/contiguous": func(d, a, b []float32) { MatmulBTAcc64Strided(a, b, d, s.M, s.K, s.N, 0, s.K, 1) },
			"MatmulBTAcc64Strided/transposed": func(d, a, b []float32) { MatmulBTAcc64Strided(a, b, d, s.M, s.K, s.N, 0, 1, s.N) },
			"Workspace.MatmulBTAcc64Strided":  func(d, a, b []float32) { (&Workspace{}).matmulBTAcc64Strided(a, b, d, s.M, s.K, s.N, 0, s.K, 1) },
		}
		for name, f := range f32mm {
			add(name+"/"+nm, func() (func(), []*outputcontract.OutBuf) {
				dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
				aa, bb := a(), b()
				return func() { f(dst, aa, bb) }, []*outputcontract.OutBuf{o}
			})
		}
		for _, wk := range []string{"int8", "int8-weightonly", "int4", "f32"} {
			add("WeightMat.MatmulBT/"+wk+"/"+nm, func() (func(), []*outputcontract.OutBuf) {
				wm := weightMatKind(wk, rf32(22, s.N*s.K), s.N, s.K)
				dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
				aa := a()
				return func() { wm.MatmulBT(aa, dst, s.M) }, []*outputcontract.OutBuf{o}
			})
			add("WeightMat.MatmulBTInto/"+wk+"/"+nm, func() (func(), []*outputcontract.OutBuf) {
				wm := weightMatKind(wk, rf32(22, s.N*s.K), s.N, s.K)
				dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
				aa := a()
				return func() { wm.MatmulBTInto(&Workspace{}, aa, dst, s.M) }, []*outputcontract.OutBuf{o}
			})
			if wk == "int4" {
				add("WeightMat.MatmulBTW4A8Into/"+nm, func() (func(), []*outputcontract.OutBuf) {
					wm := weightMatKind(wk, rf32(22, s.N*s.K), s.N, s.K)
					dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
					aa := a()
					return func() { wm.MatmulBTW4A8Into(&Workspace{}, aa, dst, s.M) }, []*outputcontract.OutBuf{o}
				})
			}
		}
		// int8 weights (per-row) for the Q8 / W8A8 family
		q8w := func() ([]int8, []float32) { return QuantizeRowsInt8(rf32(23, s.N*s.K), s.N, s.K) }
		add("MatmulBTQ8/"+nm, func() (func(), []*outputcontract.OutBuf) {
			bq, bs := q8w()
			dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
			aa := a()
			return func() { MatmulBTQ8(aa, bq, bs, dst, s.M, s.K, s.N) }, []*outputcontract.OutBuf{o}
		})
		add("MatmulBTQ8Into/"+nm, func() (func(), []*outputcontract.OutBuf) {
			bq, bs := q8w()
			dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
			aa := a()
			return func() { MatmulBTQ8Into(&Workspace{}, aa, bq, bs, dst, s.M, s.K, s.N) }, []*outputcontract.OutBuf{o}
		})
		add("MatmulBTQ8Fused/"+nm, func() (func(), []*outputcontract.OutBuf) {
			bq, bs := q8w()
			dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
			aa := a()
			return func() { MatmulBTQ8Fused(dst, aa, bq, bs, s.M, s.K, s.N) }, []*outputcontract.OutBuf{o}
		})
		add("MatmulBTQ8FusedInto/"+nm, func() (func(), []*outputcontract.OutBuf) {
			bq, bs := q8w()
			dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
			aa := a()
			return func() { MatmulBTQ8FusedInto(dst, aa, bq, bs, s.M, s.K, s.N) }, []*outputcontract.OutBuf{o}
		})
		add("MatmulBTW8A8/"+nm, func() (func(), []*outputcontract.OutBuf) {
			bq, bs := q8w()
			dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
			aa := a()
			return func() { MatmulBTW8A8(aa, bq, bs, dst, s.M, s.K, s.N) }, []*outputcontract.OutBuf{o}
		})
		add("MatmulBTW8A8Into/"+nm, func() (func(), []*outputcontract.OutBuf) {
			bq, bs := q8w()
			dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
			aa := a()
			return func() { MatmulBTW8A8Into(&Workspace{}, aa, bq, bs, dst, s.M, s.K, s.N) }, []*outputcontract.OutBuf{o}
		})
		add("MatmulBTW8A8Pre/"+nm, func() (func(), []*outputcontract.OutBuf) {
			bq, bs := q8w()
			aq := make([]int8, s.M*s.K)
			asc := make([]float32, s.M)
			quantizeActivationsInto(aq, asc, a(), s.M, s.K)
			dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
			return func() { MatmulBTW8A8Pre(&Workspace{}, aq, asc, bq, bs, dst, s.M, s.K, s.N) }, []*outputcontract.OutBuf{o}
		})
		add("MatmulBTW8A8Batch/"+nm, func() (func(), []*outputcontract.OutBuf) {
			bq1, bs1 := QuantizeRowsInt8(rf32(24, s.N*s.K), s.N, s.K)
			bq2, bs2 := QuantizeRowsInt8(rf32(25, 5*s.K), 5, s.K)
			d1, o1 := outputcontract.MkOut[float32]("op0.Dst", s.M*s.N)
			d2, o2 := outputcontract.MkOut[float32]("op1.Dst", s.M*5)
			ops := []W8A8Op{{BQ: bq1, Scales: bs1, Dst: d1, N: s.N}, {BQ: bq2, Scales: bs2, Dst: d2, N: 5}}
			aa := a()
			return func() { MatmulBTW8A8Batch(&Workspace{}, aa, s.M, s.K, ops) }, []*outputcontract.OutBuf{o1, o2}
		})
		// int4 (group 32) for the Q4 / W4A8 family
		for _, g := range []int{32} {
			q4w := func() ([]byte, []float32) { return QuantizeGroupsInt4(rf32(26, s.N*s.K), s.N, s.K, g) }
			add(fmt.Sprintf("MatmulBTQ4/%s,group=%d", nm, g), func() (func(), []*outputcontract.OutBuf) {
				pk, sc := q4w()
				dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
				aa := a()
				return func() { MatmulBTQ4(aa, pk, sc, dst, s.M, s.K, s.N, g) }, []*outputcontract.OutBuf{o}
			})
			add(fmt.Sprintf("MatmulBTQ4Into/%s,group=%d", nm, g), func() (func(), []*outputcontract.OutBuf) {
				pk, sc := q4w()
				dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
				aa := a()
				return func() { MatmulBTQ4Into(&Workspace{}, aa, pk, sc, dst, s.M, s.K, s.N, g) }, []*outputcontract.OutBuf{o}
			})
			add(fmt.Sprintf("MatmulBTW4A8/%s,group=%d", nm, g), func() (func(), []*outputcontract.OutBuf) {
				pk, sc := q4w()
				dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
				aa := a()
				return func() { MatmulBTW4A8(aa, pk, sc, dst, s.M, s.K, s.N, g) }, []*outputcontract.OutBuf{o}
			})
			add(fmt.Sprintf("MatmulBTW4A8Into/%s,group=%d", nm, g), func() (func(), []*outputcontract.OutBuf) {
				pk, sc := q4w()
				dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
				aa := a()
				return func() { MatmulBTW4A8Into(&Workspace{}, aa, pk, sc, dst, s.M, s.K, s.N, g) }, []*outputcontract.OutBuf{o}
			})
			add(fmt.Sprintf("MatmulBTW4A8F16Into/%s,group=%d", nm, g), func() (func(), []*outputcontract.OutBuf) {
				pk, sc := q4w()
				sc16 := F32ToF16Scales(sc)
				dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
				aa := a()
				return func() { MatmulBTW4A8F16Into(&Workspace{}, aa, pk, sc16, dst, s.M, s.K, s.N, g) }, []*outputcontract.OutBuf{o}
			})
			add(fmt.Sprintf("MatmulBTW4A8Batch/%s,group=%d", nm, g), func() (func(), []*outputcontract.OutBuf) {
				pk1, sc1 := QuantizeGroupsInt4(rf32(27, s.N*s.K), s.N, s.K, g)
				pk2, sc2 := QuantizeGroupsInt4(rf32(28, 5*s.K), 5, s.K, g)
				d1, o1 := outputcontract.MkOut[float32]("op0.Dst", s.M*s.N)
				d2, o2 := outputcontract.MkOut[float32]("op1.Dst", s.M*5)
				ops := []W4A8Op{{W4: pk1, Scales: sc1, Dst: d1, N: s.N}, {W4: pk2, Scales: sc2, Dst: d2, N: 5}}
				aa := a()
				return func() { MatmulBTW4A8Batch(&Workspace{}, aa, s.M, s.K, g, ops) }, []*outputcontract.OutBuf{o1, o2}
			})
		}
	}
	// GGUF K-quant GEMVs: K a multiple of 256.
	for _, q := range []string{"Q4K", "Q6K"} {
		add("MatmulBTGGUF"+q, func() (func(), []*outputcontract.OutBuf) {
			const M, K, N = 2, 512, 5
			nb := K / qkK
			rng := rand.New(rand.NewSource(31))
			a := rf32(32, M*K)
			dst, o := outputcontract.MkOut[float32]("dst", M*N)
			if q == "Q4K" {
				w := randQ4KRow(rng, N*nb)
				return func() { MatmulBTGGUFQ4K(a, w, dst, M, K, N) }, []*outputcontract.OutBuf{o}
			}
			w := randQ6KRow(rng, N*nb)
			return func() { MatmulBTGGUFQ6K(a, w, dst, M, K, N) }, []*outputcontract.OutBuf{o}
		})
	}

	// ---- the Dot tiles (they write a sums array)
	for _, n4 := range []int{1, 5, 16} {
		add(fmt.Sprintf("Dot4x4/n4=%d", n4), func() (func(), []*outputcontract.OutBuf) {
			rs := make([][]float32, 5)
			for i := range rs {
				rs[i] = rf32(int64(70+i), n4*4)
			}
			s, o := outputcontract.MkOut[float32]("sums", 16)
			return func() { dot4x4(&rs[0][0], &rs[1][0], &rs[2][0], &rs[3][0], &rs[4][0], n4, (*[16]float32)(s)) }, []*outputcontract.OutBuf{o}
		})
		add(fmt.Sprintf("Dot8x4/n4=%d", n4), func() (func(), []*outputcontract.OutBuf) {
			rs := make([][]float32, 9)
			for i := range rs {
				rs[i] = rf32(int64(80+i), n4*4)
			}
			s, o := outputcontract.MkOut[float32]("sums", 32)
			return func() {
				Dot8x4(&rs[0][0], &rs[1][0], &rs[2][0], &rs[3][0], &rs[4][0], &rs[5][0], &rs[6][0], &rs[7][0], &rs[8][0], n4, (*[32]float32)(s))
			}, []*outputcontract.OutBuf{o}
		})
		add(fmt.Sprintf("Dot2x8/n4=%d", n4), func() (func(), []*outputcontract.OutBuf) {
			rs := make([][]float32, 10)
			for i := range rs {
				rs[i] = rf32(int64(90+i), n4*4)
			}
			s, o := outputcontract.MkOut[float32]("sums", 64)
			return func() {
				Dot2x8(&rs[0][0], &rs[1][0], &rs[2][0], &rs[3][0], &rs[4][0], &rs[5][0], &rs[6][0], &rs[7][0], &rs[8][0], &rs[9][0], n4, (*[64]float32)(s))
			}, []*outputcontract.OutBuf{o}
		})
	}

	// ---- the split-half row repack is portable Go (and the arm64 quad repack is built from it).
	for _, K := range []int{32, 96} {
		add(fmt.Sprintf("RepackInt4SplitHalfRow/K=%d", K), func() (func(), []*outputcontract.OutBuf) {
			src := ri8(int64(K), K/2)
			sb := make([]byte, len(src))
			for i, v := range src {
				sb[i] = byte(v)
			}
			dst, o := outputcontract.MkOut[byte]("dst", K/2)
			return func() { RepackInt4SplitHalfRow(dst, sb, K) }, []*outputcontract.OutBuf{o}
		})
	}

	// ---- arch-specific writers: outputcontract_{arm64,amd64,other}_test.go
	cs = append(cs, archContractCases()...)
	return cs
}

func weightMatKind(kind string, w []float32, rows, cols int) *WeightMat {
	var wm WeightMat
	switch kind {
	case "f32":
		wm = WrapF32(w, rows, cols)
	case "int8":
		wm = QuantizeInt8(w, rows, cols, true)
	case "int8-weightonly":
		wm = QuantizeInt8(w, rows, cols, false)
	default:
		wm = QuantizeInt4(w, rows, cols, 32)
	}
	return &wm
}
