package linalg

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// R-11 (goinfer docs/tasks/task-recompute-audit.md §5): the OUTPUT CONTRACT of every exported aikit writer, as a gate rather than a comment. For each kernel that writes into a caller's buffer, run it with
// that buffer filled three ways (0xFF bytes, which is NaN in a float; zeros; a finite 0x5A pattern) and require
//
//   - the covered region holds no NaN afterwards (the poison test: nothing the kernel covers was left as it found it),
//   - the three results are byte-identical (the result does not depend on what the buffer held, so a caller that zeroes first is doing dead work, and one that does not is correct), and
//   - the bytes just past the region are untouched.
//
// A kernel that ACCUMULATES into dst would fail the second check, and none does. The scratch buffers a kernel takes (the f64 accumulators, the fused-attention scratch) are filled too: the result must not
// depend on them either. The doc comment of each writer states which of "overwrites", "zeroes internally" or "accumulates" it is, and TestOutputContract_everyWriterIsDocumented keeps those lines present.
//
// What this does not prove: that the covered region is the right region (the matmul parity gates own that), or that an arch's assembly is correct (the arch-tagged gates own that). Run it per build:
//
//	go test ./linalg/ -run TestOutputContract                       # the host's default kernels (AVX2/VNNI on amd64, NEON on arm64)
//	GOARCH=386 go test ./linalg/ -run TestOutputContract            # the generic (!amd64 && !arm64) kernels, natively on an amd64 Linux box
//	go test -tags aikit_noavx2 ./linalg/ -run TestOutputContract    # the narrower amd64 tiers; likewise aikit_noavx512, aikit_nodotprod (arm64)
const contractGuard = 16 // elements of guard after every output region

type outBuf struct {
	name  string
	cov   []byte // the region the kernel is documented to cover
	guard []byte // the bytes just past it
	f32   bool
	f64   bool
	// scratch marks a work buffer, not an output: it is poisoned (the outputs must not depend on its contents) and guarded, but its own contents afterwards are not compared, since a kernel may use only a prefix of it.
	scratch bool
}

func (o *outBuf) asScratch() *outBuf { o.scratch = true; return o }

func mkOut[T any](name string, n int) ([]T, *outBuf) {
	back := make([]T, n+contractGuard)
	var z T
	sz := int(unsafe.Sizeof(z))
	all := unsafe.Slice((*byte)(unsafe.Pointer(&back[0])), len(back)*sz)
	ob := &outBuf{name: name, cov: all[:n*sz], guard: all[n*sz:]}
	switch any(z).(type) {
	case float32:
		ob.f32 = true
	case float64:
		ob.f64 = true
	}
	return back[:n:n], ob
}

type contractCase struct {
	name  string
	setup func() (call func(), outs []*outBuf)
}

func (o *outBuf) hasNaN() bool {
	switch {
	case o.f32:
		for i := 0; i+4 <= len(o.cov); i += 4 {
			if math.IsNaN(float64(math.Float32frombits(uint32(o.cov[i]) | uint32(o.cov[i+1])<<8 | uint32(o.cov[i+2])<<16 | uint32(o.cov[i+3])<<24))) {
				return true
			}
		}
	case o.f64:
		for i := 0; i+8 <= len(o.cov); i += 8 {
			var u uint64
			for b := 0; b < 8; b++ {
				u |= uint64(o.cov[i+b]) << (8 * b)
			}
			if math.IsNaN(math.Float64frombits(u)) {
				return true
			}
		}
	}
	return false
}

// contractViolations runs c with its outputs filled three ways and returns every way it broke the contract (nil = sound).
func contractViolations(c contractCase) []string {
	var bad []string
	var results [][]byte
	for _, fill := range []byte{0xFF, 0x00, 0x5A} {
		call, outs := c.setup()
		for _, o := range outs {
			for i := range o.cov {
				o.cov[i] = fill
			}
			for i := range o.guard {
				o.guard[i] = 0xA7
			}
		}
		call()
		var snap []byte
		for _, o := range outs {
			for i, b := range o.guard {
				if b != 0xA7 {
					bad = append(bad, fmt.Sprintf("%s: wrote past the end of %s (guard byte %d is %#x)", c.name, o.name, i, b))
					break
				}
			}
			if o.scratch {
				continue
			}
			if fill == 0xFF && o.hasNaN() {
				bad = append(bad, fmt.Sprintf("%s: a NaN survives in %s after the call with it poisoned: the kernel leaves part of what it covers as it found it", c.name, o.name))
			}
			snap = append(snap, o.cov...)
		}
		results = append(results, snap)
	}
	if !bytes.Equal(results[0], results[1]) || !bytes.Equal(results[1], results[2]) {
		bad = append(bad, fmt.Sprintf("%s: the result depends on what the output buffers held before the call (0xFF vs 0x00 vs 0x5A fill differ)", c.name))
	}
	return bad
}

func runContract(t *testing.T, c contractCase) {
	t.Helper()
	for _, m := range contractViolations(c) {
		t.Error(m)
	}
}

// TestOutputContract_harnessCanFail shows the gate goes red: a kernel that skips one element, one that accumulates into dst, one that writes past its region, and a scratch-dependent one are each caught,
// and the same shapes done right are not.
func TestOutputContract_harnessCanFail(t *testing.T) {
	const n = 37
	mk := func(name string, kernel func(dst, scratch []float32, src []float32)) contractCase {
		return contractCase{name, func() (func(), []*outBuf) {
			src := rf32(1, n)
			dst, o := mkOut[float32]("dst", n)
			scr, so := mkOut[float32]("scratch", n)
			return func() { kernel(dst, scr, src) }, []*outBuf{o, so.asScratch()}
		}}
	}
	sound := func(dst, _, src []float32) {
		for i := range dst {
			dst[i] = src[i] * 2
		}
	}
	if v := contractViolations(mk("sound", sound)); len(v) != 0 {
		t.Fatalf("a sound kernel was flagged: %v", v)
	}
	for _, bad := range []struct {
		name string
		k    func(dst, scratch []float32, src []float32)
	}{
		{"skips-the-last-element", func(dst, _, src []float32) {
			for i := 0; i < len(dst)-1; i++ {
				dst[i] = src[i] * 2
			}
		}},
		{"accumulates", func(dst, _, src []float32) {
			for i := range dst {
				dst[i] += src[i] * 2
			}
		}},
		{"overruns", func(dst, _, src []float32) {
			for i := range dst {
				dst[i] = src[i] * 2
			}
			*(*float32)(unsafe.Add(unsafe.Pointer(&dst[0]), len(dst)*4)) = 1 // one past the region
		}},
		{"reads-scratch", func(dst, scr, src []float32) {
			for i := range dst {
				dst[i] = src[i]*2 + scr[0]*0 + float32(math.Float32bits(scr[0])&1)
			}
		}},
	} {
		if v := contractViolations(mk(bad.name, bad.k)); len(v) == 0 {
			t.Errorf("%s was NOT flagged: the gate cannot go red for it", bad.name)
		}
	}
}

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
		t.Run(c.name, func(t *testing.T) { runContract(t, c) })
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

func outputContractCases() []contractCase {
	var cs []contractCase
	add := func(name string, setup func() (func(), []*outBuf)) { cs = append(cs, contractCase{name, setup}) }

	// ---- elementwise and row kernels: dst <- f(src); they may alias, which is one more reason a pre-zero of dst is harmful.
	type ew struct {
		name string
		f    func(dst, src []float32)
	}
	for _, k := range []ew{
		{"ExpF32Into", ExpF32Into}, {"SoftmaxRowInto", SoftmaxRowInto}, {"GELUInto", GELUInto}, {"SiLUInto", SiLUInto}, {"GELUTanhInto", GELUTanhInto}, {"TanhInto", TanhInto},
		{"ExpContractInto", ExpContractInto}, {"GELUContractInto", GELUContractInto}, {"GELUTanhContractInto", GELUTanhContractInto}, {"SiLUContractInto", SiLUContractInto}, {"SoftmaxRowContractInto", SoftmaxRowContractInto},
		{"SoftmaxRowScaledInto", func(d, s []float32) { SoftmaxRowScaledInto(d, s, 0.35) }}, {"SoftmaxRowScaledContractInto", func(d, s []float32) { SoftmaxRowScaledContractInto(d, s, 0.35) }},
	} {
		for _, n := range []int{1, 5, 37, 130} {
			k, n := k, n
			add(fmt.Sprintf("%s/n=%d", k.name, n), func() (func(), []*outBuf) {
				src := rf32(int64(n), n)
				dst, o := mkOut[float32]("dst", n)
				return func() { k.f(dst, src) }, []*outBuf{o}
			})
		}
	}

	// ---- conversions and quantizers
	for _, n := range []int{5, 37, 130} {
		n := n
		add(fmt.Sprintf("F32ToF16Slice/n=%d", n), func() (func(), []*outBuf) {
			src := rf32(7, n)
			dst, o := mkOut[uint16]("dst", n)
			return func() { F32ToF16Slice(dst, src) }, []*outBuf{o}
		})
		add(fmt.Sprintf("F16ToF32Slice/n=%d", n), func() (func(), []*outBuf) {
			src := F32ToF16Scales(rf32(8, n))
			dst, o := mkOut[float32]("dst", n)
			return func() { F16ToF32Slice(dst, src) }, []*outBuf{o}
		})
		add(fmt.Sprintf("DequantizeRowInt8/n=%d", n), func() (func(), []*outBuf) {
			q := ri8(9, n)
			dst, o := mkOut[float32]("dst", n)
			return func() { DequantizeRowInt8(q, 0.07, dst) }, []*outBuf{o}
		})
		add(fmt.Sprintf("QuantizeRowInt8/n=%d", n), func() (func(), []*outBuf) {
			row := rf32(10, n)
			q, o := mkOut[int8]("q", n)
			return func() { QuantizeRowInt8(row, q) }, []*outBuf{o}
		})
		add(fmt.Sprintf("DequantizeRowInt4/cols=%d", n), func() (func(), []*outBuf) {
			packed, scales := QuantizeGroupsInt4(rf32(11, n), 1, n, 16)
			dst, o := mkOut[float32]("dst", n)
			return func() { DequantizeRowInt4(packed, scales, 16, n, dst) }, []*outBuf{o}
		})
	}
	add("DequantizeRowsInt8Into", func() (func(), []*outBuf) {
		const rows, cols = 3, 37
		q, scales := QuantizeRowsInt8(rf32(12, rows*cols), rows, cols)
		dst, o := mkOut[float32]("dst", rows*cols)
		return func() { DequantizeRowsInt8Into(dst, q, scales, rows, cols) }, []*outBuf{o}
	})
	for _, kind := range []string{"f32", "int8", "int8-weightonly", "int4"} {
		kind := kind
		add("WeightMat.Row/"+kind, func() (func(), []*outBuf) {
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
			dst, o := mkOut[float32]("dst", cols)
			return func() { wm.Row(3, dst) }, []*outBuf{o}
		})
	}
	for _, mk := range [][2]int{{1, 64}, {3, 96}, {5, 37}} {
		M, K := mk[0], mk[1]
		add(fmt.Sprintf("QuantizeActivationsInto/M=%d,K=%d", M, K), func() (func(), []*outBuf) {
			a := rf32(14, M*K)
			aq, o1 := mkOut[int8]("aq", M*K)
			sc, o2 := mkOut[float32]("scales", M)
			return func() { QuantizeActivationsInto(aq, sc, a, M, K) }, []*outBuf{o1, o2}
		})
		for _, g := range []int{16, 32} {
			g := g
			ng := (K + g - 1) / g
			add(fmt.Sprintf("QuantizeActivationsGroupedInto/M=%d,K=%d,group=%d", M, K, g), func() (func(), []*outBuf) {
				a := rf32(15, M*K)
				aq, o1 := mkOut[int8]("aq", M*K)
				sc, o2 := mkOut[float32]("scales", M*ng)
				return func() { QuantizeActivationsGroupedInto(aq, sc, a, M, K, g) }, []*outBuf{o1, o2}
			})
			add(fmt.Sprintf("SumActGroupsInto/M=%d,K=%d,group=%d", M, K, g), func() (func(), []*outBuf) {
				aq0 := ri8(16, M*K)
				sum, o := mkOut[int32]("sumAct", M*ng)
				return func() { SumActGroupsInto(sum, aq0, M, K, g) }, []*outBuf{o}
			})
		}
	}

	// ---- dense and quantized matmuls
	type shape struct{ M, K, N int }
	shapes := []shape{{1, 64, 10}, {3, 96, 19}, {8, 64, 33}}
	for _, s := range shapes {
		s := s
		nm := fmt.Sprintf("M=%d,K=%d,N=%d", s.M, s.K, s.N)
		a := func() []float32 { return rf32(20, s.M*s.K) }
		b := func() []float32 { return rf32(21, s.N*s.K) }
		f32mm := map[string]func(dst, a, b []float32){
			"MatmulBT":                        func(d, a, b []float32) { MatmulBT(a, b, d, s.M, s.K, s.N) },
			"Workspace.MatmulBT":              func(d, a, b []float32) { (&Workspace{}).MatmulBT(a, b, d, s.M, s.K, s.N) },
			"MatmulBTInto":                    func(d, a, b []float32) { MatmulBTInto(d, a, b, s.M, s.K, s.N) },
			"MatmulBTAcc64":                   func(d, a, b []float32) { MatmulBTAcc64(a, b, d, s.M, s.K, s.N) },
			"Workspace.MatmulBTAcc64":         func(d, a, b []float32) { (&Workspace{}).MatmulBTAcc64(a, b, d, s.M, s.K, s.N) },
			"MatmulBTAcc64Strided/contiguous": func(d, a, b []float32) { MatmulBTAcc64Strided(a, b, d, s.M, s.K, s.N, 0, s.K, 1) },
			"MatmulBTAcc64Strided/transposed": func(d, a, b []float32) { MatmulBTAcc64Strided(a, b, d, s.M, s.K, s.N, 0, 1, s.N) },
			"Workspace.MatmulBTAcc64Strided":  func(d, a, b []float32) { (&Workspace{}).MatmulBTAcc64Strided(a, b, d, s.M, s.K, s.N, 0, s.K, 1) },
			"MatmulQKAcc64":                   func(d, a, b []float32) { MatmulQKAcc64(a, b, d, s.M, s.K, s.N, 0, s.K) },
		}
		for name, f := range f32mm {
			f := f
			add(name+"/"+nm, func() (func(), []*outBuf) {
				dst, o := mkOut[float32]("dst", s.M*s.N)
				aa, bb := a(), b()
				return func() { f(dst, aa, bb) }, []*outBuf{o}
			})
		}
		for _, wk := range []string{"int8", "int8-weightonly", "int4", "f32"} {
			wk := wk
			add("WeightMat.MatmulBT/"+wk+"/"+nm, func() (func(), []*outBuf) {
				wm := weightMatKind(wk, rf32(22, s.N*s.K), s.N, s.K)
				dst, o := mkOut[float32]("dst", s.M*s.N)
				aa := a()
				return func() { wm.MatmulBT(aa, dst, s.M) }, []*outBuf{o}
			})
			add("WeightMat.MatmulBTInto/"+wk+"/"+nm, func() (func(), []*outBuf) {
				wm := weightMatKind(wk, rf32(22, s.N*s.K), s.N, s.K)
				dst, o := mkOut[float32]("dst", s.M*s.N)
				aa := a()
				return func() { wm.MatmulBTInto(&Workspace{}, aa, dst, s.M) }, []*outBuf{o}
			})
			if wk == "int4" {
				add("WeightMat.MatmulBTW4A8Into/"+nm, func() (func(), []*outBuf) {
					wm := weightMatKind(wk, rf32(22, s.N*s.K), s.N, s.K)
					dst, o := mkOut[float32]("dst", s.M*s.N)
					aa := a()
					return func() { wm.MatmulBTW4A8Into(&Workspace{}, aa, dst, s.M) }, []*outBuf{o}
				})
			}
		}
		// int8 weights (per-row) for the Q8 / W8A8 family
		q8w := func() ([]int8, []float32) { return QuantizeRowsInt8(rf32(23, s.N*s.K), s.N, s.K) }
		add("MatmulBTQ8/"+nm, func() (func(), []*outBuf) {
			bq, bs := q8w()
			dst, o := mkOut[float32]("dst", s.M*s.N)
			aa := a()
			return func() { MatmulBTQ8(aa, bq, bs, dst, s.M, s.K, s.N) }, []*outBuf{o}
		})
		add("MatmulBTQ8Into/"+nm, func() (func(), []*outBuf) {
			bq, bs := q8w()
			dst, o := mkOut[float32]("dst", s.M*s.N)
			aa := a()
			return func() { MatmulBTQ8Into(&Workspace{}, aa, bq, bs, dst, s.M, s.K, s.N) }, []*outBuf{o}
		})
		add("MatmulBTQ8Fused/"+nm, func() (func(), []*outBuf) {
			bq, bs := q8w()
			dst, o := mkOut[float32]("dst", s.M*s.N)
			aa := a()
			return func() { MatmulBTQ8Fused(dst, aa, bq, bs, s.M, s.K, s.N) }, []*outBuf{o}
		})
		add("MatmulBTQ8FusedInto/"+nm, func() (func(), []*outBuf) {
			bq, bs := q8w()
			dst, o := mkOut[float32]("dst", s.M*s.N)
			aa := a()
			return func() { MatmulBTQ8FusedInto(dst, aa, bq, bs, s.M, s.K, s.N) }, []*outBuf{o}
		})
		add("MatmulBTW8A8/"+nm, func() (func(), []*outBuf) {
			bq, bs := q8w()
			dst, o := mkOut[float32]("dst", s.M*s.N)
			aa := a()
			return func() { MatmulBTW8A8(aa, bq, bs, dst, s.M, s.K, s.N) }, []*outBuf{o}
		})
		add("MatmulBTW8A8Into/"+nm, func() (func(), []*outBuf) {
			bq, bs := q8w()
			dst, o := mkOut[float32]("dst", s.M*s.N)
			aa := a()
			return func() { MatmulBTW8A8Into(&Workspace{}, aa, bq, bs, dst, s.M, s.K, s.N) }, []*outBuf{o}
		})
		add("MatmulBTW8A8Pre/"+nm, func() (func(), []*outBuf) {
			bq, bs := q8w()
			aq := make([]int8, s.M*s.K)
			asc := make([]float32, s.M)
			QuantizeActivationsInto(aq, asc, a(), s.M, s.K)
			dst, o := mkOut[float32]("dst", s.M*s.N)
			return func() { MatmulBTW8A8Pre(&Workspace{}, aq, asc, bq, bs, dst, s.M, s.K, s.N) }, []*outBuf{o}
		})
		add("MatmulBTW8A8Batch/"+nm, func() (func(), []*outBuf) {
			bq1, bs1 := QuantizeRowsInt8(rf32(24, s.N*s.K), s.N, s.K)
			bq2, bs2 := QuantizeRowsInt8(rf32(25, 5*s.K), 5, s.K)
			d1, o1 := mkOut[float32]("op0.Dst", s.M*s.N)
			d2, o2 := mkOut[float32]("op1.Dst", s.M*5)
			ops := []W8A8Op{{BQ: bq1, Scales: bs1, Dst: d1, N: s.N}, {BQ: bq2, Scales: bs2, Dst: d2, N: 5}}
			aa := a()
			return func() { MatmulBTW8A8Batch(&Workspace{}, aa, s.M, s.K, ops) }, []*outBuf{o1, o2}
		})
		// int4 (group 32) for the Q4 / W4A8 family
		for _, g := range []int{32} {
			g := g
			q4w := func() ([]byte, []float32) { return QuantizeGroupsInt4(rf32(26, s.N*s.K), s.N, s.K, g) }
			add(fmt.Sprintf("MatmulBTQ4/%s,group=%d", nm, g), func() (func(), []*outBuf) {
				pk, sc := q4w()
				dst, o := mkOut[float32]("dst", s.M*s.N)
				aa := a()
				return func() { MatmulBTQ4(aa, pk, sc, dst, s.M, s.K, s.N, g) }, []*outBuf{o}
			})
			add(fmt.Sprintf("MatmulBTQ4Into/%s,group=%d", nm, g), func() (func(), []*outBuf) {
				pk, sc := q4w()
				dst, o := mkOut[float32]("dst", s.M*s.N)
				aa := a()
				return func() { MatmulBTQ4Into(&Workspace{}, aa, pk, sc, dst, s.M, s.K, s.N, g) }, []*outBuf{o}
			})
			add(fmt.Sprintf("MatmulBTW4A8/%s,group=%d", nm, g), func() (func(), []*outBuf) {
				pk, sc := q4w()
				dst, o := mkOut[float32]("dst", s.M*s.N)
				aa := a()
				return func() { MatmulBTW4A8(aa, pk, sc, dst, s.M, s.K, s.N, g) }, []*outBuf{o}
			})
			add(fmt.Sprintf("MatmulBTW4A8Into/%s,group=%d", nm, g), func() (func(), []*outBuf) {
				pk, sc := q4w()
				dst, o := mkOut[float32]("dst", s.M*s.N)
				aa := a()
				return func() { MatmulBTW4A8Into(&Workspace{}, aa, pk, sc, dst, s.M, s.K, s.N, g) }, []*outBuf{o}
			})
			add(fmt.Sprintf("MatmulBTW4A8F16Into/%s,group=%d", nm, g), func() (func(), []*outBuf) {
				pk, sc := q4w()
				sc16 := F32ToF16Scales(sc)
				dst, o := mkOut[float32]("dst", s.M*s.N)
				aa := a()
				return func() { MatmulBTW4A8F16Into(&Workspace{}, aa, pk, sc16, dst, s.M, s.K, s.N, g) }, []*outBuf{o}
			})
			add(fmt.Sprintf("MatmulBTW4A8Batch/%s,group=%d", nm, g), func() (func(), []*outBuf) {
				pk1, sc1 := QuantizeGroupsInt4(rf32(27, s.N*s.K), s.N, s.K, g)
				pk2, sc2 := QuantizeGroupsInt4(rf32(28, 5*s.K), 5, s.K, g)
				d1, o1 := mkOut[float32]("op0.Dst", s.M*s.N)
				d2, o2 := mkOut[float32]("op1.Dst", s.M*5)
				ops := []W4A8Op{{W4: pk1, Scales: sc1, Dst: d1, N: s.N}, {W4: pk2, Scales: sc2, Dst: d2, N: 5}}
				aa := a()
				return func() { MatmulBTW4A8Batch(&Workspace{}, aa, s.M, s.K, g, ops) }, []*outBuf{o1, o2}
			})
		}
	}
	// GGUF K-quant GEMVs: K a multiple of 256.
	for _, q := range []string{"Q4K", "Q6K"} {
		q := q
		add("MatmulBTGGUF"+q, func() (func(), []*outBuf) {
			const M, K, N = 2, 512, 5
			nb := K / qkK
			rng := rand.New(rand.NewSource(31))
			a := rf32(32, M*K)
			dst, o := mkOut[float32]("dst", M*N)
			if q == "Q4K" {
				w := randQ4KRow(rng, N*nb)
				return func() { MatmulBTGGUFQ4K(a, w, dst, M, K, N) }, []*outBuf{o}
			}
			w := randQ6KRow(rng, N*nb)
			return func() { MatmulBTGGUFQ6K(a, w, dst, M, K, N) }, []*outBuf{o}
		})
	}

	// ---- attention kernels (f64 accumulated): scratch is part of the contract too
	for _, g := range [][4]int{{1, 20, 16, 0}, {3, 33, 64, 16}, {6, 70, 64, 64}} {
		M, nKeys, hd, headOff := g[0], g[1], g[2], g[3]
		rowStride := headOff + hd + 8
		nm := fmt.Sprintf("M=%d,nKeys=%d,hd=%d,headOff=%d", M, nKeys, hd, headOff)
		add("MatmulAVAcc64/"+nm, func() (func(), []*outBuf) {
			scores, vals := rf32(40, M*nKeys), rf32(41, nKeys*rowStride)
			dst, o1 := mkOut[float32]("dst", M*hd)
			acc, o2 := mkOut[float64]("acc", hd)
			return func() { MatmulAVAcc64(scores, vals, dst, acc, M, nKeys, hd, headOff, rowStride) }, []*outBuf{o1, o2.asScratch()}
		})
		add("MatmulAVAcc64Group/"+nm, func() (func(), []*outBuf) {
			scores, vals := rf32(42, M*nKeys), rf32(43, nKeys*rowStride)
			dst, o1 := mkOut[float32]("dst", M*hd)
			acc, o2 := mkOut[float64]("acc", M*hd)
			return func() { MatmulAVAcc64Group(scores, vals, dst, acc, M, nKeys, hd, headOff, rowStride) }, []*outBuf{o1, o2.asScratch()}
		})
		K, N := hd, nKeys
		qkStride := headOff + K + 4
		add("MatmulQKAcc64/"+nm, func() (func(), []*outBuf) {
			a, bm := rf32(44, M*K), rf32(45, (N-1)*qkStride+headOff+K+qkStride)
			dst, o := mkOut[float32]("dst", M*N)
			return func() { MatmulQKAcc64(a, bm, dst, M, K, N, headOff, qkStride) }, []*outBuf{o}
		})
		add("MatmulQKAcc64Group/"+nm, func() (func(), []*outBuf) {
			a, bm := rf32(46, M*K), rf32(47, (N-1)*qkStride+headOff+K+qkStride)
			dst, o := mkOut[float32]("dst", M*N)
			return func() { MatmulQKAcc64Group(a, bm, dst, M, K, N, headOff, qkStride) }, []*outBuf{o}
		})
	}
	for _, fc := range []struct {
		name         string
		contractExp  bool
		maskFirstRow bool
	}{{"AttendTileFused", false, false}, {"AttendTileFusedContractExp", true, false}, {"AttendTileFused/maskedRow", false, true}} {
		fc := fc
		for _, nKeys := range []int{70, 600} {
			nKeys := nKeys
			add(fmt.Sprintf("%s/nKeys=%d", fc.name, nKeys), func() (func(), []*outBuf) {
				const kt, hd = 4, 64
				qh, kh := rf32(50, kt*hd), rf32(51, nKeys*hd)
				vBlk := rf32(52, hd*nKeys)
				ch, o1 := mkOut[float32]("ch", kt*hd)
				sbl, o2 := mkOut[float32]("sc.SBlk", kt*FusedAttnKeyBlock)
				tmp, o3 := mkOut[float32]("sc.Tmp", kt*hd)
				acc, o4 := mkOut[float32]("sc.Acc", kt*hd)
				mr, o5 := mkOut[float32]("sc.MRun", kt)
				lr, o6 := mkOut[float32]("sc.LRun", kt)
				sc := &FusedAttnScratch{SBlk: sbl, Tmp: tmp, Acc: acc, MRun: mr, LRun: lr}
				lo, hi := make([]int, kt), make([]int, kt)
				for i := range kt {
					hi[i] = nKeys - kt + i
				}
				if fc.maskFirstRow {
					lo[0], hi[0] = 1, 0 // an empty range: its output row must still be written (zero), not left alone
				}
				run := AttendTileFused
				if fc.contractExp {
					run = AttendTileFusedContractExp
				}
				return func() {
					if !run(MatmulBT, qh, kh, vBlk, ch, sc, kt, hd, nKeys, 0.125, lo, hi) {
						panic("declined")
					}
				}, []*outBuf{o1, o2.asScratch(), o3.asScratch(), o4.asScratch(), o5.asScratch(), o6.asScratch()}
			})
		}
	}
	add("GatherVBlockMajor", func() (func(), []*outBuf) {
		const hd, kvh, kvDim, nKeys = 64, 1, 128, 600
		keys, vals := rf32(53, nKeys*kvDim), rf32(54, nKeys*kvDim)
		kh, o1 := mkOut[float32]("kh", nKeys*hd)
		vb, o2 := mkOut[float32]("vBlk", hd*nKeys)
		return func() { GatherVBlockMajor(kh, vb, keys, vals, kvh, hd, kvDim, nKeys) }, []*outBuf{o1, o2}
	})

	// ---- packing, Hamming, and the Dot tiles (they write a sums array)
	for _, dim := range []int{37, 64, 130} {
		dim := dim
		words := (dim + 63) / 64
		add(fmt.Sprintf("PackSignBitsRow/dim=%d", dim), func() (func(), []*outBuf) {
			v := rf32(60, dim)
			dst, o := mkOut[uint64]("dst", words)
			return func() { PackSignBitsRow(dst, v) }, []*outBuf{o}
		})
		add(fmt.Sprintf("PackSignBits/dim=%d", dim), func() (func(), []*outBuf) {
			const n = 5
			src := rf32(61, n*dim)
			dst, o := mkOut[uint64]("dst", n*words)
			return func() { PackSignBits(dst, src, n, dim) }, []*outBuf{o}
		})
		add(fmt.Sprintf("HammingRows/dim=%d", dim), func() (func(), []*outBuf) {
			const n = 7
			q := make([]uint64, words)
			PackSignBitsRow(q, rf32(62, dim))
			codes := make([]uint64, n*words)
			PackSignBits(codes, rf32(63, n*dim), n, dim)
			dst, o := mkOut[uint16]("dst", n)
			return func() { HammingRows(q, codes, words, n, dst) }, []*outBuf{o}
		})
	}
	for _, n4 := range []int{1, 5, 16} {
		n4 := n4
		add(fmt.Sprintf("Dot4x4/n4=%d", n4), func() (func(), []*outBuf) {
			rs := make([][]float32, 5)
			for i := range rs {
				rs[i] = rf32(int64(70+i), n4*4)
			}
			s, o := mkOut[float32]("sums", 16)
			return func() { Dot4x4(&rs[0][0], &rs[1][0], &rs[2][0], &rs[3][0], &rs[4][0], n4, (*[16]float32)(s)) }, []*outBuf{o}
		})
		add(fmt.Sprintf("Dot8x4/n4=%d", n4), func() (func(), []*outBuf) {
			rs := make([][]float32, 9)
			for i := range rs {
				rs[i] = rf32(int64(80+i), n4*4)
			}
			s, o := mkOut[float32]("sums", 32)
			return func() {
				Dot8x4(&rs[0][0], &rs[1][0], &rs[2][0], &rs[3][0], &rs[4][0], &rs[5][0], &rs[6][0], &rs[7][0], &rs[8][0], n4, (*[32]float32)(s))
			}, []*outBuf{o}
		})
		add(fmt.Sprintf("Dot2x8/n4=%d", n4), func() (func(), []*outBuf) {
			rs := make([][]float32, 10)
			for i := range rs {
				rs[i] = rf32(int64(90+i), n4*4)
			}
			s, o := mkOut[float32]("sums", 64)
			return func() {
				Dot2x8(&rs[0][0], &rs[1][0], &rs[2][0], &rs[3][0], &rs[4][0], &rs[5][0], &rs[6][0], &rs[7][0], &rs[8][0], &rs[9][0], n4, (*[64]float32)(s))
			}, []*outBuf{o}
		})
	}

	// ---- the split-half row repack is portable Go (and the arm64 quad repack is built from it).
	for _, K := range []int{32, 96} {
		K := K
		add(fmt.Sprintf("RepackInt4SplitHalfRow/K=%d", K), func() (func(), []*outBuf) {
			src := ri8(int64(K), K/2)
			sb := make([]byte, len(src))
			for i, v := range src {
				sb[i] = byte(v)
			}
			dst, o := mkOut[byte]("dst", K/2)
			return func() { RepackInt4SplitHalfRow(dst, sb, K) }, []*outBuf{o}
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
