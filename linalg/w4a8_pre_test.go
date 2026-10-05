package linalg

import (
	"math"
	"math/rand/v2"
	"strconv"
	"testing"
)

// TestW4A8Pre_bitIdenticalToQuantizingEntry is R-13's gate (goinfer docs/tasks/task-recompute-audit.md): an activation
// quantized once (QuantizeActW4A8 / QuantizeActQ) and fed to the Pre entries gives exactly the bits the quantizing
// entries give, for the canonical layout and the arch's repacked one (arm64 row4, amd64 split-half; a no-op
// elsewhere), at M = 1, 2, 3 and 5, with one scale per row (group 0), per 32 (the SIMD grouped kernels) and per 64
// (the grouped reference). One block also feeds two weights of the same K (q/k/v style) and each matches its own
// quantizing call.
func TestW4A8Pre_bitIdenticalToQuantizingEntry(t *testing.T) {
	const wGroup = 32
	rng := rand.New(rand.NewPCG(0x413, 0x1313))
	rnd := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	shapes := []struct{ K, N int }{{64, 8}, {96, 20}, {256, 64}}
	for _, sh := range shapes {
		for _, repack := range []bool{false, true} {
			mk := func() WeightMat {
				q4, q4s := QuantizeGroupsInt4(rnd(sh.N*sh.K), sh.N, sh.K, wGroup)
				w := WrapInt4(q4, q4s, sh.N, sh.K, wGroup)
				if repack {
					w.RepackInt4Row4()
					w.RepackInt4SplitHalf()
				}
				return w
			}
			w1, w2 := mk(), mk()
			for _, g := range []int{0, 32, 64} {
				for _, M := range []int{1, 2, 3, 5} {
					name := "K" + strconv.Itoa(sh.K) + "N" + strconv.Itoa(sh.N) + "/repack=" + strconv.FormatBool(repack) +
						"/group" + strconv.Itoa(g) + "/M" + strconv.Itoa(M)
					a := rnd(M * sh.K)
					var wsRef, wsPre Workspace
					wsRef.SetActQuantGroup(g)
					wsPre.SetActQuantGroup(g)
					var q ActQ
					w1.QuantizeActW4A8(&wsPre, a, M, &q)
					for wi, w := range []*WeightMat{&w1, &w2} {
						want := make([]float32, M*sh.N)
						got := make([]float32, M*sh.N)
						for i := range got {
							got[i] = float32(math.NaN()) // overwrite contract
						}
						w.MatmulBTW4A8Into(&wsRef, a, want, M)
						w.MatmulBTW4A8PreInto(&wsPre, &q, got, M)
						for i := range want {
							if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
								t.Fatalf("%s weight %d: out[%d] Pre %v, quantizing %v", name, wi, i, got[i], want[i])
							}
						}
					}
					// The free function on the canonical bytes, with the workspace's group.
					if !repack {
						w4, s16, grp, ok := w1.Int4F16()
						if !ok {
							t.Fatal("Int4F16: not an int4 weight")
						}
						var qf ActQ
						QuantizeActQ(a, M, sh.K, actGroupFor(&wsPre), &qf)
						want := make([]float32, M*sh.N)
						got := make([]float32, M*sh.N)
						MatmulBTW4A8F16Into(&wsRef, a, w4, s16, want, M, sh.K, sh.N, grp)
						MatmulBTW4A8F16Pre(&wsPre, &qf, w4, s16, got, M, sh.K, sh.N, grp)
						for i := range want {
							if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
								t.Fatalf("%s free function: out[%d] Pre %v, quantizing %v", name, i, got[i], want[i])
							}
						}
					}
				}
			}
		}
	}
}

// TestW4A8Pre_refusesAMismatchedBlock: a block quantized for another K, M or activation group is refused, not run
// (it would silently give another result).
func TestW4A8Pre_refusesAMismatchedBlock(t *testing.T) {
	q4, q4s := QuantizeGroupsInt4(make([]float32, 8*64), 8, 64, 32)
	w := WrapInt4(q4, q4s, 8, 64, 32)
	var ws Workspace
	ws.SetActQuantGroup(32)
	a := make([]float32, 2*64)
	for _, c := range []struct {
		name string
		q    func() ActQ
		M    int
	}{
		{"other group", func() ActQ { var q ActQ; QuantizeActQ(a, 2, 64, 0, &q); return q }, 2},
		{"other M", func() ActQ { var q ActQ; QuantizeActQ(a, 1, 64, 32, &q); return q }, 2},
		{"other K", func() ActQ { var q ActQ; QuantizeActQ(a, 2, 32, 32, &q); return q }, 2},
	} {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("a mismatched block was accepted")
				}
			}()
			q := c.q()
			w.MatmulBTW4A8PreInto(&ws, &q, make([]float32, c.M*8), c.M)
		})
	}
}

// TestW4A8Batch_groupedQuantizesOnceBitIdentical is R-14's gate: the grouped MatmulBTW4A8Batch quantizes its
// activation once for every op (Workspace.batchQ) and each op's output is the bits its own MatmulBTW4A8F16Into gives,
// over three canonical ops of different N, groups 32 (the SIMD kernels) and 64 (the reference), M = 1 and 3.
func TestW4A8Batch_groupedQuantizesOnceBitIdentical(t *testing.T) {
	const K, wGroup = 128, 32
	rng := rand.New(rand.NewPCG(0x14, 0x1414))
	rnd := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	type opw struct {
		w4  []byte
		s16 []uint16
		N   int
	}
	var ws3 []opw
	for _, N := range []int{8, 24, 40} {
		q4, q4s := QuantizeGroupsInt4(rnd(N*K), N, K, wGroup)
		ws3 = append(ws3, opw{q4, F32ToF16Scales(q4s), N})
	}
	for _, g := range []int{32, 64} {
		for _, M := range []int{1, 3} {
			a := rnd(M * K)
			var wsB, wsR Workspace
			wsB.SetActQuantGroup(g)
			wsR.SetActQuantGroup(g)
			ops := make([]W4A8Op, len(ws3))
			for i, o := range ws3 {
				ops[i] = W4A8Op{W4: o.w4, ScalesF16: o.s16, Dst: make([]float32, M*o.N), N: o.N}
			}
			MatmulBTW4A8Batch(&wsB, a, M, K, wGroup, ops)
			for i, o := range ws3 {
				want := make([]float32, M*o.N)
				MatmulBTW4A8F16Into(&wsR, a, o.w4, o.s16, want, M, K, o.N, wGroup)
				for j := range want {
					if math.Float32bits(want[j]) != math.Float32bits(ops[i].Dst[j]) {
						t.Fatalf("group %d M %d op %d: out[%d] batch %v, per-op %v", g, M, i, j, ops[i].Dst[j], want[j])
					}
				}
			}
		}
	}
}
