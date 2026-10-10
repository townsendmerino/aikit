package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

func half16TestWeights(rng *rand.Rand, n int, bf16 bool) (bits []uint16, widened []float32) {
	bits = make([]uint16, n)
	widened = make([]float32, n)
	for i := range bits {
		f := (rng.Float32()*2 - 1) * 3
		if bf16 {
			bits[i] = F32ToBF16(f)
			widened[i] = BF16ToF32(bits[i])
		} else {
			bits[i] = F32ToF16(f)
			widened[i] = F16ToF32(bits[i])
		}
	}
	return
}

// The 16-bit kinds must equal MatmulBT over the widened weights bit for bit, at every M, over shapes that cross the tile (64), leave a
// ragged 8-column group (N % 8 != 0), put a tile edge inside a worker shard, and reach the parallel path (the work crosses parThreshold).
func TestWeightMat_half16_bitIdenticalToF32(t *testing.T) {
	half16BitIdentical(t)
}

// The same, with the fan-out forced on at several widths (the shapes above are all under the default threshold, so they ran serially).
func TestWeightMat_half16_bitIdenticalToF32_parallel(t *testing.T) {
	oldThr, oldW := ParallelThreshold(), ParallelWidth()
	defer func() { SetParallelThreshold(oldThr); SetParallelWidth(oldW) }()
	SetParallelThreshold(1)
	for _, w := range []int{2, 3, 7} {
		SetParallelWidth(w)
		half16BitIdentical(t)
	}
}

func half16BitIdentical(t *testing.T) {
	t.Helper()
	rng := rand.New(rand.NewPCG(5, 11))
	shapes := [][2]int{{1, 8}, {7, 33}, {8, 64}, {63, 100}, {64, 768}, {65, 768}, {130, 96}, {257, 300}, {512, 1536}, {1030, 256}}
	for _, bf := range []bool{true, false} {
		for _, sh := range shapes {
			N, K := sh[0], sh[1]
			bits, wf := half16TestWeights(rng, N*K, bf)
			var wm WeightMat
			if bf {
				wm = WrapBF16(bits, N, K)
			} else {
				wm = WrapF16(bits, N, K)
			}
			for _, M := range []int{1, 2, 5, 17} {
				a := make([]float32, M*K)
				for i := range a {
					a[i] = rng.Float32()*2 - 1
				}
				want := make([]float32, M*N)
				MatmulBT(a, wf, want, M, K, N)
				for name, run := range map[string]func(dst []float32){
					"MatmulBT":     func(dst []float32) { wm.MatmulBT(a, dst, M) },
					"MatmulBTInto": func(dst []float32) { var ws Workspace; wm.MatmulBTInto(&ws, a, dst, M) },
				} {
					got := make([]float32, M*N)
					for i := range got {
						got[i] = float32(math.NaN()) // the contract is overwrite, not accumulate
					}
					run(got)
					for i := range want {
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							t.Fatalf("bf16=%v N=%d K=%d M=%d %s: element %d got %v want %v", bf, N, K, M, name, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestWeightMat_half16_rowKindAccessors(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 3))
	for _, bf := range []bool{true, false} {
		const N, K = 9, 40
		bits, wf := half16TestWeights(rng, N*K, bf)
		wm := WrapF16(bits, N, K)
		wantKind := "f16"
		if bf {
			wm, wantKind = WrapBF16(bits, N, K), "bf16"
		}
		if wm.Kind() != wantKind || wm.Rows() != N || wm.Cols() != K {
			t.Fatalf("Kind/Rows/Cols = %q %d %d", wm.Kind(), wm.Rows(), wm.Cols())
		}
		if _, ok := wm.F32(); ok {
			t.Fatal("F32() must report !ok for a 16-bit kind")
		}
		if u, gotBF, ok := wm.Half16(); !ok || gotBF != bf || &u[0] != &bits[0] {
			t.Fatalf("Half16 = ok %v bf16 %v, and it must alias the wrapped slice", ok, gotBF)
		}
		row := make([]float32, K)
		for i := range N {
			wm.Row(i, row)
			for k := range K {
				if row[k] != wf[i*K+k] {
					t.Fatalf("bf16=%v Row(%d)[%d] = %v want %v", bf, i, k, row[k], wf[i*K+k])
				}
			}
		}
	}
}

// Rounding a bf16-sourced weight to bf16 loses nothing; rounding an arbitrary f32 is nearest-even; NaN stays NaN; the wrappers refuse a wrong length.
func TestQuantizeBF16_roundTripAndRounding(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 8))
	_, src := half16TestWeights(rng, 4*16, true)
	wm := QuantizeBF16(src, 4, 16)
	row := make([]float32, 16)
	for i := range 4 {
		wm.Row(i, row)
		for k := range 16 {
			if row[k] != src[i*16+k] {
				t.Fatalf("a bf16-sourced weight changed on the round trip: row %d col %d %v -> %v", i, k, src[i*16+k], row[k])
			}
		}
	}
	// 1 + 2^-8 is exactly halfway between bf16 1.0 and 1+2^-7: ties go to even (1.0); one ulp above goes up.
	if got := BF16ToF32(F32ToBF16(math.Float32frombits(0x3f808000))); got != 1 {
		t.Fatalf("tie rounded to %v, want 1 (even)", got)
	}
	if got := BF16ToF32(F32ToBF16(math.Float32frombits(0x3f808001))); got != math.Float32frombits(0x3f810000) {
		t.Fatalf("above the tie rounded to %v, want 1+2^-7", got)
	}
	if got := BF16ToF32(F32ToBF16(float32(math.NaN()))); !math.IsNaN(float64(got)) {
		t.Fatalf("NaN became %v", got)
	}
	if got := BF16ToF32(F32ToBF16(float32(math.Inf(-1)))); !math.IsInf(float64(got), -1) {
		t.Fatalf("-Inf became %v", got)
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("WrapBF16 accepted a short slice")
			}
		}()
		WrapBF16(make([]uint16, 5), 2, 3)
	}()
}

func BenchmarkWeightMat_half16_decode(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 1))
	const N, K = 4096, 2048
	bits, wf := half16TestWeights(rng, N*K, true)
	a := make([]float32, K)
	for i := range a {
		a[i] = rng.Float32()
	}
	dst := make([]float32, N)
	f32 := WrapF32(wf, N, K)
	bf := WrapBF16(bits, N, K)
	b.Run("f32", func(b *testing.B) {
		for range b.N {
			f32.MatmulBT(a, dst, 1)
		}
	})
	b.Run("bf16", func(b *testing.B) {
		for range b.N {
			bf.MatmulBT(a, dst, 1)
		}
	})
}
