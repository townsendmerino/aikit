package linalg

import (
	"math/rand"
	"testing"
)

// f16Scales returns N*nGroups positive scales that are exactly representable in half precision, their
// f16 encodings, and nibble-packed int4 weights.
func f16S0Fixture(rng *rand.Rand, N, K int) (w4 []byte, s32 []float32, s16 []uint16) {
	nGroups := K / 32
	w4 = make([]byte, N*K/2)
	rng.Read(w4)
	s32 = make([]float32, N*nGroups)
	s16 = make([]uint16, N*nGroups)
	for i := range s32 {
		v := f16ToF32(f32ToF16ExactRound(float32(rng.Float64()*0.05 + 1e-4)))
		s32[i] = v
		s16[i] = f32ToF16Exact(v)
	}
	return
}

// f32ToF16ExactRound truncates to f16 (test-only): produces a representable value to feed the exact encoder.
func f32ToF16ExactRound(s float32) uint16 {
	b := mathFloat32bits(s) &^ ((1 << 13) - 1)
	return f32ToF16Exact(mathFloat32frombits(b))
}

func TestW4A8F16S0_matchesF32(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, sh := range []struct{ M, K, N int }{{1, 1536, 256}, {1, 8960, 97}, {2, 896, 130}, {3, 3584, 64}} {
		w4, s32, s16 := f16S0Fixture(rng, sh.N, sh.K)
		a := make([]float32, sh.M*sh.K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		want := make([]float32, sh.M*sh.N)
		got := make([]float32, sh.M*sh.N)
		var ws1, ws2 Workspace
		MatmulBTW4A8Into(&ws1, a, w4, s32, want, sh.M, sh.K, sh.N, 32)
		MatmulBTW4A8F16Into(&ws2, a, w4, s16, got, sh.M, sh.K, sh.N, 32)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("M=%d K=%d N=%d: dst[%d] f16 %v != f32 %v", sh.M, sh.K, sh.N, i, got[i], want[i])
			}
		}
		// The batch op and the WeightMat entry point, switched on.
		SetInt4F16ScalesS0(true)
		gotB := make([]float32, sh.M*sh.N)
		MatmulBTW4A8Batch(&ws2, a, sh.M, sh.K, 32, []W4A8Op{{W4: w4, Scales: s32, ScalesF16: s16, Dst: gotB, N: sh.N}})
		wm := WrapInt4(w4, s32, sh.N, sh.K, 32)
		if !wm.AttachInt4ScalesF16S0() {
			t.Fatal("attach failed")
		}
		gotW := make([]float32, sh.M*sh.N)
		wm.MatmulBTW4A8Into(&ws2, a, gotW, sh.M)
		SetInt4F16ScalesS0(false)
		for i := range want {
			if gotB[i] != want[i] || gotW[i] != want[i] {
				t.Fatalf("M=%d K=%d N=%d: dst[%d] batch %v / weightmat %v != f32 %v", sh.M, sh.K, sh.N, i, gotB[i], gotW[i], want[i])
			}
		}
		// Mutation: one scale off by one f16 ulp must change the output.
		s16[0]++
		mut := make([]float32, sh.M*sh.N)
		MatmulBTW4A8F16Into(&ws2, a, w4, s16, mut, sh.M, sh.K, sh.N, 32)
		s16[0]--
		same := true
		for i := range want {
			if mut[i] != want[i] {
				same = false
			}
		}
		if same {
			t.Fatalf("M=%d K=%d N=%d: a changed f16 scale left the output unchanged — the f16 path is not reading its scales", sh.M, sh.K, sh.N)
		}
	}
	t.Logf("hasW4A8F16 kernel path: %v", f16KernelActive())
}
