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

func TestW4A8F16S0_perGroupVariantMatches(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	w4, s32, s16 := f16S0Fixture(rng, 97, 4864)
	a := make([]float32, 4864)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	want, got := make([]float32, 97), make([]float32, 97)
	var ws Workspace
	MatmulBTW4A8Into(&ws, a, w4, s32, want, 1, 4864, 97, 32)
	for _, pg := range []bool{false, true} {
		SetInt4F16PerGroupS0(pg)
		MatmulBTW4A8F16Into(&ws, a, w4, s16, got, 1, 4864, 97, 32)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("perGroup=%v: dst[%d] %v != %v", pg, i, got[i], want[i])
			}
		}
	}
	SetInt4F16PerGroupS0(false)
}

// BenchmarkW4A8F16S0 times one M=1 row-block at the 0.5B's gate/up and down shapes: f32 scales, f16
// widened per row (default), f16 converted per group. Cache-hot by construction (a diagnostic of the
// kernels' compute cost, not a decode prediction).
func BenchmarkW4A8F16S0(b *testing.B) {
	for _, sh := range []struct {
		name string
		K, N int
	}{{"K896xN4864", 896, 4864}, {"K4864xN896", 4864, 896}, {"K1536xN8960", 1536, 8960}} {
		rng := rand.New(rand.NewSource(3))
		w4, s32, s16 := f16S0Fixture(rng, sh.N, sh.K)
		a := make([]float32, sh.K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		dst := make([]float32, sh.N)
		ws := Workspace{}
		ws.SetThreshold(1 << 62) // serial: kernel cost, not fan-out
		b.Run(sh.name+"/f32", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				MatmulBTW4A8Into(&ws, a, w4, s32, dst, 1, sh.K, sh.N, 32)
			}
		})
		b.Run(sh.name+"/f16-row", func(b *testing.B) {
			SetInt4F16PerGroupS0(false)
			for i := 0; i < b.N; i++ {
				MatmulBTW4A8F16Into(&ws, a, w4, s16, dst, 1, sh.K, sh.N, 32)
			}
		})
		b.Run(sh.name+"/f16-group", func(b *testing.B) {
			SetInt4F16PerGroupS0(true)
			for i := 0; i < b.N; i++ {
				MatmulBTW4A8F16Into(&ws, a, w4, s16, dst, 1, sh.K, sh.N, 32)
			}
			SetInt4F16PerGroupS0(false)
		})
	}
}
