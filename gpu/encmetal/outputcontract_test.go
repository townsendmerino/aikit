//go:build darwin

package encmetal

import (
	"math"
	"math/rand"
	"testing"
)

// TestEncMetal_outputContract proves the documented output contract of MatmulBT and MatmulBTQ8 — "overwrites
// dst[:M*N]; do not pre-zero" — on the DEVICE path (goinfer audit R-11; aikit v1.56.1 checked only the doc line's text).
// dst is filled with NaN and carries a guard tail past M*N. After the call every element of dst[:M*N] must be finite,
// the guard must be untouched bit for bit, and the result must be byte-identical to the same call on a dst pre-filled
// with 7.5, so no element depends on what dst held. Shapes: one aligned (gemm_f32_sg_big) and one unaligned (the
// general gemm_f32_sg, whose partial tiles are where an unwritten edge would hide), both well over minGPUFlops.
//
// The device path is asserted, not inferred from the shape: the backend's device output buffer must have grown to M*N
// (only the device path touches it), and the result must differ from the pure-Go CPU backend's in at least one bit (the
// CPU fallback zeroes and overwrites too, so a silent fallback would otherwise pass). MatmulBTQ8 must also return true,
// and both are checked against the float64 references within the parity tests' 2e-4.
func TestEncMetal_outputContract(t *testing.T) {
	const guard = 64
	guardBits := math.Float32bits(-12345.5)
	shapes := []struct {
		name    string
		M, K, N int
	}{
		{"aligned/sg_big", 256, 768, 768},
		{"unaligned/sg", 300, 770, 650},
	}
	fill := func(dst []float32, n int, v float32) {
		for i := range dst {
			if i < n {
				dst[i] = v
			} else {
				dst[i] = math.Float32frombits(guardBits)
			}
		}
	}
	check := func(t *testing.T, what string, dst []float32, n int) {
		t.Helper()
		for i := 0; i < n; i++ {
			if v := float64(dst[i]); math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("%s: dst[%d] = %v after the call: not overwritten (NaN pre-fill survived)", what, i, dst[i])
			}
		}
		for i := n; i < len(dst); i++ {
			if math.Float32bits(dst[i]) != guardBits {
				t.Fatalf("%s: guard element %d (dst[%d]) written: %v", what, i-n, i, dst[i])
			}
		}
	}
	same := func(t *testing.T, what string, x, y []float32, n int) {
		t.Helper()
		for i := 0; i < n; i++ {
			if math.Float32bits(x[i]) != math.Float32bits(y[i]) {
				t.Fatalf("%s: dst[%d] depends on the pre-fill: %v after NaN, %v after 7.5", what, i, x[i], y[i])
			}
		}
	}
	within := func(t *testing.T, what string, dst []float32, ref []float64) {
		t.Helper()
		worst := 0.0
		for i := range ref {
			den := math.Max(math.Abs(ref[i]), 1)
			worst = math.Max(worst, math.Abs(float64(dst[i])-ref[i])/den)
		}
		if worst > 2e-4 {
			t.Fatalf("%s: worst relative Δ %.3g against the float64 reference", what, worst)
		}
	}
	for _, s := range shapes {
		n := s.M * s.N
		if 2*int64(s.M)*int64(s.K)*int64(s.N) < 2*minGPUFlops {
			t.Fatalf("%s: %d FLOP is not comfortably over minGPUFlops (%d)", s.name, 2*s.M*s.K*s.N, minGPUFlops)
		}
		rng := rand.New(rand.NewSource(int64(n)))
		a, w := randF(rng, s.M*s.K), randF(rng, s.N*s.K)

		t.Run("MatmulBT/"+s.name, func(t *testing.T) {
			b := mustBackend(t)
			defer b.Close()
			nanDst, sevenDst := make([]float32, n+guard), make([]float32, n+guard)
			fill(nanDst, n, float32(math.NaN()))
			b.MatmulBT(a, w, nanDst, s.M, s.K, s.N)
			if b.cCap < n {
				t.Fatalf("the device output buffer holds %d floats, want >= %d: the call did not take the device path", b.cCap, n)
			}
			check(t, "NaN pre-fill", nanDst, n)
			fill(sevenDst, n, 7.5)
			b.MatmulBT(a, w, sevenDst, s.M, s.K, s.N)
			check(t, "7.5 pre-fill", sevenDst, n)
			same(t, "MatmulBT", nanDst, sevenDst, n)
			cpu := make([]float32, n)
			b.cpu.MatmulBT(a, w, cpu, s.M, s.K, s.N)
			differ := 0
			for i := 0; i < n; i++ {
				if math.Float32bits(cpu[i]) != math.Float32bits(nanDst[i]) {
					differ++
				}
			}
			if differ == 0 {
				t.Fatalf("result bit-identical to the CPU backend in all %d elements: the device path did not run", n)
			}
			within(t, "MatmulBT", nanDst, refMatmulBT(a, w, s.M, s.K, s.N))
			t.Logf("M=%d K=%d N=%d: device path (output buffer %d floats, %d of %d elements differ from the CPU backend), contract holds", s.M, s.K, s.N, b.cCap, differ, n)
		})

		t.Run("MatmulBTQ8/"+s.name, func(t *testing.T) {
			b := mustBackend(t)
			defer b.Close()
			wq := randQ8(rng, s.N*s.K)
			ws := make([]float32, s.N)
			for i := range ws {
				ws[i] = 0.002 + float32(i%13)*0.0001
			}
			nanDst, sevenDst := make([]float32, n+guard), make([]float32, n+guard)
			fill(nanDst, n, float32(math.NaN()))
			if !b.MatmulBTQ8(nanDst, a, wq, ws, s.M, s.K, s.N) {
				t.Fatalf("MatmulBTQ8 returned false at %d FLOP: the device path declined", 2*s.M*s.K*s.N)
			}
			if b.cCap < n {
				t.Fatalf("the device output buffer holds %d floats, want >= %d", b.cCap, n)
			}
			check(t, "NaN pre-fill", nanDst, n)
			fill(sevenDst, n, 7.5)
			if !b.MatmulBTQ8(sevenDst, a, wq, ws, s.M, s.K, s.N) {
				t.Fatal("MatmulBTQ8 returned false on the second call")
			}
			check(t, "7.5 pre-fill", sevenDst, n)
			same(t, "MatmulBTQ8", nanDst, sevenDst, n)
			within(t, "MatmulBTQ8", nanDst, cpuQ8Ref(a, wq, ws, s.M, s.K, s.N))
			t.Logf("M=%d K=%d N=%d: device path returned true, contract holds", s.M, s.K, s.N)
		})
	}
}
