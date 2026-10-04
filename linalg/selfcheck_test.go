package linalg

import (
	"math"
	"strings"
	"testing"
)

// A healthy host passes, in every build (normal, and under each forced-fallback tag), and the check is cheap.
func TestSelfCheck_healthyHostPasses(t *testing.T) {
	rep := SelfCheck(false)
	if !rep.OK() {
		t.Fatalf("a healthy host failed its own kernels' contracts: %v", rep.Mismatches)
	}
	if len(rep.Disabled) != 0 || len(rep.Remaining) != 0 {
		t.Errorf("without repair nothing is disabled and nothing remains: %+v", rep)
	}
	if rep.Elapsed.Milliseconds() > 250 {
		t.Errorf("the self-check took %v; it runs once at startup and is meant to be well under a millisecond (a loose 250 ms guard against it growing)", rep.Elapsed)
	}
	t.Logf("self-check passed in %v", rep.Elapsed)
}

// Each check goes red on the defect it exists for. The kernels under check are swapped through the seams for broken ones.
func TestSelfCheck_detectsBrokenKernels(t *testing.T) {
	restore := func() { scDotI8, scDotW4A8, scQuant = dotI8, dotW4A8, quantizeRowInt8Core }
	t.Cleanup(restore)

	t.Run("dotI8 off by one at a length that straddles the wide loop", func(t *testing.T) {
		defer restore()
		scDotI8 = func(a, b []int8) int32 {
			s := dotI8(a, b)
			if len(a) == 65 {
				s++
			}
			return s
		}
		rep := SelfCheck(false)
		if len(rep.Mismatches) != 1 || rep.Mismatches[0].Kernel != "dotI8" || !strings.Contains(rep.Mismatches[0].Detail, "n=65") {
			t.Fatalf("want exactly a dotI8 mismatch naming n=65, got %v", rep.Mismatches)
		}
	})
	t.Run("dotW4A8 off by 1e-3 relative", func(t *testing.T) {
		defer restore()
		scDotW4A8 = func(act []int8, packed []byte, scales []float32, group, K int) float32 {
			return dotW4A8(act, packed, scales, group, K) * 1.001
		}
		rep := SelfCheck(false)
		found := false
		for _, m := range rep.Mismatches {
			found = found || m.Kernel == "dotW4A8"
		}
		if !found {
			t.Fatalf("a 1e-3 relative error passed the 1e-5 bound: %v", rep.Mismatches)
		}
	})
	t.Run("quantizer returns a different scale", func(t *testing.T) {
		defer restore()
		scQuant = func(row []float32, q []int8, zeroScale float32) float32 {
			return quantizeRowInt8Core(row, q, zeroScale) * (1 + 1e-6)
		}
		rep := SelfCheck(false)
		if len(rep.Mismatches) != 1 || rep.Mismatches[0].Kernel != "quantizeRowInt8" {
			t.Fatalf("want a quantizeRowInt8 mismatch, got %v", rep.Mismatches)
		}
	})
	t.Run("NaN is a mismatch, not a pass", func(t *testing.T) {
		defer restore()
		scDotW4A8 = func(act []int8, packed []byte, scales []float32, group, K int) float32 { return float32(math.NaN()) }
		if rep := SelfCheck(false); rep.OK() {
			t.Fatal("a kernel returning NaN passed")
		}
	})
}

// The real defect of 2026-09-24, modelled: the pre-fix AVX-512 VNNI kernel folded sum(nib*act) and -8*sum(act) into two f32 accumulators and combined them at the end,
// so each carried the UNCENTERED magnitude and their rounding did not cancel (measured up to 3.2e-3 per logit through goinfer's int4 forward). The self-check must
// decline that arithmetic, and must accept the fixed one (the centered int32 partial, then one f32 multiply per group).
func TestSelfCheck_declinesTheUncenteredFoldOf20260924(t *testing.T) {
	t.Cleanup(func() { scDotW4A8 = dotW4A8 })
	uncentered := func(act []int8, packed []byte, scales []float32, group, K int) float32 {
		var accNibAct, accCorr float32 // the two f32 accumulators of the pre-fix kernel
		for g := 0; g < (K+group-1)/group; g++ {
			var raw, sum int32
			for k := g * group; k < min((g+1)*group, K); k++ {
				nib := packed[k>>1] & 0x0F
				if k&1 == 1 {
					nib = packed[k>>1] >> 4
				}
				raw += int32(act[k]) * int32(nib)
				sum += int32(act[k])
			}
			accNibAct += float32(raw) * scales[g]
			accCorr += float32(-8*sum) * scales[g]
		}
		return accNibAct + accCorr
	}
	centered := func(act []int8, packed []byte, scales []float32, group, K int) float32 {
		return dotW4A8Scalar(act, packed, scales, group, K)
	}
	scDotW4A8 = uncentered
	bad := checkDotW4A8Centering()
	if bad == nil {
		t.Fatal("the pre-fix uncentered fold passed the centering check")
	}
	if bad.Observed < 5 {
		t.Errorf("the uncentered fold should miss the bound by an order of magnitude; it missed by only %.2gx", bad.Observed)
	}
	t.Logf("uncentered fold: %s", bad)
	scDotW4A8 = centered
	if bad := checkDotW4A8Centering(); bad != nil {
		t.Fatalf("the centered fold was declined: %s", bad)
	}
}

func TestActiveKernels_isConsistent(t *testing.T) {
	k := ActiveKernels()
	if k.Arch == "" {
		t.Fatal("no arch")
	}
	for _, a := range k.Active {
		if len(k.Forced) == 0 && !contains(k.Detected, a) && a != "avx512vnni+vl" {
			t.Errorf("active tier %q is not detected on this CPU and nothing forced it", a)
		}
	}
	if len(k.Forced) > 0 && len(k.Active) > len(k.Detected) {
		t.Errorf("a forced build cannot activate more than the CPU has: detected %v active %v", k.Detected, k.Active)
	}
	t.Logf("arch %s detected %v active %v forced %v", k.Arch, k.Detected, k.Active, k.Forced)
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}
