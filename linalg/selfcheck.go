package linalg

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"time"
)

// Runtime self-check of the CPU kernels the dispatcher selected (goinfer's hardware-coverage H2, CPU-ISA part).
//
// A dispatcher here picks its kernel from a flag detected once at init, so a machine nobody here owns can select a kernel that has never run, and a wrong kernel
// does not crash: it returns slightly wrong numbers (the 2026-09-24 AVX-512 VNNI accumulator bug moved a logit by 3.2e-3 and was found only because one CI runner
// happened to have the CPU). SelfCheck runs the DISPATCHED kernel against the portable scalar reference on a small fixed, seeded input and holds it to the SAME
// agreement aikit's own tests assert for that kernel (exact where they assert bit-identity, the same relative bound where they allow one), so it accepts exactly what
// the suite certified and rejects what the suite would reject: it invents no tolerance. It costs well under a millisecond.
//
// With repair, a mismatch turns the highest active ISA tier off (AVX-512 VNNI, then AVX2 on amd64; DotProd on arm64) and checks again, so a bad tier degrades to the
// next one instead of producing wrong numbers. The flags are plain package variables read without synchronisation by every kernel, so call SelfCheck with repair
// BEFORE concurrent use of the package (goinfer calls it once, before the first model load); without repair it only reads.

// Mismatch is one kernel that disagreed with its reference.
type Mismatch struct {
	Kernel   string
	Observed float64 // what was measured: a count of differing cases (exact checks), a relative error, or the ratio of the error to its bound
	Allowed  float64 // the bar the kernel's own tests hold it to
	Detail   string
}

func (m Mismatch) String() string {
	return fmt.Sprintf("%s: observed %.3g, allowed %.3g (%s)", m.Kernel, m.Observed, m.Allowed, m.Detail)
}

// SelfCheckReport is SelfCheck's result.
type SelfCheckReport struct {
	// Mismatches is what the FIRST pass found: the dispatcher as it was handed to the caller.
	Mismatches []Mismatch
	// Disabled names each tier repair turned off, in order ("avx512vnni", "avx2", "dotprod").
	Disabled []string
	// Remaining is what the LAST pass found; empty means the process ends on kernels that agree with their references.
	Remaining []Mismatch
	Elapsed   time.Duration
}

// OK reports that the first pass found nothing.
func (r SelfCheckReport) OK() bool { return len(r.Mismatches) == 0 }

// Seams for the tests: the kernels under check. Production leaves them at the dispatched functions.
var (
	scDotI8   = dotI8
	scDotW4A8 = dotW4A8
	scQuant   = quantizeRowInt8Core
)

// SelfCheck runs the checks once, or until repair has no tier left to turn off. See the package comment above for what it compares.
func SelfCheck(repair bool) SelfCheckReport {
	start := time.Now()
	var rep SelfCheckReport
	for pass := 0; pass < 4; pass++ {
		ms := runSelfChecks()
		if pass == 0 {
			rep.Mismatches = ms
		}
		rep.Remaining = ms
		if len(ms) == 0 || !repair {
			break
		}
		name := disableTopTier()
		if name == "" {
			break
		}
		rep.Disabled = append(rep.Disabled, name)
	}
	rep.Elapsed = time.Since(start)
	return rep
}

func runSelfChecks() []Mismatch {
	var out []Mismatch
	for _, c := range []func() *Mismatch{checkDotI8, checkDotW4A8, checkDotW4A8Centering, checkQuantizeRowInt8} {
		if m := c(); m != nil {
			out = append(out, *m)
		}
	}
	return out
}

// checkDotI8 is TestDotI8_matchesScalar: the int32 dot is integer arithmetic, so the dispatched kernel must equal the scalar exactly, at the lengths that straddle
// the wide loops and their remainders, and at the extremes.
func checkDotI8() *Mismatch {
	rng := rand.New(rand.NewSource(8))
	bad, first := 0, ""
	test := func(a, b []int8) {
		if got, want := scDotI8(a, b), dotI8Scalar(a, b); got != want {
			if bad == 0 {
				first = fmt.Sprintf("n=%d: dispatched %d, scalar %d", len(a), got, want)
			}
			bad++
		}
	}
	for _, n := range []int{0, 1, 7, 15, 16, 17, 31, 47, 63, 64, 65, 80, 127, 128, 129, 191, 192, 2048, 2049} {
		a, b := make([]int8, n), make([]int8, n)
		for i := range a {
			a[i], b[i] = int8(rng.Intn(255)-127), int8(rng.Intn(255)-127)
		}
		test(a, b)
	}
	for _, n := range []int{16, 2048} {
		a, b := make([]int8, n), make([]int8, n)
		for i := range a {
			a[i], b[i] = -127, 127
		}
		test(a, b)
	}
	if bad > 0 {
		return &Mismatch{Kernel: "dotI8", Observed: float64(bad), Allowed: 0, Detail: first}
	}
	return nil
}

// checkDotW4A8 is TestAVX512VNNI_dotW4A8_dispatchesThroughEveryTier: the dispatched W4A8 dot against the scalar to a relative 1e-5, across the K that reach each tier.
func checkDotW4A8() *Mismatch {
	const group = 32
	rng := rand.New(rand.NewSource(103))
	worst, detail := 0.0, ""
	for _, K := range []int{32, 64, 96, 300, 768, 2048, 3072, 3100} {
		nGroups := (K + group - 1) / group
		act := make([]int8, K)
		for i := range act {
			act[i] = int8(rng.Intn(256) - 128)
		}
		packed := make([]byte, (K+1)/2)
		for i := range packed {
			packed[i] = byte(rng.Intn(256))
		}
		scales := make([]float32, nGroups)
		for i := range scales {
			scales[i] = float32(rng.NormFloat64())
		}
		got, want := scDotW4A8(act, packed, scales, group, K), dotW4A8Scalar(act, packed, scales, group, K)
		if rel := math.Abs(float64(got-want)) / (math.Abs(float64(want)) + 1e-9); rel > worst || math.IsNaN(rel) {
			worst, detail = rel, fmt.Sprintf("K=%d: dispatched %v, scalar %v", K, got, want)
			if math.IsNaN(rel) {
				worst = math.Inf(1)
			}
		}
	}
	if worst > 1e-5 {
		return &Mismatch{Kernel: "dotW4A8", Observed: worst, Allowed: 1e-5, Detail: detail}
	}
	return nil
}

// checkDotW4A8Centering is checkDotW4A8FoldCentering from the AVX-512 VNNI test, the check that pins the 2026-09-24 fix: the activation-sum correction is applied to
// the exact int32 partials BEFORE the f32 fold. Weights almost all 0 (nibble 8, a sparse +-1) against large same-sign activations make the uncentered terms ~100x the
// centered result, so a kernel whose error scales with the uncentered magnitude misses the textbook recursive-summation bound by an order of magnitude. The reference
// is the exact float64 of the exact integer partials.
func checkDotW4A8Centering() *Mismatch {
	rng := rand.New(rand.NewSource(211))
	worst, detail := 0.0, ""
	for _, nGroups := range []int{1, 2, 7, 24, 96, 256} {
		for range 4 {
			K := nGroups * 32
			act := make([]int8, K)
			nib := make([]byte, K)
			for i := range act {
				act[i] = int8(100 + rng.Intn(28))
				nib[i] = 8
				if rng.Intn(16) == 0 {
					nib[i] = byte(7 + 2*rng.Intn(2))
				}
			}
			packed := make([]byte, K/2)
			for i := 0; i < K; i += 2 {
				packed[i/2] = (nib[i] & 0x0F) | ((nib[i+1] & 0x0F) << 4)
			}
			scales := make([]float32, nGroups)
			for g := range scales {
				scales[g] = float32(rng.Float64()*0.05 + 0.0001)
			}
			var exact, mag float64
			for g, s := range scales {
				var c, m int64
				for k := g * 32; k < g*32+32; k++ {
					v := int64(int(nib[k])-8) * int64(act[k])
					c += v
					if v < 0 {
						v = -v
					}
					m += v
				}
				exact += float64(s) * float64(c)
				mag += math.Abs(float64(s)) * float64(m)
			}
			tol := float64(nGroups+8)*mag/(1<<24) + 1e-30
			got := scDotW4A8(act, packed, scales, 32, K)
			d := math.Abs(float64(got) - exact)
			ratio := d / tol
			if math.IsNaN(d) {
				ratio = math.Inf(1)
			}
			if ratio > worst {
				worst, detail = ratio, fmt.Sprintf("nGroups=%d: dispatched %.9g, exact %.9g, |diff| %.3g against a bound of %.3g", nGroups, got, exact, d, tol)
			}
		}
	}
	if worst > 1 {
		return &Mismatch{Kernel: "dotW4A8 (centering)", Observed: worst, Allowed: 1,
			Detail: detail + ": the error scales with the uncentered magnitude, so the activation-sum correction is not applied in int32 before the f32 fold"}
	}
	return nil
}

// checkQuantizeRowInt8 is TestQuantizeRowInt8_bitIdenticalToScalar: the activation quantizer, dispatched, equals the scalar bit for bit (scale bits and every code),
// for rows of the lengths that straddle its vector widths, an all-zero row, and a wide dynamic range.
func checkQuantizeRowInt8() *Mismatch {
	rng := rand.New(rand.NewSource(0x5103))
	bad, first := 0, ""
	row := func(n int, f func(i int) float32) {
		r := make([]float32, n)
		for i := range r {
			r[i] = f(i)
		}
		for _, zeroScale := range []float32{0, 1} {
			want, got := make([]int8, n), make([]int8, n)
			for i := range got {
				got[i] = 0x55
			}
			ws, gs := quantizeRowInt8CoreScalar(r, want, zeroScale), scQuant(r, got, zeroScale)
			ok := math.Float32bits(ws) == math.Float32bits(gs)
			for i := range want {
				ok = ok && want[i] == got[i]
			}
			if !ok {
				if bad == 0 {
					first = fmt.Sprintf("n=%d zeroScale=%v: scalar scale %v, dispatched %v", n, zeroScale, ws, gs)
				}
				bad++
			}
		}
	}
	for _, n := range []int{1, 7, 8, 15, 16, 17, 31, 32, 33, 64, 127, 300, 1536} {
		row(n, func(int) float32 { return float32(rng.NormFloat64()) })
		row(n, func(int) float32 { return 0 })
		row(n, func(i int) float32 { return float32(rng.NormFloat64()) * float32(math.Pow(10, float64(i%7-3))) })
	}
	if bad > 0 {
		return &Mismatch{Kernel: "quantizeRowInt8", Observed: float64(bad), Allowed: 0, Detail: first}
	}
	return nil
}

// Kernels reports which CPU kernel tiers the dispatcher uses, for a hardware report (goinfer's `check --hardware`).
type Kernels struct {
	Arch string
	// Detected is what the CPU supports, read fresh; Active is what the dispatcher uses now. They differ when a build tag forced a fallback, or SelfCheck repaired one.
	Detected, Active []string
	// Forced is ForcedFallbacks().
	Forced []string
}

// ActiveKernels reports the ISA tiers detected and in use ("avx2", "f16c", "avx512vnni", "avx512vnni+vl", "popcnt" on amd64; "dotprod" on arm64).
func ActiveKernels() Kernels {
	arch, detected, active := archKernels()
	slices.Sort(detected)
	slices.Sort(active)
	return Kernels{Arch: arch, Detected: detected, Active: active, Forced: ForcedFallbacks()}
}
