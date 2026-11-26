package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// TestFMAIssueProbe is the marginal-FMA injection issue-width probe from
// docs/internal/priors-microgpt-c.md §1 — an external prior, run for real
// here for the first time. That doc's own words: "One bench file, in-process
// A/B... Until someone runs it on our boxes it stays a prior, not a
// practice." This is that run.
//
// Method (theirs): inject N independent dead FMAs into a hot loop and
// measure the marginal ns cost per added FMA. If it matches the core's
// achievable FMA throughput, the loop was already issue-limited — busy, not
// waiting — and further scheduling/unrolling/load-reduction work on it is
// provably wasted. A gap means idle issue slots exist and are worth hunting.
//
// Candidate here: dequantRowInt8 at K=768, a real weight-row width
// (dequant_i8_arm64.go's own comment: "every real weight row here — K is
// 768/2048/3072") — the "dequant inner loops" candidate the prior names, and
// the only one of its three candidates that actually exists in `linalg`
// (norm/RoPE live in package encoder; "matters at small K" wasn't a citation
// of an existing finding, just a generic bucket — not forcing either).
//
// Comparison, and why it's shaped this way: the injected dead work is scalar
// math.FMA calls (real hardware FMA instructions — math.FMA is one of the Go
// compiler's few float intrinsics, specifically so this is a true fused
// multiply-add, not a separate mul+add pair that plain a*b+c deliberately
// compiles to, to preserve IEEE-754 reproducibility). That's scalar,
// single-lane — NOT the same shape as linalg.MeasuredFMAPeakGFLOPS()'s
// NEON/AVX2-width-saturating assembly ceiling, so comparing against that
// directly would be apples to oranges. Instead this measures the SAME
// dead-FMA loop TWICE: once alone (nothing else running), once stacked on
// top of dequantRowInt8. Both measurements are the identical instruction
// shape, so their marginal costs are directly comparable to each other
// without needing any external spec:
//
//   - stacked marginal cost ≈ alone marginal cost  → dequant left idle issue
//     slots the FMAs filled for free; NOT issue-limited (likely waiting on
//     memory, given dequant is a load/sign-extend/convert/multiply/store chain).
//   - stacked marginal cost > alone marginal cost   → dequant was already
//     occupying those slots; issue-limited, the dotNEON2x8 story
//     (cpu-acceleration.md: "compute-bound... no load-reduction lever helps").
//
// linalg.MeasuredFMAPeakGFLOPS() is still logged, converted to ns/FMA, as
// the measuring-performance.md §1.2-required physical-plausibility sanity
// check: the alone-loop's marginal cost must be SLOWER (higher ns/FMA) than
// the vector peak, since it's scalar work on one lane — if it isn't, the
// measurement is measuring nothing and should not be trusted.
func TestFMAIssueProbe(t *testing.T) {
	harnessOnly(t)
	const K = 768 // real weight-row width; see doc comment

	rng := rand.New(rand.NewSource(1))
	q := make([]int8, K)
	for i := range q {
		q[i] = int8(rng.Intn(255) - 128)
	}
	dst := make([]float32, K)
	const scale = float32(0.01)

	const maxN = 64
	acc := make([]float64, maxN)

	ns := []float64{0, 8, 16, 32, 64}
	stackedNs := make([]float64, len(ns))
	aloneNs := make([]float64, len(ns))

	// best-of-3 per point, mirroring measureFMAPeak's "max of several short
	// runs" — for a duration metric (lower = less noise), best means minimum.
	const repeats = 3

	for i, nf := range ns {
		n := int(nf)

		stackedNs[i] = math.Inf(1)
		for range repeats {
			resetAcc(acc)
			r := testing.Benchmark(func(b *testing.B) {
				for b.Loop() {
					dequantRowInt8(dst, q, scale)
					deadFMAs(acc[:n])
				}
			})
			stackedNs[i] = min(stackedNs[i], float64(r.NsPerOp()))
		}

		aloneNs[i] = math.Inf(1)
		for range repeats {
			resetAcc(acc)
			r := testing.Benchmark(func(b *testing.B) {
				for b.Loop() {
					deadFMAs(acc[:n])
				}
			})
			aloneNs[i] = min(aloneNs[i], float64(r.NsPerOp()))
		}
	}
	sinkFMAProbeDst = dst
	sinkFMAProbeAcc = acc

	stackedSlope := lstsqSlope(ns, stackedNs)
	aloneSlope := lstsqSlope(ns, aloneNs)

	t.Logf("dequantRowInt8(K=%d) + N dead FMAs, stacked:  %v ns/op over N=%v (marginal %.3f ns/FMA)", K, stackedNs, ns, stackedSlope)
	t.Logf("N dead FMAs alone:                            %v ns/op over N=%v (marginal %.3f ns/FMA)", aloneNs, ns, aloneSlope)

	ratio := stackedSlope / aloneSlope
	switch {
	case aloneSlope <= 0 || stackedSlope <= 0:
		t.Fatalf("non-positive marginal slope (stacked=%.4f alone=%.4f) — measurement produced nothing usable", stackedSlope, aloneSlope)
	case ratio < 1.05:
		t.Logf("VERDICT: dequantRowInt8(K=%d) is NOT issue-limited — stacked/alone = %.2f (~1.0). Idle issue slots exist; a load-reduction or scheduling change could plausibly help.", K, ratio)
	default:
		t.Logf("VERDICT: dequantRowInt8(K=%d) IS issue-limited — stacked/alone = %.2f (>1.0, the dead FMAs contended for slots dequant was using). Matches dotNEON2x8's story (cpu-acceleration.md): compute/issue-bound, no scheduling lever helps here.", K, ratio)
	}

	// §1.2 physical-plausibility sanity check: the alone-loop is scalar,
	// single-lane math.FMA — it must be slower (higher ns/FMA) than the
	// SIMD-width-saturating assembly ceiling, or the measurement is bogus.
	if gflops, ok := measuredFMAPeakGFLOPS(); ok {
		peakNsPerFMA := 2.0 / gflops // 1 FMA = 2 flops
		t.Logf("context: measured SIMD FMA peak %.1f GFLOPS = %.4f ns/FMA (vector-width-saturating; scalar alone-loop above should be slower)", gflops, peakNsPerFMA)
		if aloneSlope < peakNsPerFMA {
			t.Errorf("implausible: scalar alone-loop marginal cost (%.4f ns/FMA) is FASTER than the measured vector peak (%.4f ns/FMA) — not physically possible, don't trust this run", aloneSlope, peakNsPerFMA)
		}
	} else {
		t.Log("context: no FMA peak probe available on this arch; skipping the plausibility cross-check")
	}
}

func resetAcc(acc []float64) {
	for i := range acc {
		acc[i] = 1.0
	}
}

// deadFMAs performs one independent hardware FMA into each accumulator slot.
// Separate slots, not one running sum, so the instructions have no data
// dependency between them — occupying distinct issue slots, not serializing
// through one, is the entire point of the injection. math.FMA (not a*b+c)
// because it is one of the Go compiler's few float intrinsics and therefore
// emits a true single FMA on every architecture, which is what this probe
// needs to inject a known unit of work.
//
// CORRECTION 2026-09-03 (task-simd-audit.md S-10). This comment used to claim
// that plain a*b+c is "deliberately NOT auto-fused into hardware FMA", on the
// grounds that fusing would break Go's IEEE-754 reproducibility guarantee.
// THAT IS WRONG ON arm64, and it is wrong in this very package. The Go spec
// explicitly PERMITS fusing an explicit floating-point multiply-add, and the
// arm64 backend takes it:
//
//	$ go build -gcflags=-S ./linalg | grep -c 'FMADD[SD]'
//	59            # 46 FMADDS + 13 FMADDD
//	$ grep -rn 'math\.FMA' linalg/*.go | grep -v _test | wc -l
//	0             # not one explicit call — every FMADD above is auto-fused
//
// The sites include linalg/dot.go's scalar tail (`sum += a[k] * b[k]`) and,
// more consequentially, matmul_qk_acc64.go — one of the f64 attention kernels
// whose bit-identity is load-bearing.
//
// The consequence is the reason this correction matters rather than being
// pedantry: a*b+c and fma(a,b,c) round DIFFERENTLY (one rounding, not two), so
// the same Go source produces different f32/f64 results on arm64 than on an
// amd64 target that did not fuse. That is precisely why goinfer keys its
// goldens by GOARCH. Anyone auditing the bit-identity of the pure-Go tails who
// believed the old comment would conclude the arch split was unnecessary.
//
// fmaX/fmaY sit a few ULPs from the identity (1, 0) — real, data-dependent
// arithmetic the compiler cannot constant-fold, without letting the
// accumulator drift far enough across repeated calls to risk overflow/NaN
// (resetAcc also re-seeds between every timed trial, so drift never
// compounds across trials either).
func deadFMAs(acc []float64) {
	for i := range acc {
		acc[i] = math.FMA(acc[i], fmaX, fmaY)
	}
}

const (
	fmaX = 1.0 + 1e-12
	fmaY = 1e-12
)

// lstsqSlope returns the least-squares linear fit's slope for y ~ slope*x +
// intercept — the marginal cost per unit x (here, ns per added FMA) across
// the whole N sweep, not just the two endpoints.
func lstsqSlope(xs, ys []float64) float64 {
	n := float64(len(xs))
	var sx, sy, sxy, sxx float64
	for i := range xs {
		sx += xs[i]
		sy += ys[i]
		sxy += xs[i] * ys[i]
		sxx += xs[i] * xs[i]
	}
	return (n*sxy - sx*sy) / (n*sxx - sx*sx)
}

var (
	sinkFMAProbeDst []float32
	sinkFMAProbeAcc []float64
)
