//go:build arm64

package linalg

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// DotProd versus the base SMULL/SADALP kernels, interleaved in ONE process, on the production decode and prefill projections of Qwen2.5 0.5B / 1.5B / 7B
// (goinfer's hardware-coverage task; the pre-registration and the verdict rule are docs/measurements/dotprod-windows-arm-2026-10-04.md in goinfer, and the rule's code is dotprod_ab_stats_test.go).
//
// Why in one process: hasDotProd is a package flag read at call time by the dot kernels and the tiles, so one process can flip it between runs. Each cell runs the arms back to back, round after round,
// alternating which arm goes first, so slow drift on a shared machine hits both and an arm that only wins going second shows up as an order effect. The arms:
//
//	base       canonical int4 / int8 weights, hasDotProd=false: the SMULL/SADALP kernels a core without DotProd (or aikit before v1.56.0 on Windows ARM) runs
//	dot        the PRODUCTION DotProd path: int4 as the repacked-only row4 layout (what goinfer's loaders build on arm64), int8 unchanged, hasDotProd=true
//	dot-canon  (int4 only) canonical layout with SDOT kernels: attribution of the layout versus the instruction
//
// The headline per cell is base/dot. A kernel-level result gives DIRECTION, not size: the repo's own finding is that a kernel microbenchmark's served effect ranged 0.05-1.72x of it. It is enabled by
// AIKIT_DOTPROD_AB=1, and skips on a core without DotProd (nothing to compare). Run it on the machine you are asking about; a shared CI runner is disclosed as such in the record.

type abModel struct {
	name              string
	hidden, kv, inter int
	withM64           bool // prefill-sized cells: skipped for the 7B, where one M=64 call is seconds
}

var abModels = []abModel{
	{"qwen2.5-0.5B", 896, 128, 4864, true},
	{"qwen2.5-1.5B", 1536, 256, 8960, true},
	{"qwen2.5-7B", 3584, 512, 18944, false},
}

type abProj struct {
	name string
	K, N int // K = input features, N = output features
}

func abProjections(m abModel) []abProj {
	return []abProj{{"q_proj", m.hidden, m.hidden}, {"k_proj", m.hidden, m.kv}, {"o_proj", m.hidden, m.hidden}, {"gate", m.hidden, m.inter}, {"down", m.inter, m.hidden}}
}

func abEnvInt(name string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(name)); err == nil && v > 0 {
		return v
	}
	return def
}

// abTime runs fn n times and returns ns per call.
func abTime(n int, fn func()) float64 {
	t0 := time.Now()
	for i := 0; i < n; i++ {
		fn()
	}
	return float64(time.Since(t0).Nanoseconds()) / float64(n)
}

type abArm struct {
	name string
	dot  bool // hasDotProd while this arm runs
	w    *WeightMat
}

func TestDotProdVsBase_AB(t *testing.T) {
	if os.Getenv("AIKIT_DOTPROD_AB") == "" {
		t.Skip("AIKIT_DOTPROD_AB unset: this is a measurement, run on purpose")
	}
	if !detectDotProd() {
		t.Skip("this core has no DotProd: nothing to compare")
	}
	orig := hasDotProd
	t.Cleanup(func() { hasDotProd = orig })
	rounds := abEnvInt("AIKIT_DOTPROD_AB_ROUNDS", 21)
	blockNs := float64(abEnvInt("AIKIT_DOTPROD_AB_BLOCK_MS", 20)) * 1e6
	t.Logf("DotProd A/B: arch %s/%s, %d logical CPUs, GOMAXPROCS %d, %d rounds per cell (alternating arm order), >=%.0f ms per timed block", runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), runtime.GOMAXPROCS(0), rounds, blockNs/1e6)
	t.Logf("ActiveKernels = %+v", ActiveKernels())

	var m1Verdicts []abVerdict
	var slowerCells []string
	only := os.Getenv("AIKIT_DOTPROD_AB_MODEL") // substring filter, for a smoke run
	for _, mod := range abModels {
		if only != "" && !strings.Contains(mod.name, only) {
			continue
		}
		for _, p := range abProjections(mod) {
			// One f32 source per projection, quantized into every arm's weights, then released: the 7B gate is 270 MB of f32.
			src := randF(p.N * p.K)
			for _, quant := range []string{"int4", "int8"} {
				arms := abBuildArms(t, quant, src, p)
				for _, M := range []int{1, 64} {
					if M == 64 && !mod.withM64 {
						continue
					}
					for _, workers := range []int{1, 0} {
						label := fmt.Sprintf("%s %s %s K=%d N=%d M=%d workers=%d", quant, mod.name, p.name, p.K, p.N, M, workers)
						a := randF(M * p.K)
						dst := make([][]float32, len(arms))
						for i := range dst {
							dst[i] = make([]float32, M*p.N)
						}
						ws := &Workspace{}
						ws.SetWorkers(workers)
						run := func(i int) { arms[i].w.MatmulBTInto(ws, a, dst[i], M) }
						// Both arms must compute the same thing, or the timings compare different work.
						for i := range arms {
							hasDotProd = arms[i].dot
							run(i)
						}
						if d := abRelDiff(dst[0], dst[1]); d > 1e-4 {
							t.Fatalf("%s: base and dot results differ by %.2e relative: not the same computation, so a timing would be meaningless", label, d)
						}
						// Calibrate the block length on the base arm, then alternate.
						hasDotProd = arms[0].dot
						n := 1
						if t1 := abTime(1, func() { run(0) }); t1 < blockNs {
							n = int(math.Min(2000, math.Ceil(blockNs/math.Max(t1, 1))))
						}
						ratios := make([][]float64, len(arms)-1) // base / arm_i for each non-base arm
						var baseFirst []bool
						ns := make([]float64, len(arms))
						for r := 0; r < rounds; r++ {
							order := make([]int, len(arms))
							for i := range order {
								order[i] = i
							}
							first := r%2 == 0
							if !first {
								for i, j := 0, len(order)-1; i < j; i, j = i+1, j-1 {
									order[i], order[j] = order[j], order[i]
								}
							}
							for _, i := range order {
								hasDotProd = arms[i].dot
								run(i) // warm this arm after the switch
								ns[i] = abTime(n, func() { run(i) })
							}
							for i := 1; i < len(arms); i++ {
								ratios[i-1] = append(ratios[i-1], ns[0]/ns[i])
							}
							baseFirst = append(baseFirst, first)
						}
						for i := 1; i < len(arms); i++ {
							c := abSummarize(ratios[i-1], baseFirst)
							v := abClassify(c)
							t.Logf("AB %-62s %-9s speedup median %.3f IQR [%.3f, %.3f] faster-rounds %.0f%% base-first %.3f dot-first %.3f -> %s", label, "base/"+arms[i].name, c.Median, c.Q1, c.Q3, 100*c.FracFaster, c.MedianFwd, c.MedianRev, v)
							if arms[i].name == "dot" {
								if M == 1 {
									m1Verdicts = append(m1Verdicts, v)
								}
								if v == abSlower {
									slowerCells = append(slowerCells, label)
								}
							}
						}
					}
				}
			}
		}
	}
	claim, faster, slower := abClaim(m1Verdicts)
	t.Logf("SUMMARY (decode cells, M=1, production DotProd arm vs base): %d cells, %d FASTER, %d SLOWER, %d other; overall claim \"DotProd is faster on this core\" = %v (needs >= %.0f%% FASTER and none SLOWER)",
		len(m1Verdicts), faster, slower, len(m1Verdicts)-faster-slower, claim, 100*abClaimFasterMin)
	for _, c := range slowerCells {
		t.Logf("SLOWER CELL: %s", c)
	}
}

// abBuildArms builds the weights each arm runs: from the same f32 source, so the quantized bytes are identical across arms.
func abBuildArms(t *testing.T, quant string, src []float32, p abProj) []abArm {
	t.Helper()
	switch quant {
	case "int4":
		hasDotProd = false
		canon := QuantizeInt4(src, p.N, p.K, 32)
		base := canon // an independent copy of the struct; the byte slices are shared and never written
		hasDotProd = true
		q4, s16, _, okView := canon.Int4F16()
		if !okView {
			t.Fatalf("int4 weight has no canonical view")
		}
		r4 := RepackW4A8Row4(q4, p.N, p.K, 32)
		r4s := RepackW4A8Row4ScalesF16(s16, p.N, p.K, 32)
		row4, ok := WrapInt4Row4OnlyF16(r4, r4s, p.N, p.K, 32)
		if !ok {
			t.Fatalf("row4-only wrap rejected %dx%d: shape or core does not qualify", p.N, p.K)
		}
		dc := canon
		return []abArm{{"base", false, &base}, {"dot", true, &row4}, {"dot-canon", true, &dc}}
	default:
		hasDotProd = true
		w := QuantizeInt8(src, p.N, p.K, true)
		w2 := w
		return []abArm{{"base", false, &w}, {"dot", true, &w2}}
	}
}

func abRelDiff(a, b []float32) float64 {
	var maxAbs, maxDiff float64
	for i := range a {
		if v := math.Abs(float64(a[i])); v > maxAbs {
			maxAbs = v
		}
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > maxDiff {
			maxDiff = d
		}
	}
	if maxAbs == 0 {
		return maxDiff
	}
	return maxDiff / maxAbs
}
