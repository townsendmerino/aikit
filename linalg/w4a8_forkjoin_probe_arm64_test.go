//go:build arm64

package linalg

import (
	"fmt"
	"math/rand"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// forkJoinLayer is one distinct weight matrix in the row4 layout, sized and
// packed exactly as the production decode path holds them.
type forkJoinLayer struct {
	packed4 []byte
	scales4 []float32
}

// buildForkJoinLayers packs nLayers distinct [N,K] int4 matrices into the row4
// layout. Distinct matters: the point of the work-per-barrier sweep is to
// simulate q‖k‖v — several DIFFERENT weight matrices under one fork/join — so
// reusing one matrix would measure cache residency instead.
func buildForkJoinLayers(t *testing.T, nLayers, N, K, group int) ([]forkJoinLayer, []int8, float32) {
	t.Helper()
	nGroups := K / group
	bpr := (K + 1) / 2
	rng := rand.New(rand.NewSource(31))
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	aq := make([]int8, K)
	aScale := quantizeRowInt8(a, aq)

	layers := make([]forkJoinLayer, nLayers)
	for l := range layers {
		w := make([]float32, N*K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		packed, scales := QuantizeGroupsInt4(w, N, K, group)
		p4 := make([]byte, N*bpr)
		s4 := make([]float32, N*nGroups)
		for q := 0; q < N/4; q++ {
			r0, r1, r2, r3 := q*4, q*4+1, q*4+2, q*4+3
			blk := repackSplitHalf4RowBlock(
				packed[r0*bpr:r0*bpr+bpr], packed[r1*bpr:r1*bpr+bpr],
				packed[r2*bpr:r2*bpr+bpr], packed[r3*bpr:r3*bpr+bpr], K)
			copy(p4[q*4*bpr:(q*4+4)*bpr], blk)
			sblk := interleaveScales4Row(
				scales[r0*nGroups:r0*nGroups+nGroups], scales[r1*nGroups:r1*nGroups+nGroups],
				scales[r2*nGroups:r2*nGroups+nGroups], scales[r3*nGroups:r3*nGroups+nGroups], nGroups)
			copy(s4[q*4*nGroups:(q*4+4)*nGroups], sblk)
		}
		layers[l] = forkJoinLayer{packed4: p4, scales4: s4}
	}
	return layers, aq, aScale
}

// TestW4A8ForkJoinShardTiming is the first half of docs/task-simd-audit.md
// S-02's "measure first": per-shard (start, end) timestamps across one
// Workspace.parallel fan-out, which discriminates the finding's two competing
// root causes.
//
//	(a) GOROUTINE-WAKE STAGGER — newproc/wakep wakes one spinning M at a time, so
//	    successive shards START progressively later. Signature: start times rising
//	    monotonically with shard index, durations roughly equal.
//	(b) STATIC-SHARD SKEW — equal column counts across 6 P-cores + 2 E-cores, so
//	    the E-core shards take far longer and set the barrier. Signature: start
//	    times bunched, durations BIMODAL.
//
// The two call for different remedies, which is why S-02 says to measure before
// building: (a) is fixed by dynamic chunking or by more work per barrier, (b) is
// fixed by uneven shards or by capping the fan-out to the P-core count. Reported,
// never asserted — this is a diagnostic, and a threshold here would be a guess
// about a machine rather than a contract.
type rec2 struct {
	j0, j1     int
	start, end time.Duration
}

func TestW4A8ForkJoinShardTiming(t *testing.T) {
	harnessOnly(t)
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const (
		K, group, N = 1536, 32, 8960
		nLayers     = 8
	)
	nGroups := K / group
	bpr := (K + 1) / 2
	layers, aq, aScale := buildForkJoinLayers(t, nLayers, N, K, group)
	dst := make([]float32, N)

	for _, workers := range []int{6, 8} {
		var ws Workspace
		ws.SetWorkers(workers)
		ws.SetThreshold(0)

		// Warm: first fan-out pays goroutine-pool and page-fault costs that are
		// not what this is measuring.
		for i := range nLayers {
			l := layers[i]
			ws.parallel(N/4, func(q0, q1 int) { row4SpanProbe(l, aq, aScale, dst, nGroups, bpr, q0, q1) })
		}

		const reps = 24
		agg := make([][]rec2, 0, reps)
		for rep := range reps {
			l := layers[rep%nLayers]
			recs := make([]rec2, 64)
			var slot atomic.Int32
			t0 := time.Now()
			ws.parallel(N/4, func(q0, q1 int) {
				s := time.Since(t0)
				row4SpanProbe(l, aq, aScale, dst, nGroups, bpr, q0, q1)
				i := slot.Add(1) - 1
				if int(i) < len(recs) {
					recs[i] = rec2{q0, q1, s, time.Since(t0)}
				}
			})
			n := int(slot.Load())
			recs = recs[:min(n, len(recs))]
			sort.Slice(recs, func(i, j int) bool { return recs[i].j0 < recs[j].j0 })
			agg = append(agg, recs)
		}

		// Report the median rep by total wall time, so one scheduling hiccup does
		// not become the picture.
		sort.Slice(agg, func(i, j int) bool { return agg[i][len(agg[i])-1].end < agg[j][len(agg[j])-1].end })
		med := agg[len(agg)/2]
		fmt.Fprintf(os.Stderr, "\n[shard-timing] workers=%d  (%d shards)\n", workers, len(med))
		fmt.Fprintf(os.Stderr, "  %-6s %-14s %10s %10s %10s\n", "shard", "quads", "start_us", "end_us", "dur_us")
		// max-min, NOT last-minus-first: shards are sorted by column range for
		// readability, but the ORDER THEY START IN is not the column order — the
		// goroutine the scheduler picks up first is arbitrary. Indexing the spread
		// by position produced a negative "spread" on the first run of this probe.
		var minStart, maxStart, minDur, maxDur float64
		for i, r := range med {
			st := float64(r.start.Nanoseconds()) / 1000
			en := float64(r.end.Nanoseconds()) / 1000
			du := en - st
			if i == 0 {
				minStart, maxStart, minDur, maxDur = st, st, du, du
			}
			minStart, maxStart = min(minStart, st), max(maxStart, st)
			minDur, maxDur = min(minDur, du), max(maxDur, du)
			fmt.Fprintf(os.Stderr, "  %-6d %-14s %10.1f %10.1f %10.1f\n",
				i, fmt.Sprintf("[%d,%d)", r.j0, r.j1), st, en, du)
		}
		// The discriminator: a start spread that is LARGE relative to a shard's own
		// duration is (a) wake stagger; bunched starts with bimodal durations
		// would be (b) P/E-core shard skew.
		fmt.Fprintf(os.Stderr, "  START SPREAD (max-min): %.1f us   DURATION SPREAD (max/min): %.2fx   median shard duration: %.1f us\n",
			maxStart-minStart, maxDur/max(minDur, 0.001), medDur(med))
		t.Logf("workers=%d: start spread %.1f us vs median shard duration %.1f us; duration spread %.2fx over %d shards",
			workers, maxStart-minStart, medDur(med), maxDur/max(minDur, 0.001), len(med))
	}
}

func medDur(recs []rec2) float64 {
	d := make([]float64, len(recs))
	for i, r := range recs {
		d[i] = float64((r.end - r.start).Nanoseconds()) / 1000
	}
	sort.Float64s(d)
	return d[len(d)/2]
}

// dynamicChunkRun is S-02's OTHER remedy, measured against the batch one rather
// than assumed to be equivalent: workers pull fixed-size quad blocks from an
// atomic counter instead of owning a static equal shard. It attacks the same
// wake stagger from the opposite side — a worker that wakes late simply takes
// fewer blocks, and workers that woke early keep pulling instead of finishing a
// fixed shard and idling at the barrier — so it needs no API change and no
// caller cooperation at all.
//
// chunk is S-02's own suggestion: 32 quads is ~96 KB of weights at K=1536, big
// enough that one atomic add is amortized over real work.
func dynamicChunkRun(workers, total, chunk int, body func(q0, q1 int)) {
	var next atomic.Int64
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				q0 := int(next.Add(int64(chunk))) - chunk
				if q0 >= total {
					return
				}
				body(q0, min(q0+chunk, total))
			}
		}()
	}
	wg.Wait()
}

func row4SpanProbe(l forkJoinLayer, aq []int8, aScale float32, dst []float32, nGroups, bpr, q0, q1 int) {
	var out [4]float32
	for q := q0; q < q1; q++ {
		blk := l.packed4[q*4*bpr : q*4*bpr+4*bpr]
		sblk := l.scales4[q*4*nGroups : q*4*nGroups+4*nGroups]
		dotW4A8SplitHalf4Row(&aq[0], &blk[0], &sblk[0], &out[0], nGroups)
		dst[q*4] = out[0] * aScale
		dst[q*4+1] = out[1] * aScale
		dst[q*4+2] = out[2] * aScale
		dst[q*4+3] = out[3] * aScale
	}
}

// TestW4A8WorkPerBarrier is the second and decisive half of S-02's "measure
// first", and it is a direct simulation of MatmulBTW4A8Batch rather than an
// analogy: the batch form's entire mechanism is "put several matrices under ONE
// fork/join", and that is exactly what the matsPerBarrier axis varies. The
// global quad index is mapped back to (matrix, local quad) the same way a batch
// span would map a concatenated column space.
//
// S-02's own reading of the outcome, pre-registered there: "if GB/s climbs
// toward 100+ the barrier is the limiter, if it stays ~60 the memory system is."
// So this decides whether MatmulBTW4A8Batch is worth building AT ALL, before any
// of its API surface exists. matsPerBarrier=3 is the q‖k‖v case and 2 is
// gate‖up — the two shapes goinfer would actually call.
func TestW4A8WorkPerBarrier(t *testing.T) {
	harnessOnly(t)
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const (
		K, group, N = 1536, 32, 8960
		nLayers     = 8
	)
	nGroups := K / group
	bpr := (K + 1) / 2
	layers, aq, aScale := buildForkJoinLayers(t, nLayers, N, K, group)
	dsts := make([][]float32, nLayers)
	for i := range dsts {
		dsts[i] = make([]float32, N)
	}
	quadsPer := N / 4
	bytesPerMat := float64(N) * float64(bpr+nGroups*4)

	fmt.Fprintf(os.Stderr, "\n[work-per-barrier] K=%d N=%d, %.1f MB per matrix, %d distinct matrices\n",
		K, N, bytesPerMat/(1<<20), nLayers)

	for _, workers := range []int{1, 6, 8} {
		var ws Workspace
		ws.SetWorkers(workers)
		ws.SetThreshold(0)
		fmt.Fprintf(os.Stderr, "  workers=%d\n", workers)
		var base float64
		for _, mats := range []int{1, 2, 3, 4, 8} {
			// One fork/join covering `mats` matrices: the parallel region spans
			// the CONCATENATED quad space, so there is one barrier per group of
			// matrices rather than one per matrix.
			best := 0.0
			for rep := 0; rep < 3; rep++ {
				base0 := 0
				r := testing.Benchmark(func(b *testing.B) {
					for b.Loop() {
						ws.parallel(mats*quadsPer, func(g0, g1 int) {
							for g := g0; g < g1; g++ {
								mi := g / quadsPer
								q := g % quadsPer
								l := layers[(base0+mi)%nLayers]
								row4SpanProbe(l, aq, aScale, dsts[(base0+mi)%nLayers], nGroups, bpr, q, q+1)
							}
						})
						base0 += mats
						if base0 >= nLayers {
							base0 = 0
						}
					}
				})
				gbs := float64(mats) * bytesPerMat / float64(r.NsPerOp())
				best = max(best, gbs)
			}
			if mats == 1 {
				base = best
			}
			label := ""
			switch mats {
			case 2:
				label = "  (gate‖up)"
			case 3:
				label = "  (q‖k‖v)"
			}
			fmt.Fprintf(os.Stderr, "    mats/barrier=%-2d  %6.1f GB/s   %.3fx vs 1%s\n", mats, best, best/base, label)
			t.Logf("workers=%d mats/barrier=%d: %.1f GB/s (%.3fx vs 1 matrix per barrier)%s",
				workers, mats, best, best/base, label)
		}
		// The competing remedy, at ONE matrix per barrier — i.e. with no API
		// change and no batched caller. If this reaches what mats/barrier=3 reaches,
		// MatmulBTW4A8Batch's extra surface buys nothing that the scheduler fix
		// does not already give for free.
		for _, chunk := range []int{8, 32, 128} {
			bestDyn := 0.0
			for rep := 0; rep < 3; rep++ {
				i := 0
				r := testing.Benchmark(func(b *testing.B) {
					for b.Loop() {
						l := layers[i%nLayers]
						d := dsts[i%nLayers]
						dynamicChunkRun(workers, quadsPer, chunk, func(q0, q1 int) {
							row4SpanProbe(l, aq, aScale, d, nGroups, bpr, q0, q1)
						})
						i++
					}
				})
				bestDyn = max(bestDyn, bytesPerMat/float64(r.NsPerOp()))
			}
			fmt.Fprintf(os.Stderr, "    dynamic chunk=%-4d %6.1f GB/s   %.3fx vs static 1-mat  (no API change)\n",
				chunk, bestDyn, bestDyn/base)
			t.Logf("workers=%d dynamic-chunk=%d: %.1f GB/s (%.3fx vs static 1 matrix per barrier)",
				workers, chunk, bestDyn, bestDyn/base)
		}
	}
}
