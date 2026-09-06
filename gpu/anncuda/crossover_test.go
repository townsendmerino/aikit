//go:build linux

package anncuda_test

// crossover_test.go is the CUDA half of the ANN GPU first slice (docs/BENCH-gpu.md) — the exact
// mirror of gpu/annmetal/crossover_test.go: FlatI8 CPU-SIMD vs CUDA QueryBatch (N × batch) on
// REAL Model2Vec embeddings, gated on recall + parity against the exact-CPU top-k, emitting
// bench.Record rows that the SAME report generator (bench/cmd/benchreport) renders.
//
// It lives here so importing anncuda — the opt-in that registers the ann.Backend — never pulls
// the GPU into the root cgo-free module. Not a default `go test`: needs a CUDA device, the
// Model2Vec checkpoint, and AIKIT_GPU_BENCH set (a documented periodic pass).
//
// The one platform difference from Metal: transfers are EXPLICIT copies on a discrete GPU (no
// UMA), so H2D (query upload) and D2H (the M*N score matrix readback) are real costs. They are
// folded into Wall here (the true user-facing latency); Compute is set equal to Wall, and the
// backend would need instrumentation to split them out further.

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	_ "github.com/townsendmerino/aikit/gpu/anncuda" // registers the CUDA ann.Backend via init

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/bench"
	"github.com/townsendmerino/aikit/embed"
	gpu "github.com/townsendmerino/aikit/gpu"
)

const (
	modelDir  = "../../testdata/model" // potion-code-16M (Model2Vec), same as benchmarks/
	kTop      = 10
	queryPool = 256
	warmup    = 3
	iters     = 10
)

// shardShares are the CPU∥GPU shard-split candidates the crossover sweep also
// measures (backend "cpu+cuda"): at N=100_000 they put the device shard at
// ~35k rows (below both gpu/anncuda's and gpu/annmetal's topkMinN — forces
// ScoreBatch on both) and ~60k rows (above CUDA's 50_000 threshold — CUDA
// gets on-device top-k fusion for its shard here — but still below Metal's
// 100_000, so Metal's mirror of this sweep stays in the ScoreBatch-fallback
// regime at both shares). See gpu/annmetal/crossover_test.go's identical
// comment for the full reasoning; kept in sync across both files by hand.
var shardShares = []float64{0.35, 0.6}

func envInts(key string, def []int) []int {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	var out []int
	for s := range strings.SplitSeq(v, ",") {
		if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
			out = append(out, n)
		}
	}
	if len(out) == 0 {
		return def
	}
	return out
}

func TestCUDAANNCrossover(t *testing.T) {
	if os.Getenv("AIKIT_GPU_BENCH") == "" {
		t.Skip("periodic GPU pass — set AIKIT_GPU_BENCH=1 to run (docs/BENCH-gpu.md)")
	}
	if _, err := os.Stat(modelDir + "/model.safetensors"); err != nil {
		t.Skipf("no Model2Vec model at %s — see benchmarks/README.md", modelDir)
	}
	m, err := embed.LoadFromFS(os.DirFS(modelDir), ".")
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	dim := len(m.Encode("probe"))

	gpuName := "NVIDIA GPU (CUDA)"
	if dev, derr := gpu.CreateSystemDefaultDevice(); derr == nil {
		gpuName = dev.Name()
		dev.ReleaseObjects()
	}
	machine := envStr("AIKIT_BENCH_MACHINE", "nvidia")
	chip := envStr("AIKIT_BENCH_CHIP", "")
	recordsPath := envStr("AIKIT_BENCH_RECORDS", "../../docs/bench-records/crossover-cuda.jsonl")
	_ = os.MkdirAll(filepath.Dir(recordsPath), 0o755)
	_ = os.Remove(recordsPath)

	meta := bench.CaptureMeta()
	if meta.AikitCommit == "" {
		meta.AikitCommit = envStr("AIKIT_BENCH_COMMIT", "unknown")
	}
	meta.WarmupLaunches, meta.Iters = warmup, iters
	cpuDev := bench.HostDevice()
	cpuDev.Machine, cpuDev.Chip = machine, chip
	gpuDev := bench.HostDevice()
	gpuDev.Machine, gpuDev.GPU = machine, gpuName
	gpuDev.SMorFamily = envStr("AIKIT_BENCH_SM", "")

	Ns := envInts("AIKIT_BENCH_N", []int{10_000, 100_000})
	batches := envInts("AIKIT_BENCH_BATCH", []int{1, 8, 64, 256})

	for _, N := range Ns {
		vecs := genCorpus(m, N)
		exact := ann.New(vecs)
		fi8 := ann.NewFlatI8(vecs)

		qpool := genQueries(m, queryPool)
		truth := make([]map[int]bool, len(qpool))
		for i, q := range qpool {
			truth[i] = bench.TruthSet(exact.Query(q, kTop))
		}

		cpuThr := map[int]float64{}
		cpuRecall := map[int]float64{}
		cpuHitsByBatch := map[int][][]ann.Hit{}
		for _, b := range batches {
			qs := qpool[:b]
			for range warmup {
				fi8.QueryBatch(qs, kTop)
			}
			var hits [][]ann.Hit
			best := bench.MinDuration(iters, func() { hits = fi8.QueryBatch(qs, kTop) })
			cpuThr[b] = float64(b) / best.Seconds()
			cpuRecall[b] = meanRecall(hits, truth)
			cpuHitsByBatch[b] = hits
		}

		// --- single-query FlatI8.Query — the path gemv_w8a8 serves, and a DIFFERENT
		// kernel from the batch one. Without this row the harness cannot answer "is
		// EnableGPU worth it for one query", which is the question a caller with an
		// interactive workload actually has. Mirrors the Metal harness (ab803ab). ---
		sq := qpool[:min(64, queryPool)]
		cpuQThr, cpuQHits := timeSingleQuery(fi8, sq, kTop)
		cpuQRecall := meanRecall(cpuQHits, truth[:len(sq)])

		t0 := time.Now()
		if err := fi8.EnableGPU(); err != nil {
			t.Skipf("EnableGPU: %v (no CUDA device?)", err)
		}
		oneTimeMs := float64(time.Since(t0).Nanoseconds()) / 1e6

		gpuQThr, gpuQHits := timeSingleQuery(fi8, sq, kTop)
		gpuQRecall := meanRecall(gpuQHits, truth[:len(sq)])
		qParityOK, qMaxDelta := parity(gpuQHits, cpuQHits, cpuQRecall, gpuQRecall)
		if !qParityOK {
			t.Errorf("PARITY(Query): N=%d single-query CUDA ranking diverged from CPU int8 — a bug, not a knob.", N)
		}
		qsp := gpuQThr / cpuQThr
		qShape := bench.Shape{N: N, Dim: dim, Batch: 1, K: kTop}
		cpuQR, gpuQR := cpuQRecall, gpuQRecall
		cpuQRec := bench.Record{
			Workload: "ann.FlatI8.Query", Backend: "cpu-simd", Precision: "int8",
			Device: cpuDev, Shape: qShape,
			Timing:     bench.Timing{Compute: msPer(cpuQThr, 1), Wall: msPer(cpuQThr, 1)},
			Throughput: cpuQThr, ThroughputUnit: "queries/s",
			Quality: bench.Quality{RecallAtK: &cpuQR, ParityOK: true},
			Meta:    meta,
		}
		gpuQRec := bench.Record{
			Workload: "ann.FlatI8.Query", Backend: "cuda", Precision: "int8",
			Device: gpuDev, Shape: qShape,
			Timing:     bench.Timing{OneTime: oneTimeMs, Compute: msPer(gpuQThr, 1), Wall: msPer(gpuQThr, 1)},
			Throughput: gpuQThr, ThroughputUnit: "queries/s",
			Quality:      bench.Quality{RecallAtK: &gpuQR, ParityOK: qParityOK, MaxDeltaVsCPU: qMaxDelta},
			SpeedupVsCPU: &qsp,
			Meta:         meta,
		}
		if err := bench.AppendRecords(recordsPath, cpuQRec, gpuQRec); err != nil {
			t.Fatalf("append query records: %v", err)
		}
		t.Logf("N=%-7d Query(1)     cpu %8.0f q/s  cuda %8.0f q/s  %5.2f×  recall cpu %.4f cuda %.4f  parity %v",
			N, cpuQThr, gpuQThr, qsp, cpuQR, gpuQR, qParityOK)

		for _, b := range batches {
			qs := qpool[:b]
			for range warmup {
				fi8.QueryBatch(qs, kTop)
			}
			var hits [][]ann.Hit
			best := bench.MinDuration(iters, func() { hits = fi8.QueryBatch(qs, kTop) })
			wallMs := float64(best.Nanoseconds()) / 1e6
			gpuThr := float64(b) / best.Seconds()
			gRecall := meanRecall(hits, truth)
			parityOK, maxDelta := parity(hits, cpuHitsByBatch[b], cpuRecall[b], gRecall)
			if !parityOK {
				t.Errorf("PARITY: N=%d batch=%d — CUDA ranking diverged from CPU int8 (recall %.4f vs %.4f). This is a bug, not a knob.",
					N, b, gRecall, cpuRecall[b])
			}
			sp := gpuThr / cpuThr[b]
			shape := bench.Shape{N: N, Dim: dim, Batch: b, K: kTop}
			cr, gr := cpuRecall[b], gRecall

			cpuRec := bench.Record{
				Workload: "ann.FlatI8.QueryBatch", Backend: "cpu-simd", Precision: "int8",
				Device: cpuDev, Shape: shape,
				Timing:     bench.Timing{Compute: msPer(cpuThr[b], b), Wall: msPer(cpuThr[b], b)},
				Throughput: cpuThr[b], ThroughputUnit: "queries/s",
				Quality: bench.Quality{RecallAtK: &cr, ParityOK: true},
				Meta:    meta,
			}
			gpuRec := bench.Record{
				Workload: "ann.FlatI8.QueryBatch", Backend: "cuda", Precision: "int8",
				Device: gpuDev, Shape: shape,
				// Discrete GPU: H2D (query upload) + D2H (the M*N score matrix) are explicit
				// copies, folded into Wall (the real latency). one_time is the residency upload.
				Timing:     bench.Timing{OneTime: oneTimeMs, Compute: wallMs, Wall: wallMs},
				Throughput: gpuThr, ThroughputUnit: "queries/s",
				Quality:      bench.Quality{RecallAtK: &gr, ParityOK: parityOK, MaxDeltaVsCPU: maxDelta},
				SpeedupVsCPU: &sp,
				Meta:         meta,
			}
			if err := bench.AppendRecords(recordsPath, cpuRec, gpuRec); err != nil {
				t.Fatalf("append records: %v", err)
			}
			t.Logf("N=%-7d batch=%-4d  cpu %8.0f q/s  cuda %8.0f q/s  %5.2f×  recall cpu %.4f cuda %.4f  parity %v",
				N, b, cpuThr[b], gpuThr, sp, cr, gr, parityOK)
		}

		// --- CPU∥GPU shard-split QueryBatch: see shardShares' doc comment for the
		// topkMinN regime this sweep does and doesn't reach at these N values. ---
		for _, share := range shardShares {
			if err := fi8.EnableGPUShardSplit(share); err != nil {
				t.Logf("N=%d share=%.2f: EnableGPUShardSplit: %v (skipping)", N, share, err)
				continue
			}
			for _, b := range batches {
				qs := qpool[:b]
				for range warmup {
					fi8.QueryBatch(qs, kTop)
				}
				var hits [][]ann.Hit
				best := bench.MinDuration(iters, func() { hits = fi8.QueryBatch(qs, kTop) })
				wallMs := float64(best.Nanoseconds()) / 1e6
				thr := float64(b) / best.Seconds()
				recall := meanRecall(hits, truth)
				parityOK, maxDelta := parity(hits, cpuHitsByBatch[b], cpuRecall[b], recall)
				if !parityOK {
					t.Errorf("PARITY(shard share=%.2f): N=%d batch=%d — cpu+cuda ranking diverged from CPU int8 (recall %.4f vs %.4f).",
						share, N, b, recall, cpuRecall[b])
				}
				sp := thr / cpuThr[b]
				shape := bench.Shape{N: N, Dim: dim, Batch: b, K: kTop}
				rec := bench.Record{
					Workload: "ann.FlatI8.QueryBatch", Backend: "cpu+cuda", Precision: "int8",
					Device: gpuDev, Shape: shape,
					// Same discrete-GPU framing as the whole-corpus cuda row above: H2D+D2H
					// folded into Wall, Compute set equal to it.
					Timing:     bench.Timing{Compute: wallMs, Wall: wallMs},
					Throughput: thr, ThroughputUnit: "queries/s",
					Quality:      bench.Quality{RecallAtK: &recall, ParityOK: parityOK, MaxDeltaVsCPU: maxDelta},
					SpeedupVsCPU: &sp,
					Meta:         meta,
				}
				if err := bench.AppendRecords(recordsPath, rec); err != nil {
					t.Fatalf("append shard records: %v", err)
				}
				t.Logf("N=%-7d batch=%-4d share=%.2f (gpu shard≈%d rows)  cpu+cuda %8.0f q/s  %5.2f×  recall %.4f  parity %v",
					N, b, share, int(share*float64(N)), thr, sp, recall, parityOK)
			}
		}

		fi8.Close()
	}
	t.Logf("records → %s ; generate the doc with:\n  go run ./bench/cmd/benchreport docs/bench-records/*.jsonl > docs/BENCH-gpu-results.md", recordsPath)
}

func envStr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// timeSingleQuery times FlatI8.Query ONE query at a time (min per-query over iters),
// returning queries/s and the best pass's hits. Deliberately not QueryBatch with a
// 1-element slice: that takes the batch kernel, and the whole point of this row is the
// single-query GEMV path.
func timeSingleQuery(f *ann.FlatI8, qs [][]float32, k int) (float64, [][]ann.Hit) {
	for range warmup {
		for _, q := range qs {
			f.Query(q, k)
		}
	}
	var hits [][]ann.Hit
	best := bench.MinDuration(iters, func() {
		h := make([][]ann.Hit, len(qs))
		for i, q := range qs {
			h[i] = f.Query(q, k)
		}
		hits = h
	})
	// MinDuration times the whole sweep of len(qs) queries; report per query.
	best /= time.Duration(len(qs))
	return 1.0 / best.Seconds(), hits
}

func msPer(thr float64, batch int) float64 {
	if thr == 0 {
		return 0
	}
	return float64(batch) / thr * 1000
}

var (
	verbs = []string{"get", "set", "make", "parse", "read", "write", "open", "close", "find", "build", "scan", "load", "save", "merge", "sort", "hash", "diff", "join"}
	nouns = []string{"User", "Index", "Buffer", "Token", "Vector", "Config", "Result", "Node", "Query", "Cache", "Graph", "Record", "Shard", "Frame"}
	types = []string{"int", "string", "[]byte", "error", "bool", "float64", "[]int", "rune"}
)

func genCorpus(m *embed.StaticModel, n int) [][]float32 {
	out := make([][]float32, n)
	for i := range n {
		v, nn, ty := verbs[i%len(verbs)], nouns[(i/len(verbs))%len(nouns)], types[i%len(types)]
		out[i] = m.Encode(fmt.Sprintf("func %s%s%d(in %s) (%s, error)", v, nn, i, ty, ty))
	}
	return out
}

func genQueries(m *embed.StaticModel, n int) [][]float32 {
	out := make([][]float32, n)
	for i := range n {
		v, nn, ty := verbs[(i*7)%len(verbs)], nouns[(i*3)%len(nouns)], types[(i*5)%len(types)]
		out[i] = m.Encode(fmt.Sprintf("%s the %s %s value at %d", v, nn, ty, i))
	}
	return out
}

func meanRecall(hits [][]ann.Hit, truth []map[int]bool) float64 {
	if len(hits) == 0 {
		return 0
	}
	var s float64
	for i, h := range hits {
		s += bench.RecallAt(h, truth[i])
	}
	return s / float64(len(hits))
}

func parity(gpu, cpu [][]ann.Hit, cpuRecall, gpuRecall float64) (bool, float64) {
	ok := true
	for i := range gpu {
		if !sameSet(gpu[i], cpu[i]) {
			ok = false
			break
		}
	}
	d := gpuRecall - cpuRecall
	if d < 0 {
		d = -d
	}
	return ok, d
}

func sameSet(a, b []ann.Hit) bool {
	if len(a) != len(b) {
		return false
	}
	ai := make([]int, len(a))
	bi := make([]int, len(b))
	for i := range a {
		ai[i], bi[i] = a[i].Index, b[i].Index
	}
	sort.Ints(ai)
	sort.Ints(bi)
	for i := range ai {
		if ai[i] != bi[i] {
			return false
		}
	}
	return true
}
