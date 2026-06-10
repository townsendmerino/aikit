// Package bench is a reproducible recall + latency harness for aikit's dense
// retrieval indexes — Flat (exact), HNSW (approximate), and FlatI8 (int8). It
// turns "parity-tested" into concrete, comparable numbers: recall@k measured
// against the exact Flat scan, per-query latency percentiles (p50/p95/p99), build
// time, and index memory. Run it over your own corpus, or use the harness tests
// (bench_test.go) for the synthetic-scale and real-embedding tables.
//
// It is a benchmarking tool, not a retrieval primitive — Experimental, outside
// the 1.0 compatibility guarantee.
package bench

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/townsendmerino/aikit/ann"
)

// Result is one index's measured profile over a query set.
type Result struct {
	Name      string  // "Flat", "HNSW", "FlatI8"
	N, Dim, K int     // corpus size, dimension, top-k
	BuildMs   float64 // index build time
	Recall    float64 // mean recall@k vs the exact Flat top-k (Flat itself = 1.0)
	P50, P95  float64 // query latency percentiles, ms
	P99, Mean float64 // ms
	MemMB     float64 // index storage (HNSW includes the graph, via MarshalBinary)
	QueriesA  int     // number of queries measured
}

// Run benchmarks Flat, HNSW, and FlatI8 over corpus (each vector L2-normalized,
// the ann invariant), using queries as the workload and k as top-k. Recall is
// measured against Flat's exact top-k, so Flat reports 1.0 by definition. cfg
// tunes the HNSW build.
func Run(corpus, queries [][]float32, k int, cfg ann.Config) []Result {
	n := len(corpus)
	d := 0
	if n > 0 {
		d = len(corpus[0])
	}

	t0 := time.Now()
	flat := ann.New(corpus)
	flatBuild := msSince(t0)
	t0 = time.Now()
	hnsw := ann.BuildHNSW(corpus, cfg)
	hnswBuild := msSince(t0)
	t0 = time.Now()
	fi8 := ann.NewFlatI8(corpus)
	fi8Build := msSince(t0)

	// Ground truth: the exact Flat top-k for each query.
	truth := make([]map[int]bool, len(queries))
	for i, q := range queries {
		truth[i] = idxSet(flat.Query(q, k))
	}

	hnswMem := float64(len(mustMarshal(hnsw)))
	return []Result{
		measure("Flat", flat.Query, queries, k, truth, n, d, flatBuild, float64(n*d*4)),
		measure("HNSW", hnsw.Query, queries, k, truth, n, d, hnswBuild, hnswMem),
		measure("FlatI8", fi8.Query, queries, k, truth, n, d, fi8Build, float64(n*d+n*4)),
	}
}

func measure(name string, query func([]float32, int) []ann.Hit, queries [][]float32, k int, truth []map[int]bool, n, d int, buildMs, memBytes float64) Result {
	lat := make([]float64, len(queries))
	var recallSum float64
	for i, q := range queries {
		t0 := time.Now()
		hits := query(q, k)
		lat[i] = msSince(t0)
		want := truth[i]
		got := 0
		for _, h := range hits {
			if want[h.Index] {
				got++
			}
		}
		if len(want) > 0 {
			recallSum += float64(got) / float64(len(want))
		}
	}
	sort.Float64s(lat)
	return Result{
		Name: name, N: n, Dim: d, K: k, BuildMs: buildMs,
		Recall: recallSum / float64(max(len(queries), 1)),
		P50:    pct(lat, 50), P95: pct(lat, 95), P99: pct(lat, 99), Mean: meanOf(lat),
		MemMB:    memBytes / 1e6,
		QueriesA: len(queries),
	}
}

// Table renders results as a GitHub-flavored Markdown table — paste-ready for a
// README or a benchmark report.
func Table(results []Result) string {
	var b strings.Builder
	fmt.Fprintf(&b, "| index | N | dim | recall@%d | build ms | p50 ms | p95 ms | p99 ms | mem MB |\n", results[0].K)
	b.WriteString("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range results {
		fmt.Fprintf(&b, "| %s | %d | %d | %.4f | %.1f | %.3f | %.3f | %.3f | %.1f |\n",
			r.Name, r.N, r.Dim, r.Recall, r.BuildMs, r.P50, r.P95, r.P99, r.MemMB)
	}
	return b.String()
}

func msSince(t time.Time) float64 { return float64(time.Since(t).Nanoseconds()) / 1e6 }

func idxSet(hits []ann.Hit) map[int]bool {
	s := make(map[int]bool, len(hits))
	for _, h := range hits {
		s[h.Index] = true
	}
	return s
}

// pct returns the p-th percentile (0..100) of an already-sorted slice (ms).
func pct(sorted []float64, p int) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := p * (len(sorted) - 1) / 100
	return sorted[i]
}

func meanOf(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func mustMarshal(h *ann.HNSW) []byte {
	b, err := h.MarshalBinary()
	if err != nil {
		panic(err)
	}
	return b
}
