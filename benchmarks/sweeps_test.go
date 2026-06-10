package main

import (
	"sort"
	"testing"
	"time"

	cohnsw "github.com/coder/hnsw"
	ann "github.com/townsendmerino/aikit/ann"
)

// These sweeps document why coder/hnsw's recall row in the published table is
// fair, not a misconfiguration: its recall is construction-limited on real
// embeddings — flat across search-ef, and only crawling up with M (at growing
// memory/latency), never approaching aikit's. Run: go test -run Sweep -v.

func realSweepCorpus(t *testing.T) (corpus, queries [][]float32, truth [][]int) {
	corpus, queries = realCorpus(4000, 300)
	return corpus, queries, groundTruth(corpus, queries, 10)
}

func recallP50(queries [][]float32, truth [][]int, q func([]float32, int) []int) (float64, float64) {
	lat := make([]float64, len(queries))
	var rs float64
	for i, qq := range queries {
		t0 := time.Now()
		got := q(qq, 10)
		lat[i] = float64(time.Since(t0).Microseconds()) / 1000
		rs += recallAt(got, truth[i])
	}
	sort.Float64s(lat)
	return rs / float64(len(queries)), lat[len(lat)/2]
}

func coderQuery(g *cohnsw.Graph[int]) func([]float32, int) []int {
	return func(q []float32, k int) []int {
		res := g.Search(q, k)
		ids := make([]int, len(res))
		for i, n := range res {
			ids[i] = n.Key
		}
		return ids
	}
}

func buildCoder(corpus [][]float32, m, ef int) *cohnsw.Graph[int] {
	g := cohnsw.NewGraph[int]()
	g.M, g.EfSearch, g.Distance = m, ef, cohnsw.CosineDistance
	nodes := make([]cohnsw.Node[int], len(corpus))
	for i, v := range corpus {
		nodes[i] = cohnsw.MakeNode(i, v)
	}
	g.Add(nodes...)
	return g
}

// TestEfSweep: search-ef does not lift coder's recall (it's not search-limited),
// while aikit reaches ~1.0 at efS=64 (efC=200).
func TestEfSweep(t *testing.T) {
	corpus, queries, truth := realSweepCorpus(t)
	h := ann.BuildHNSW(corpus, ann.Config{M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	r, l := recallP50(queries, truth, func(q []float32, k int) []int { return hitIDs(h.Query(q, k)) })
	t.Logf("aikit HNSW (efC=200, efS=64): recall@10 %.4f  p50 %.3fms", r, l)
	for _, ef := range []int{64, 128, 200, 400, 800} {
		g := buildCoder(corpus, 16, ef)
		r, l := recallP50(queries, truth, coderQuery(g))
		t.Logf("coder/hnsw (M=16, ef=%d): recall@10 %.4f  p50 %.3fms", ef, r, l)
	}
}

// TestMSweep: more neighbors (M) lift coder's recall only slowly and it plateaus
// far below aikit, at multiplying memory and latency.
func TestMSweep(t *testing.T) {
	corpus, queries, truth := realSweepCorpus(t)
	for _, m := range []int{16, 32, 48, 64} {
		g := buildCoder(corpus, m, 100)
		r, l := recallP50(queries, truth, coderQuery(g))
		t.Logf("coder/hnsw (M=%d, ef=100): recall@10 %.4f  p50 %.3fms", m, r, l)
	}
}
