// Command bench measures aikit's dense indexes against pure-Go ANN competitors
// (coder/hnsw, chromem-go) on one common synthetic corpus, and prints a markdown
// table. Every library is driven through the same idx interface and the same
// measurement code — same corpus, same queries, same independently-computed
// ground truth — so the numbers are comparable. See README.md for methodology and
// the cgo/inference caveats on libraries not benchmarked here (bleve, hugot).
package main

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	cohnsw "github.com/coder/hnsw"
	chromem "github.com/philippgille/chromem-go"
	ann "github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/embed"
)

const modelDir = "../testdata/model" // Model2Vec potion-code-16M, for real embeddings

const (
	N        = 8000 // corpus vectors
	Dim      = 256  // matches Model2Vec potion dim
	Q        = 1000 // queries
	K        = 10   // top-k
	M        = 16   // HNSW max neighbors (aikit + coder)
	EfSearch = 64   // HNSW query-time ef (aikit + coder)
	Seed     = 1
)

// idx is the common surface every library is wrapped in, so build + query are
// timed identically across them.
type idx interface {
	name() string
	build(corpus [][]float32)
	query(q []float32, k int) []int // returns the top-k corpus ids
}

type result struct {
	name                       string
	buildMs, recall            float64
	p50, p95, p99, mean, memMB float64
}

func main() {
	corpus, queries := realCorpus(N, Q)
	fmt.Printf("corpus N=%d dim=%d, %d queries, k=%d, M=%d, EfSearch=%d (real Model2Vec embeddings)\n\n",
		len(corpus), len(corpus[0]), len(queries), K, M, EfSearch)
	truth := groundTruth(corpus, queries, K)

	idxs := []idx{&aikitFlat{}, &aikitHNSW{}, &aikitFlatI8{}, &coderHNSW{}, &chromemDB{}}
	results := make([]result, 0, len(idxs))
	for _, ix := range idxs {
		r := measure(ix, corpus, queries, truth)
		results = append(results, r)
		fmt.Printf("  %-22s build %7.1fms  recall@%d %.4f  p50 %.3fms  p95 %.3fms  mem ~%.1fMB\n",
			r.name, r.buildMs, K, r.recall, r.p50, r.p95, r.memMB)
	}
	fmt.Print("\n", table(results))
}

func measure(ix idx, corpus, queries [][]float32, truth [][]int) result {
	runtime.GC()
	runtime.GC()
	var m0 runtime.MemStats
	runtime.ReadMemStats(&m0)

	t0 := time.Now()
	ix.build(corpus)
	buildMs := float64(time.Since(t0).Microseconds()) / 1000

	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)
	mem := float64(int64(m1.HeapAlloc)-int64(m0.HeapAlloc)) / 1e6
	if mem < 0 {
		mem = 0
	}

	lat := make([]float64, len(queries))
	var rsum float64
	for i, q := range queries {
		qt := time.Now()
		got := ix.query(q, K)
		lat[i] = float64(time.Since(qt).Microseconds()) / 1000
		rsum += recallAt(got, truth[i])
	}
	sort.Float64s(lat)
	return result{ix.name(), buildMs, rsum / float64(len(queries)), pctl(lat, 50), pctl(lat, 95), pctl(lat, 99), meanf(lat), mem}
}

// --- library wrappers ---

type aikitFlat struct{ f *ann.Flat }

func (*aikitFlat) name() string                     { return "aikit Flat (exact)" }
func (a *aikitFlat) build(c [][]float32)            { a.f = ann.New(c) }
func (a *aikitFlat) query(q []float32, k int) []int { return hitIDs(a.f.Query(q, k)) }

type aikitHNSW struct{ h *ann.HNSW }

func (*aikitHNSW) name() string { return "aikit HNSW" }
func (a *aikitHNSW) build(c [][]float32) {
	a.h = ann.BuildHNSW(c, ann.Config{M: M, EfConstruction: 200, EfSearch: EfSearch, Seed: Seed})
}
func (a *aikitHNSW) query(q []float32, k int) []int { return hitIDs(a.h.Query(q, k)) }

type aikitFlatI8 struct{ f *ann.FlatI8 }

func (*aikitFlatI8) name() string                     { return "aikit FlatI8 (int8)" }
func (a *aikitFlatI8) build(c [][]float32)            { a.f = ann.NewFlatI8(c) }
func (a *aikitFlatI8) query(q []float32, k int) []int { return hitIDs(a.f.Query(q, k)) }

func hitIDs(hits []ann.Hit) []int {
	ids := make([]int, len(hits))
	for i, h := range hits {
		ids[i] = h.Index
	}
	return ids
}

type coderHNSW struct{ g *cohnsw.Graph[int] }

func (*coderHNSW) name() string { return "coder/hnsw" }
func (a *coderHNSW) build(c [][]float32) {
	g := cohnsw.NewGraph[int]()
	g.M, g.EfSearch, g.Distance = M, EfSearch, cohnsw.CosineDistance
	// Leave g.Rng at the default (non-seeded): coder's docs warn a deterministic
	// RNG can produce degenerate graphs. Mean recall over many queries is stable.
	nodes := make([]cohnsw.Node[int], len(c))
	for i, v := range c {
		nodes[i] = cohnsw.MakeNode(i, v)
	}
	g.Add(nodes...)
	a.g = g
}
func (a *coderHNSW) query(q []float32, k int) []int {
	res := a.g.Search(q, k)
	ids := make([]int, len(res))
	for i, n := range res {
		ids[i] = n.Key
	}
	return ids
}

type chromemDB struct{ c *chromem.Collection }

func (*chromemDB) name() string { return "chromem-go (exact)" }
func (a *chromemDB) build(c [][]float32) {
	db := chromem.NewDB()
	noEmbed := func(context.Context, string) ([]float32, error) {
		return nil, errors.New("embeddings supplied directly")
	}
	col, err := db.CreateCollection("bench", nil, noEmbed)
	if err != nil {
		panic(err)
	}
	docs := make([]chromem.Document, len(c))
	for i, v := range c {
		docs[i] = chromem.Document{ID: strconv.Itoa(i), Content: strconv.Itoa(i), Embedding: v}
	}
	if err := col.AddDocuments(context.Background(), docs, runtime.NumCPU()); err != nil {
		panic(err)
	}
	a.c = col
}
func (a *chromemDB) query(q []float32, k int) []int {
	res, err := a.c.QueryEmbedding(context.Background(), q, k, nil, nil)
	if err != nil {
		panic(err)
	}
	ids := make([]int, len(res))
	for i, r := range res {
		ids[i], _ = strconv.Atoi(r.ID)
	}
	return ids
}

// --- corpus / metrics ---

// realCorpus embeds nCorpus + nQuery deterministically-generated code-ish phrases
// with Model2Vec (potion-code-16M, the repo's testdata model). Real embeddings
// give a natural neighbor structure with stable, well-separated top-k — unlike
// synthetic vectors, where high-dim distance concentration (or near-duplicate
// clusters) makes recall@k meaningless. Queries are held-out phrases (not in the
// corpus). Vectors are L2-normalized by Model2Vec. Requires the testdata model.
func realCorpus(nCorpus, nQuery int) (corpus, queries [][]float32) {
	m, err := embed.Load(modelDir)
	if err != nil {
		fmt.Printf("FATAL: load %s: %v\n\nThe benchmark needs the Model2Vec model in testdata/ — fetch it per the\nrepo README (huggingface-cli download minishlab/potion-code-16M ...).\n", modelDir, err)
		os.Exit(1)
	}
	phrases := genPhrases(nCorpus + nQuery)
	all := make([][]float32, len(phrases))
	for i, p := range phrases {
		all[i] = m.Encode(p)
	}
	return all[:nCorpus], all[nCorpus : nCorpus+nQuery]
}

// genPhrases builds a deterministic, shuffled set of distinct code-ish phrases
// (verb × qualifier × noun) — enough variety that Model2Vec places them in a
// realistic, clustered embedding space.
func genPhrases(n int) []string {
	verbs := []string{"open", "close", "read", "write", "parse", "encode", "decode", "sort", "search", "merge", "filter", "append", "delete", "create", "update", "connect", "listen", "send", "receive", "compress", "hash", "validate", "render", "compile", "execute", "allocate", "release", "lock", "flush", "scan"}
	quals := []string{"the", "a", "an empty", "a large", "the binary", "the temporary", "two", "all", "each", "the cached", "a shared", "the pending"}
	nouns := []string{"file", "socket", "buffer", "json document", "array", "string", "hash map", "queue", "stack", "tree", "graph", "request", "response", "header", "token", "stream", "packet", "index", "cache", "database", "record", "field", "column", "matrix", "vector", "config", "session", "thread", "channel", "log entry"}
	all := make([]string, 0, len(verbs)*len(quals)*len(nouns))
	for _, v := range verbs {
		for _, q := range quals {
			for _, nn := range nouns {
				all = append(all, v+" "+q+" "+nn)
			}
		}
	}
	rng := rand.New(rand.NewSource(Seed))
	rng.Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })
	if n > len(all) {
		n = len(all)
	}
	return all[:n]
}

// groundTruth is the exact top-k by cosine (dot, since unit-norm) for each query.
func groundTruth(corpus, queries [][]float32, k int) [][]int {
	truth := make([][]int, len(queries))
	type sc struct {
		id  int
		dot float64
	}
	for qi, q := range queries {
		scores := make([]sc, len(corpus))
		for i, v := range corpus {
			var d float64
			for j := range q {
				d += float64(q[j]) * float64(v[j])
			}
			scores[i] = sc{i, d}
		}
		sort.Slice(scores, func(a, b int) bool { return scores[a].dot > scores[b].dot })
		ids := make([]int, k)
		for i := range k {
			ids[i] = scores[i].id
		}
		truth[qi] = ids
	}
	return truth
}

func recallAt(got, truth []int) float64 {
	set := make(map[int]bool, len(truth))
	for _, id := range truth {
		set[id] = true
	}
	hit := 0
	for _, id := range got {
		if set[id] {
			hit++
		}
	}
	return float64(hit) / float64(len(truth))
}

func pctl(sorted []float64, p int) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := p * len(sorted) / 100
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return sorted[i]
}

func meanf(xs []float64) float64 {
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func table(rs []result) string {
	var s strings.Builder
	s.WriteString("| index | recall@" + strconv.Itoa(K) + " | build | p50 | p95 | p99 | mem |\n")
	s.WriteString("|---|---|---|---|---|---|---|\n")
	for _, r := range rs {
		s.WriteString(fmt.Sprintf("| %s | %.4f | %.0f ms | %.3f ms | %.3f ms | %.3f ms | ~%.0f MB |\n",
			r.name, r.recall, r.buildMs, r.p50, r.p95, r.p99, r.memMB))
	}
	return s.String()
}
