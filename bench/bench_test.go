package bench

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/embed"
)

// realCorpus loads the Model2Vec checkpoint and embeds a clustered corpus of
// code-ish strings (shared token pools → near-ties), plus held-out queries.
func realCorpus(t *testing.T, modelDir string) (m *embed.StaticModel, vecs, queries [][]float32) {
	t.Helper()
	m, err := embed.Load(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	verbs := []string{"get", "set", "make", "parse", "read", "write", "open", "close", "find", "build", "scan", "load", "save", "merge", "sort"}
	nouns := []string{"User", "Index", "Buffer", "Token", "Vector", "Config", "Result", "Node", "Query", "Cache", "Graph", "Record"}
	types := []string{"int", "string", "[]byte", "error", "bool", "float64", "[]int"}
	for _, v := range verbs {
		for _, n := range nouns {
			for _, ty := range types {
				vecs = append(vecs, m.Encode(fmt.Sprintf("func %s%s(in %s) (%s, error)", v, n, ty, ty)))
			}
		}
	}
	for i := range 40 {
		queries = append(queries, m.Encode(fmt.Sprintf("%s the %s %s value", verbs[i%len(verbs)], nouns[i%len(nouns)], types[i%len(types)])))
	}
	return m, vecs, queries
}

func randUnitSet(seed uint64, n, d int) [][]float32 {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, d)
		var ss float64
		for j := range v {
			x := rng.NormFloat64()
			v[j] = float32(x)
			ss += x * x
		}
		inv := float32(1 / math.Sqrt(ss))
		for j := range v {
			v[j] *= inv
		}
		out[i] = v
	}
	return out
}

// efCurve logs HNSW recall@k as a function of EfSearch against the exact Flat
// top-k. With the default Alg-4 diversity heuristic the graph is well-connected,
// so recall is high even at low ef; opting into Config.SimpleNeighbors (Alg-3)
// makes this curve cap low on clustered data regardless of ef — the §4.1 finding.
func efCurve(t *testing.T, corpus, queries [][]float32, k int) {
	t.Helper()
	f := ann.New(corpus)
	h := ann.BuildHNSW(corpus, ann.Config{M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	truth := make([]map[int]bool, len(queries))
	for i, q := range queries {
		truth[i] = make(map[int]bool, k)
		for _, hit := range f.Query(q, k) {
			truth[i][hit.Index] = true
		}
	}
	for _, ef := range []int{64, 128, 256, 512} {
		var rec float64
		for i, q := range queries {
			got := 0
			for _, hit := range h.QueryEf(q, k, ef) {
				if truth[i][hit.Index] {
					got++
				}
			}
			rec += float64(got) / float64(k)
		}
		t.Logf("  HNSW recall@%d at ef=%-4d : %.4f", k, ef, rec/float64(len(queries)))
	}
}

// TestHarness is the reproducible scale harness (synthetic unit vectors,
// deterministic): the per-index latency/memory table plus the HNSW recall-vs-ef
// curve. The FlatI8 quantization recall is a machine-independent regression gate;
// HNSW recall is ef-dependent (see efCurve) so it's gated only loosely here.
func TestHarness(t *testing.T) {
	const n, d, k = 2000, 256, 10
	corpus := randUnitSet(1, n, d)
	queries := randUnitSet(2, 200, d)

	res := Run(corpus, queries, k, ann.Config{M: 16, EfConstruction: 200, EfSearch: 256, Seed: 1})
	t.Logf("synthetic N=%d, d=%d, 200 queries (HNSW ef=256):\n%s", n, d, Table(res))
	efCurve(t, corpus, queries, k)

	for _, r := range res {
		switch r.Name {
		case "FlatI8":
			if r.Recall < 0.90 {
				t.Errorf("FlatI8 recall@%d = %.4f, want ≥ 0.90 (quantization regression)", k, r.Recall)
			}
		case "HNSW":
			if r.Recall < 0.85 {
				t.Errorf("HNSW recall@%d at ef=256 = %.4f, below sanity floor — graph regression?", k, r.Recall)
			}
		}
	}
}

// TestHarnessReal is the headline table on REAL Model2Vec embeddings (clustered,
// where HNSW does best). Skips without the per-machine model.
func TestHarnessReal(t *testing.T) {
	const modelDir = "../testdata/model"
	if _, err := os.Stat(filepath.Join(modelDir, "model.safetensors")); err != nil {
		t.Skipf("no model at %s — see testdata/README.md", modelDir)
	}
	m, vecs, queries := realCorpus(t, modelDir)
	if len(vecs) == 0 {
		t.Skip("empty corpus")
	}
	res := Run(vecs, queries, 10, ann.Config{M: 16, EfConstruction: 200, EfSearch: 256, Seed: 1})
	t.Logf("real Model2Vec: %d docs, dim %d, %d queries (HNSW ef=256):\n%s", len(vecs), m.Dim(), len(queries), Table(res))
	efCurve(t, vecs, queries, 10)

	for _, r := range res {
		switch r.Name {
		case "HNSW":
			// §4.1 RESOLVED: the Alg-4 diversity heuristic (now the default) fixed
			// the clustered-data recall this harness first exposed — it was ~0.68
			// with the old Alg-3 selection, now ~1.0. This is the recall TARGET.
			// (Set Config.SimpleNeighbors to opt back to Alg-3 and watch it drop.)
			if r.Recall < 0.95 {
				t.Errorf("HNSW recall@10 on real embeddings = %.4f, want ≥ 0.95 (Alg-4 default)", r.Recall)
			}
		case "FlatI8":
			if r.Recall < 0.95 {
				t.Errorf("FlatI8 recall@10 on real embeddings = %.4f, want ≥ 0.95", r.Recall)
			}
		}
	}
}
