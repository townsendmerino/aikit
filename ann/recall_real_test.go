package ann_test

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/embed"
)

// realCorpus loads the Model2Vec checkpoint and embeds a clustered corpus of
// code-ish strings over shared token pools (so many embeddings land close
// together — the near-tie stress case), plus a few held-out queries. Skips
// without the per-machine model so CI stays green; run with testdata/model.
func realCorpus(t *testing.T) (m *embed.StaticModel, vecs [][]float32, queries []string) {
	t.Helper()
	const modelDir = "../testdata/model"
	if _, err := os.Stat(filepath.Join(modelDir, "model.safetensors")); err != nil {
		t.Skipf("no model at %s — see testdata/README.md", modelDir)
	}
	m, err := embed.Load(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	verbs := []string{"get", "set", "make", "parse", "read", "write", "open", "close", "find", "build"}
	nouns := []string{"User", "Index", "Buffer", "Token", "Vector", "Config", "Result", "Node", "Query", "Cache"}
	types := []string{"int", "string", "[]byte", "error", "bool", "float64"}
	var texts []string
	for _, v := range verbs {
		for _, n := range nouns {
			for _, ty := range types {
				texts = append(texts, fmt.Sprintf("func %s%s(in %s) (%s, error)", v, n, ty, ty))
			}
		}
	}
	vecs = make([][]float32, len(texts))
	for i, txt := range texts {
		vecs[i] = m.Encode(txt)
	}
	queries = []string{
		"function that parses a user token",
		"open and read a config buffer",
		"build an index over result vectors",
		"close the query cache node",
		"write bytes to an output buffer",
	}
	return m, vecs, queries
}

// TestFlat_recallReal_Model2Vec is the spec's "recall@k on a REAL embedding set"
// check for the float32 SIMD-dot swap: Flat.Query (float32 SIMD) must return the
// same top-k as an exact float64 scan, with only sub-ULP boundary ties reordering.
func TestFlat_recallReal_Model2Vec(t *testing.T) {
	m, vecs, queries := realCorpus(t)
	f := ann.New(vecs)
	t.Logf("real corpus: %d Model2Vec embeddings, dim %d", len(vecs), m.Dim())

	const k, tieEps = 10, 1e-5
	flips := 0
	for qi, qtext := range queries {
		q := m.Encode(qtext)
		got := f.Query(q, k)

		type sc struct {
			d int
			s float64
		}
		ref := make([]sc, len(vecs))
		for d, v := range vecs {
			var dot float64
			for j := range v {
				dot += float64(v[j]) * float64(q[j])
			}
			ref[d] = sc{d, dot}
		}
		sort.Slice(ref, func(i, j int) bool {
			if ref[i].s != ref[j].s {
				return ref[i].s > ref[j].s
			}
			return ref[i].d < ref[j].d
		})

		inNew := make(map[int]bool, k)
		for _, h := range got {
			inNew[h.Index] = true
		}
		kth := got[len(got)-1].Score
		for r := 0; r < k; r++ {
			if inNew[ref[r].d] {
				continue
			}
			if ref[r].s-kth > tieEps {
				t.Errorf("query %q rank %d doc %d (f64 %.8f) absent from f32 top-k (kth %.8f) — recall loss",
					queries[qi], r, ref[r].d, ref[r].s, kth)
			}
			flips++
		}
	}
	t.Logf("%d queries × top-%d on real embeddings: %d boundary tie-flips (recall unchanged)", len(queries), k, flips)
}

// TestFlatI8_recallReal_Model2Vec is the §2.4 quantized-storage recall check on
// REAL embeddings: the int8 FlatI8 index should keep nearly all of the exact
// float32 Flat top-k. Real embeddings cluster, so this is the realistic recall
// (the embedded/RAM-constrained niche), not just random unit vectors.
func TestFlatI8_recallReal_Model2Vec(t *testing.T) {
	m, vecs, queries := realCorpus(t)
	f32 := ann.New(vecs)
	i8 := ann.NewFlatI8(vecs)

	const k = 10
	var sum float64
	for _, qtext := range queries {
		q := m.Encode(qtext)
		truth := f32.Query(q, k)
		got := i8.Query(q, k)

		set := make(map[int]bool, k)
		for _, h := range truth {
			set[h.Index] = true
		}
		hit := 0
		for _, h := range got {
			if set[h.Index] {
				hit++
			}
		}
		sum += float64(hit) / float64(k)
	}
	mean := sum / float64(len(queries))
	t.Logf("FlatI8 recall@%d vs float32 Flat on %d real embeddings: mean %.4f", k, len(vecs), mean)
	if mean < 0.90 {
		t.Errorf("real-embedding recall@%d = %.4f, want ≥ 0.90", k, mean)
	}
}
