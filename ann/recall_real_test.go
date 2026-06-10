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

// TestFlat_recallReal_Model2Vec is the spec's "recall@k on a REAL embedding set"
// check: it embeds a corpus of varied code-like texts with the actual Model2Vec
// model, then confirms Flat.Query (float32 SIMD dot) returns the same top-k as an
// exact float64 brute-force scan. Real embeddings cluster — far more near-ties
// than random unit vectors — so this is the harder test that the f32 swap leaves
// recall unchanged (only sub-ULP boundary ties may reorder). Skips without the
// per-machine model checkpoint (so CI stays green); run with testdata/model.
func TestFlat_recallReal_Model2Vec(t *testing.T) {
	const modelDir = "../testdata/model"
	if _, err := os.Stat(filepath.Join(modelDir, "model.safetensors")); err != nil {
		t.Skipf("no model at %s — see testdata/README.md", modelDir)
	}
	m, err := embed.Load(modelDir)
	if err != nil {
		t.Fatal(err)
	}

	// A real, clustered corpus: varied code-ish strings over shared token pools,
	// so many embeddings land close together (the near-tie stress case).
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
	vecs := make([][]float32, len(texts))
	for i, txt := range texts {
		vecs[i] = m.Encode(txt)
	}
	f := ann.New(vecs)
	t.Logf("real corpus: %d Model2Vec embeddings, dim %d", len(vecs), m.Dim())

	const k, tieEps = 10, 1e-5
	queries := []string{
		"function that parses a user token",
		"open and read a config buffer",
		"build an index over result vectors",
		"close the query cache node",
		"write bytes to an output buffer",
	}
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
