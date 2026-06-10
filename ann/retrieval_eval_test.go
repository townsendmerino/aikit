package ann_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/embed"
)

// §4.2 end-to-end retrieval-quality regression. The golden tests pin embedding
// numerics; this pins what the RETRIEVER does with them. A small fixed corpus of
// code-ish strings in five clear topics is embedded once (TestGenRetrievalFixture,
// model-gated) and frozen into testdata/retrieval_eval.json with a hand-curated
// relevance set (same-topic docs). TestRetrievalRecall then runs model-free in CI:
// it asserts Flat/HNSW/FlatI8 recall@10 against that frozen relevance, so an index
// or scoring change that degrades retrieval quality is caught.

const evalFixturePath = "../testdata/retrieval_eval.json"

type evalDoc struct {
	Text string    `json:"text"`
	Cat  string    `json:"cat"`
	Emb  []float32 `json:"emb"`
}
type evalQuery struct {
	Text     string    `json:"text"`
	Cat      string    `json:"cat"`
	Relevant []int     `json:"relevant"`
	Emb      []float32 `json:"emb"`
}
type evalFixture struct {
	Dim     int         `json:"dim"`
	Docs    []evalDoc   `json:"docs"`
	Queries []evalQuery `json:"queries"`
}

var evalTopics = []struct {
	cat   string
	query string
	docs  []string
}{
	{"file", "how to work with files on disk", []string{"open a file", "read file contents", "write bytes to a file", "close the file handle", "create a temporary file", "delete a file from disk", "check if a file exists", "read a file line by line", "append text to a file", "flush the file buffer"}},
	{"sort", "ordering and searching arrays", []string{"sort a slice ascending", "binary search in an array", "quicksort implementation", "find the maximum element", "merge two sorted lists", "reverse a slice in place", "sort items by key", "stable sort algorithm", "partition an array", "find the median element"}},
	{"http", "making web requests over http", []string{"send an HTTP GET request", "parse a URL string", "set request headers", "handle an HTTP response", "start a web server", "route HTTP requests", "read the response body", "make a POST request with JSON", "set a client timeout", "follow HTTP redirects"}},
	{"json", "serializing data as json", []string{"marshal a struct to JSON", "unmarshal JSON into a map", "decode a JSON stream", "encode nested objects", "parse a JSON array", "pretty-print JSON output", "handle unknown JSON fields", "validate against a JSON schema", "convert JSON to a struct", "stream a large JSON file"}},
	{"math", "basic arithmetic and math operations", []string{"add two numbers together", "compute the square root", "matrix multiplication", "calculate the average value", "generate a random number", "round to two decimal places", "compute a factorial", "greatest common divisor", "exponentiation by squaring", "compute the dot product"}},
}

// TestGenRetrievalFixture (re)builds testdata/retrieval_eval.json from the model.
// Run manually when the corpus changes: go test ./ann/ -run TestGenRetrievalFixture
func TestGenRetrievalFixture(t *testing.T) {
	const modelDir = "../testdata/model"
	if _, err := os.Stat(filepath.Join(modelDir, "model.safetensors")); err != nil {
		t.Skipf("no model at %s — see testdata/README.md", modelDir)
	}
	m, err := embed.Load(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	fx := evalFixture{Dim: m.Dim()}
	for _, topic := range evalTopics {
		rel := make([]int, 0, len(topic.docs))
		for _, d := range topic.docs {
			rel = append(rel, len(fx.Docs))
			fx.Docs = append(fx.Docs, evalDoc{Text: d, Cat: topic.cat, Emb: m.Encode(d)})
		}
		fx.Queries = append(fx.Queries, evalQuery{Text: topic.query, Cat: topic.cat, Relevant: rel, Emb: m.Encode(topic.query)})
	}
	b, err := json.Marshal(fx)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(evalFixturePath, b, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s: %d docs, %d queries, dim %d (%d KB)", evalFixturePath, len(fx.Docs), len(fx.Queries), fx.Dim, len(b)/1024)
}

func loadEvalFixture(t *testing.T) evalFixture {
	t.Helper()
	b, err := os.ReadFile(evalFixturePath)
	if err != nil {
		t.Skipf("no fixture at %s — run TestGenRetrievalFixture", evalFixturePath)
	}
	var fx evalFixture
	if err := json.Unmarshal(b, &fx); err != nil {
		t.Fatal(err)
	}
	return fx
}

// recallVsRelevance returns mean recall@k of the index's top-k against each
// query's frozen relevant set.
func recallVsRelevance(query func([]float32, int) []ann.Hit, fx evalFixture, k int) float64 {
	var sum float64
	for _, q := range fx.Queries {
		rel := make(map[int]bool, len(q.Relevant))
		for _, id := range q.Relevant {
			rel[id] = true
		}
		got := 0
		for _, h := range query(q.Emb, k) {
			if rel[h.Index] {
				got++
			}
		}
		denom := k
		if len(q.Relevant) < k {
			denom = len(q.Relevant)
		}
		sum += float64(got) / float64(denom)
	}
	return sum / float64(len(fx.Queries))
}

// TestRetrievalRecall is the model-free regression gate: against the frozen
// relevance set, Flat (exact) pins the embedding+retrieval quality baseline, and
// HNSW / FlatI8 must stay within tolerance of it.
func TestRetrievalRecall(t *testing.T) {
	fx := loadEvalFixture(t)
	docs := make([][]float32, len(fx.Docs))
	for i, d := range fx.Docs {
		docs[i] = d.Emb
	}
	const k = 10

	flat := recallVsRelevance(ann.New(docs).Query, fx, k)
	hnsw := recallVsRelevance(ann.BuildHNSW(docs, ann.Config{M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1}).Query, fx, k)
	fi8 := recallVsRelevance(ann.NewFlatI8(docs).Query, fx, k)
	t.Logf("retrieval recall@%d vs frozen relevance — Flat %.4f, HNSW %.4f, FlatI8 %.4f", k, flat, hnsw, fi8)

	// Flat pins embedding + exact-scan quality (frozen embeddings → deterministic).
	if flat < 0.82 {
		t.Errorf("Flat recall@%d = %.4f, want ≥ 0.82 — embeddings or exact scan regressed", k, flat)
	}
	// Approximate / quantized indexes must stay within tolerance of Flat.
	if flat-hnsw > 0.03 {
		t.Errorf("HNSW recall@%d = %.4f, more than 0.03 below Flat (%.4f) — graph regression", k, hnsw, flat)
	}
	if flat-fi8 > 0.05 {
		t.Errorf("FlatI8 recall@%d = %.4f, more than 0.05 below Flat (%.4f) — quantization regression", k, fi8, flat)
	}
}
