// Command beir evaluates aikit's retrieval quality on the BeIR/scifact test slice:
// potion-retrieval-32M embeddings (aikit's pure-Go embed) + exact Flat cosine ANN,
// scored by nDCG@10 — a cross-referenceable standard-benchmark number (SciFact is a
// canonical BEIR task). Prep the data first with scripts/prep_beir.py.
//
//	cd benchmarks && GOWORK=off go run ./beir
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/embed"
)

type dataset struct {
	Dataset string                    `json:"dataset"`
	Corpus  map[string]string         `json:"corpus"`
	Queries map[string]string         `json:"queries"`
	Qrels   map[string]map[string]int `json:"qrels"`
}

func main() {
	raw, err := os.ReadFile("../testdata/beir-scifact/scifact.json")
	if err != nil {
		fmt.Fprintln(os.Stderr, "beir: no data — run scripts/prep_beir.py first:", err)
		os.Exit(1)
	}
	var d dataset
	if err := json.Unmarshal(raw, &d); err != nil {
		fmt.Fprintln(os.Stderr, "beir:", err)
		os.Exit(1)
	}
	model, err := embed.Load("../testdata/retrieval-model")
	if err != nil {
		fmt.Fprintln(os.Stderr, "beir: load potion-retrieval-32M:", err)
		os.Exit(1)
	}

	// Embed the corpus (ordered), build an exact Flat index.
	docIDs := make([]string, 0, len(d.Corpus))
	vecs := make([][]float32, 0, len(d.Corpus))
	for id, text := range d.Corpus {
		docIDs = append(docIDs, id)
		vecs = append(vecs, model.Encode(text))
	}
	idx := ann.New(vecs)

	// Retrieve + score nDCG@10 per query.
	var sum float64
	n := 0
	for qid, qtext := range d.Queries {
		rel := d.Qrels[qid]
		if len(rel) == 0 {
			continue
		}
		hits := idx.Query(model.Encode(qtext), 10)
		ranked := make([]string, len(hits))
		for i, h := range hits {
			ranked[i] = docIDs[h.Index]
		}
		sum += ndcgAt10(ranked, rel)
		n++
	}
	fmt.Printf("BeIR/scifact (test): aikit potion-retrieval-32M + Flat — nDCG@10 = %.4f over %d queries (%d docs)\n",
		sum/float64(n), n, len(d.Corpus))
}

func ndcgAt10(ranked []string, rel map[string]int) float64 {
	dcg := 0.0
	for i, did := range ranked {
		if r := rel[did]; r > 0 {
			dcg += float64(r) / math.Log2(float64(i+2))
		}
	}
	rels := make([]int, 0, len(rel))
	for _, r := range rel {
		if r > 0 {
			rels = append(rels, r)
		}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(rels)))
	idcg := 0.0
	for i := 0; i < len(rels) && i < 10; i++ {
		idcg += float64(rels[i]) / math.Log2(float64(i+2))
	}
	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}
