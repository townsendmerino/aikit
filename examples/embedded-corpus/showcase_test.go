package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/ann"
	m2v "github.com/townsendmerino/aikit/embed"
)

// TestShowcase_query loads the embedded assets and checks a query returns sensible
// hits. Skips when only the placeholder model is embedded (run `go generate` then
// rebuild to exercise it).
func TestShowcase_query(t *testing.T) {
	model, err := m2v.LoadFromFS(modelFS, "assets/model")
	if err != nil {
		t.Skipf("no real model embedded (run `go generate`): %v", err)
	}
	index, err := ann.LoadFlatI8(indexBlob)
	if err != nil {
		t.Fatal(err)
	}
	var corpus []Chunk
	if err := json.Unmarshal(corpusJSON, &corpus); err != nil {
		t.Fatal(err)
	}
	if len(corpus) == 0 {
		t.Fatal("empty corpus")
	}
	// A json query should surface encoding/json in the dense top-5.
	hits := index.Query(model.Encode("parse json into a struct"), 5)
	if len(hits) == 0 {
		t.Fatal("no dense hits")
	}
	found := false
	for _, h := range hits {
		if h.Index < 0 || h.Index >= len(corpus) {
			t.Fatalf("hit index %d out of range [0,%d)", h.Index, len(corpus))
		}
		if strings.Contains(corpus[h.Index].Ref, "json") {
			found = true
		}
	}
	if !found {
		t.Error("expected an encoding/json hit in the dense top-5")
	}
}
