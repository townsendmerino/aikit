package embed

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"
)

// Item 8 (docs/internal/task-memory-campaign.md): Model2Vec mapped by default.

// mappedTestModelDir is a local Model2Vec snapshot: testdata/model, else the embedded-corpus example's model.
func mappedTestModelDir(t *testing.T) string {
	for _, d := range []string{filepath.Join("..", "testdata", "model"), filepath.Join("..", "examples", "embedded-corpus", "assets", "model")} {
		if _, err := os.Stat(filepath.Join(d, "model.safetensors")); err == nil {
			return d
		}
	}
	t.Skip("no local Model2Vec snapshot (testdata/model, or the embedded-corpus example's go generate)")
	return ""
}

// mappedTestTexts is the embedded-corpus example's chunks when present, plus edge strings.
func mappedTestTexts(t *testing.T) []string {
	texts := []string{"", " ", "hello world", "func main() { fmt.Println(\"hi\") }", "ünïcödé 日本語 🙂", "a"}
	raw, err := os.ReadFile(filepath.Join("..", "examples", "embedded-corpus", "assets", "corpus.json"))
	if err != nil {
		return texts
	}
	var chunks []struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &chunks); err != nil {
		t.Fatal(err)
	}
	for _, c := range chunks {
		texts = append(texts, c.Text)
	}
	return texts
}

// Gate (a): an os.DirFS load is mapped, an fstest.MapFS load and LoadFromFSHeap are not, and every Encode and
// EncodeBatch output is bit-identical across the three.
func TestLoadFromFS_mappedByDefaultMatchesHeap(t *testing.T) {
	dir := mappedTestModelDir(t)
	mapped, err := LoadFromFS(os.DirFS(dir), ".")
	if err != nil {
		t.Fatal(err)
	}
	defer mapped.Close()
	heap, err := LoadFromFSHeap(os.DirFS(dir), ".")
	if err != nil {
		t.Fatal(err)
	}
	read := func(name string) []byte {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	mfs, err := LoadFromFS(fstest.MapFS{
		"tokenizer.json":    {Data: read("tokenizer.json")},
		"config.json":       {Data: read("config.json")},
		"model.safetensors": {Data: read("model.safetensors")},
	}, ".")
	if err != nil {
		t.Fatal(err)
	}
	if len(mapped.st.mmapped) == 0 {
		t.Error("LoadFromFS(os.DirFS) did not map the weights")
	}
	if len(heap.st.mmapped) != 0 || len(mfs.st.mmapped) != 0 {
		t.Error("LoadFromFSHeap or a MapFS load mapped the weights")
	}
	texts := mappedTestTexts(t)
	want := heap.EncodeBatch(texts, 4)
	for name, m := range map[string]*StaticModel{"mapped": mapped, "MapFS": mfs} {
		got := m.EncodeBatch(texts, 4)
		for i := range texts {
			one := m.Encode(texts[i])
			for j := range want[i] {
				if math.Float32bits(got[i][j]) != math.Float32bits(want[i][j]) || math.Float32bits(one[j]) != math.Float32bits(want[i][j]) {
					t.Fatalf("%s, text %d, component %d: differs from the heap load", name, i, j)
				}
			}
		}
	}
	t.Logf("%d texts bit-identical across mapped, MapFS and heap loads", len(texts))
}

// Close releases a mapped model's mapping; use after Close panics; Close is idempotent and fine on a heap model.
func TestStaticModel_close(t *testing.T) {
	dir := mappedTestModelDir(t)
	m, err := LoadFromFS(os.DirFS(dir), ".")
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("Encode after Close did not panic")
			}
		}()
		m.Encode("hello")
	}()
	h, err := LoadFromFSHeap(os.DirFS(dir), ".")
	if err != nil {
		t.Fatal(err)
	}
	if err := h.Close(); err != nil {
		t.Fatalf("Close on a heap model: %v", err)
	}
}
