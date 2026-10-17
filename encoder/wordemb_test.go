package encoder

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Item 9 (docs/internal/task-memory-campaign.md): f16 / int8 word-embedding tables for BERT and GTE. The bars
// were registered before the first run: per fixture and precision, the MIN over texts of the cosine between the
// output embedding and the f32 load's — f16 ≥ 0.9999, int8 ≥ 0.999.

var wordEmbBars = map[WordEmbPrecision]float64{WordEmbF16: 0.9999, WordEmbInt8: 0.999}

// wordEmbFixedTexts is the fixed 20-text set every fixture runs besides its own golden texts.
var wordEmbFixedTexts = []string{
	"how do I parse json in go", "binary search over a sorted slice", "the cat sat on the mat",
	"func main() { fmt.Println(\"hello\") }", "SELECT name FROM users WHERE id = 7", "machine learning for code search",
	"a", "", "Ünïcödé, accents, and 日本語 text", "error: undefined reference to `main'",
	"The quick brown fox jumps over the lazy dog.", "def add(a, b):\n    return a + b", "rare tokens: zygote quixotic",
	"numbers 3.14159 2.71828 1e-9", "class Dog: bark()", "HTTP 404 not found",
	"repeat repeat repeat repeat repeat repeat", "emoji 🙂🚀", "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
	"a long sentence about retrieval augmented generation and dense versus sparse indexes in production systems",
}

// goldenTexts reads the input texts of a golden file: cases[].text, or query and doc for the cross-encoder's.
func goldenTexts(t *testing.T, name string) []string {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", name))
	if err != nil {
		return nil
	}
	var g struct {
		Cases []struct{ Text, Query, Doc string } `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, c := range g.Cases {
		for _, s := range []string{c.Text, c.Query, c.Doc} {
			if s != "" {
				out = append(out, s)
			}
		}
	}
	return out
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 && nb == 0 {
		return 1
	}
	return dot / math.Sqrt(na*nb)
}

type wordEmbModel interface {
	Encode(string) ([]float32, error)
	Close() error
	wordEmbBytes() int
}

// Item 9's gate: the quality bars, per fixture and precision; the f32 option is bit-identical to the plain loader.
// Model-gated (the fixtures are local).
func TestWordEmbPrecision_qualityGate(t *testing.T) {
	type fixture struct {
		dir, golden string
		load        func(string, LoadOptions) (wordEmbModel, error)
	}
	bert := func(d string, o LoadOptions) (wordEmbModel, error) { return LoadBERTWithOptions(d, o) }
	gte := func(d string, o LoadOptions) (wordEmbModel, error) { return LoadGTEWithOptions(d, o) }
	ran := 0
	for _, f := range []fixture{
		{"minilm-model", "minilm_golden.json", bert},
		{"crossencoder-model", "crossencoder_golden.json", bert},
		{"splade-model", "splade_golden.json", bert},
		{"arctic2-m", "arctic_golden.json", gte},
	} {
		dir := filepath.Join("..", "testdata", f.dir)
		if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
			t.Logf("%s: not present, skipped", f.dir)
			continue
		}
		ran++
		texts := append(goldenTexts(t, f.golden), wordEmbFixedTexts...)
		ref, err := f.load(dir, LoadOptions{})
		if err != nil {
			t.Fatalf("%s: %v", f.dir, err)
		}
		want := make([][]float32, len(texts))
		for i, s := range texts {
			if want[i], err = ref.Encode(s); err != nil {
				t.Fatalf("%s %q: %v", f.dir, s, err)
			}
		}
		f32Bytes := ref.wordEmbBytes()
		_ = ref.Close()
		for _, p := range []WordEmbPrecision{WordEmbF16, WordEmbInt8} {
			m, err := f.load(dir, LoadOptions{WordEmb: p})
			if err != nil {
				t.Fatalf("%s %v: %v", f.dir, p, err)
			}
			minCos, sum := 1.0, 0.0
			worst := ""
			for i, s := range texts {
				got, err := m.Encode(s)
				if err != nil {
					t.Fatal(err)
				}
				c := cosine(got, want[i])
				sum += c
				if c < minCos {
					minCos, worst = c, s
				}
			}
			t.Logf("%-18s %-4v table %6.1f MiB (f32 %6.1f): cosine vs f32 over %d texts — min %.7f, mean %.7f (worst %.30q); bar %.4f",
				f.dir, p, float64(m.wordEmbBytes())/(1<<20), float64(f32Bytes)/(1<<20), len(texts), minCos, sum/float64(len(texts)), worst, wordEmbBars[p])
			if minCos < wordEmbBars[p] {
				t.Errorf("%s %v: min cosine %.7f below the registered bar %.4f", f.dir, p, minCos, wordEmbBars[p])
			}
			_ = m.Close()
		}
	}
	if ran == 0 {
		t.Skip("no BERT/GTE fixture present; see scripts/README.md")
	}
}

// The default option is the plain loader, bit for bit (it is the same code path; this pins that it stays so).
func TestLoadBERTWithOptions_defaultIsLoadBERT(t *testing.T) {
	dir := filepath.Join("..", "testdata", "minilm-model")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skip("testdata/minilm-model/ not present")
	}
	a, err := LoadBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := LoadBERTWithOptions(dir, LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	for _, s := range wordEmbFixedTexts {
		x, _ := a.Encode(s)
		y, _ := b.Encode(s)
		for i := range x {
			if math.Float32bits(x[i]) != math.Float32bits(y[i]) {
				t.Fatalf("%q: default options differ from LoadBERT", s)
			}
		}
	}
	if _, err := LoadBERTWithOptions(dir, LoadOptions{WordEmb: 9}); err == nil {
		t.Fatal("an unknown precision loaded")
	}
}

// Model-free: the narrow tables round-trip a synthetic table within each format's own error.
func TestWordTable_rowInto(t *testing.T) {
	const vocab, dim = 50, 33
	f32 := make([]float32, vocab*dim)
	for i := range f32 {
		f32[i] = float32(math.Sin(float64(i)*0.37)) * float32(1+i%7)
	}
	for _, p := range []WordEmbPrecision{WordEmbF16, WordEmbInt8} {
		tab, err := newWordTable(f32, vocab, dim, p)
		if err != nil {
			t.Fatal(err)
		}
		row := make([]float32, dim)
		for id := range vocab {
			tab.rowInto(id, row)
			src := f32[id*dim : (id+1)*dim]
			var maxAbs float64
			for _, v := range src {
				maxAbs = max(maxAbs, math.Abs(float64(v)))
			}
			for j := range row {
				tol := maxAbs / 127 // int8: half a quantization step is the bound; one step is generous
				if p == WordEmbF16 {
					tol = math.Abs(float64(src[j]))*1e-3 + 1e-7
				}
				if d := math.Abs(float64(row[j] - src[j])); d > tol {
					t.Fatalf("%v row %d col %d: %v vs %v", p, id, j, row[j], src[j])
				}
			}
		}
	}
	if _, err := newWordTable(f32, vocab, dim, WordEmbF32); err == nil {
		t.Fatal("newWordTable accepted f32")
	}
}
