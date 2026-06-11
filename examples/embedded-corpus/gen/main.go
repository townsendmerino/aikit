// Command gen builds the committed assets the showcase //go:embeds. It extracts a
// doc-oriented corpus — Go stdlib package docs (via `go doc -all`), aikit package
// docs, and aikit's markdown — embeds each chunk with the Model2Vec model in
// ../assets/model, quantizes the matrix to an int8 FlatI8 index, and writes
// assets/index.bin (the prebuilt index) + assets/corpus.json (the chunk texts).
//
// Run via `go generate` from the example root (it also fetches the model first).
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/embed"
)

// Chunk is one searchable unit; corpus.json is a []Chunk, parallel to the index rows.
type Chunk struct {
	Source string `json:"source"` // "stdlib" | "aikit" | "aikit-docs"
	Ref    string `json:"ref"`    // package import path or file name
	Text   string `json:"text"`
}

// A curated bread-and-butter slice of the standard library (full stdlib would bloat
// the committed assets; this is enough to make the demo answer real Go questions).
var stdlibPkgs = []string{
	"fmt", "errors", "io", "io/fs", "os", "bufio", "bytes", "strings", "strconv",
	"sort", "slices", "maps", "context", "sync", "sync/atomic", "time", "regexp",
	"encoding/json", "encoding/binary", "net/http", "net/url", "math", "math/rand",
	"unicode/utf8", "log", "flag", "path/filepath", "crypto/sha256", "container/heap",
}

var aikitPkgs = []string{
	"ann", "bm25", "embed", "encoder", "fuse", "sparse", "topk", "linalg", "chunk",
}

var aikitDocs = []string{
	"../../README.md", "../../docs/architecture.md",
}

const (
	minChunk = 40
	maxChunk = 1500
)

func main() {
	var chunks []Chunk
	for _, p := range stdlibPkgs {
		chunks = append(chunks, goDocChunks(p, "stdlib", p)...)
	}
	for _, p := range aikitPkgs {
		chunks = append(chunks, goDocChunks("github.com/townsendmerino/aikit/"+p, "aikit", p)...)
	}
	for _, f := range aikitDocs {
		chunks = append(chunks, markdownChunks(f)...)
	}
	if len(chunks) == 0 {
		fatal("no chunks collected")
	}
	fmt.Fprintf(os.Stderr, "gen: collected %d chunks\n", len(chunks))

	model, err := embed.Load("assets/model")
	if err != nil {
		fatal("load model (run `go generate` to fetch it): %v", err)
	}
	vecs := make([][]float32, len(chunks))
	for i, c := range chunks {
		vecs[i] = model.Encode(c.Text)
	}
	idx := ann.NewFlatI8(vecs)
	blob, err := idx.MarshalBinary()
	if err != nil {
		fatal("marshal index: %v", err)
	}
	if err := os.WriteFile("assets/index.bin", blob, 0o644); err != nil {
		fatal("write index: %v", err)
	}
	cj, err := json.Marshal(chunks)
	if err != nil {
		fatal("marshal corpus: %v", err)
	}
	if err := os.WriteFile("assets/corpus.json", cj, 0o644); err != nil {
		fatal("write corpus: %v", err)
	}
	fmt.Fprintf(os.Stderr, "gen: wrote assets/index.bin (%d KB, dim %d) + assets/corpus.json (%d chunks)\n",
		len(blob)/1024, len(vecs[0]), len(chunks))
}

// goDocChunks runs `go doc -all pkg` (from the aikit module root, so both stdlib and
// aikit import paths resolve) and splits the output into one chunk per exported
// symbol, plus a leading package-overview chunk.
func goDocChunks(pkg, source, ref string) []Chunk {
	cmd := exec.Command("go", "doc", "-all", pkg)
	cmd.Dir = "../.." // the aikit module root
	out, err := cmd.Output()
	if err != nil {
		fmt.Fprintf(os.Stderr, "  skip %s: %v\n", pkg, err)
		return nil
	}
	var chunks []Chunk
	var cur []string
	flush := func() {
		text := strings.TrimSpace(strings.Join(cur, "\n"))
		cur = nil
		if len(text) < minChunk {
			return
		}
		if len(text) > maxChunk {
			text = text[:maxChunk]
		}
		chunks = append(chunks, Chunk{Source: source, Ref: ref, Text: text})
	}
	sc := bufio.NewScanner(strings.NewReader(string(out)))
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if symbolStart(line) && len(cur) > 0 {
			flush()
		}
		cur = append(cur, line)
	}
	flush()
	return chunks
}

func symbolStart(line string) bool {
	for _, kw := range []string{"func ", "type ", "const ", "var "} {
		if strings.HasPrefix(line, kw) {
			return true
		}
	}
	return false
}

// markdownChunks splits a markdown file into one chunk per `##` section.
func markdownChunks(path string) []Chunk {
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "  skip %s: %v\n", path, err)
		return nil
	}
	ref := filepath.Base(path)
	var chunks []Chunk
	var cur []string
	flush := func() {
		text := strings.TrimSpace(strings.Join(cur, "\n"))
		cur = nil
		if len(text) < minChunk {
			return
		}
		if len(text) > maxChunk {
			text = text[:maxChunk]
		}
		chunks = append(chunks, Chunk{Source: "aikit-docs", Ref: ref, Text: text})
	}
	for line := range strings.SplitSeq(string(data), "\n") {
		if strings.HasPrefix(line, "## ") && len(cur) > 0 {
			flush()
		}
		cur = append(cur, line)
	}
	flush()
	return chunks
}

func fatal(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "gen: "+format+"\n", a...)
	os.Exit(1)
}
