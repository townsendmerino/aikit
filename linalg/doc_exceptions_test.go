package linalg

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// TestDoc_rawSliceExceptionsListed holds the package doc's "Raw-slice exceptions" to the source: every exported
// quantized matmul that takes raw slices (MatmulBTQ*, MatmulBTW*) is listed there, and every name listed is one. A new
// raw-slice entry point fails here until the doc says who needs it and what WeightMat form it waits for.
func TestDoc_rawSliceExceptionsListed(t *testing.T) {
	fset := token.NewFileSet()
	names, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	raw := regexp.MustCompile(`^MatmulBT(Q|W)\w*$`)
	var exported []string
	for _, n := range names {
		if strings.HasSuffix(n, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, n, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv == nil && raw.MatchString(fd.Name.Name) && !slices.Contains(exported, fd.Name.Name) {
				exported = append(exported, fd.Name.Name)
			}
		}
	}
	doc, err := os.ReadFile("doc.go")
	if err != nil {
		t.Fatal(err)
	}
	sec := string(doc)
	i := strings.Index(sec, "// # Raw-slice exceptions")
	if i < 0 {
		t.Fatal(`doc.go has no "# Raw-slice exceptions" section`)
	}
	sec = sec[i:]
	if j := strings.Index(sec[1:], "// # "); j >= 0 {
		sec = sec[:j+1]
	}
	var listed []string
	for _, m := range regexp.MustCompile(`\[(MatmulBT(?:Q|W)\w*)\]`).FindAllStringSubmatch(sec, -1) {
		listed = append(listed, m[1])
	}
	slices.Sort(exported)
	slices.Sort(listed)
	listed = slices.Compact(listed)
	if !slices.Equal(exported, listed) {
		t.Errorf("raw-slice quantized matmuls exported: %v; listed in the package doc's exceptions: %v", exported, listed)
	}
}
