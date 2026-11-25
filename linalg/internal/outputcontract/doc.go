package outputcontract

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// ContractForm matches the four recognised "Output contract:" lines:
//
//	Output contract: overwrites <out>; do not pre-zero.      (the caller must not zero first)
//	Output contract: zeroes <out> internally; ...            (it clears the span itself, then accumulates)
//	Output contract: accumulates into <out>; caller zeroes.  (reserved: no writer does today)
//	Output contract: rewrites <out> in place ...             (input and output share a buffer)
var ContractForm = regexp.MustCompile(`Output contract: (overwrites .+; do not pre-zero|zeroes .+ internally|accumulates into .+; caller zeroes|rewrites .+ in place)`)

// outputParamNames: a parameter with one of these names is an output buffer.
var outputParamNames = map[string]bool{"dst": true, "sums": true}

// Decl is one function, method or interface method declared in a directory's non-test sources.
type Decl struct {
	File     string // base name
	Key      string // Name, (Recv).Name or Interface.Method
	Doc      string // the doc comment, whitespace-normalised
	Exported bool
	OutParam bool // a parameter is named dst or sums
}

// ParseDecls parses every non-test Go file in dir, whatever its build constraints.
func ParseDecls(t testing.TB, dir string) []Decl {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no sources under %s (%v)", dir, err)
	}
	var out []Decl
	for _, p := range paths {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(token.NewFileSet(), p, nil, parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		norm := func(g *ast.CommentGroup) string {
			if g == nil {
				return ""
			}
			return strings.Join(strings.Fields(g.Text()), " ")
		}
		hasOut := func(ft *ast.FuncType) bool {
			for _, fld := range ft.Params.List {
				for _, n := range fld.Names {
					if outputParamNames[n.Name] {
						return true
					}
				}
			}
			return false
		}
		for _, d := range f.Decls {
			switch d := d.(type) {
			case *ast.FuncDecl:
				key := d.Name.Name
				if d.Recv != nil && len(d.Recv.List) == 1 {
					key = "(" + typeName(d.Recv.List[0].Type) + ")." + key
				}
				out = append(out, Decl{filepath.Base(p), key, norm(d.Doc), d.Name.IsExported(), hasOut(d.Type)})
			case *ast.GenDecl: // interface methods (encoder.Backend)
				for _, s := range d.Specs {
					ts, ok := s.(*ast.TypeSpec)
					if !ok {
						continue
					}
					it, ok := ts.Type.(*ast.InterfaceType)
					if !ok {
						continue
					}
					for _, m := range it.Methods.List {
						ft, ok := m.Type.(*ast.FuncType)
						if !ok || len(m.Names) == 0 {
							continue
						}
						out = append(out, Decl{filepath.Base(p), ts.Name.Name + "." + m.Names[0].Name, norm(m.Doc), m.Names[0].IsExported(), hasOut(ft)})
					}
				}
			}
		}
	}
	return out
}

func typeName(e ast.Expr) string {
	switch e := e.(type) {
	case *ast.StarExpr:
		return "*" + typeName(e.X)
	case *ast.Ident:
		return e.Name
	case *ast.IndexExpr:
		return typeName(e.X)
	}
	return "?"
}
