package linalg

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// R-11 (goinfer docs/tasks/task-recompute-audit.md §5), the documentation half: every exported function that writes into a caller's buffer says, in its doc comment,
// which of these it is —
//
//	Output contract: overwrites <out>; do not pre-zero.      (the caller must not zero first: it is dead work, or hides a missing write)
//	Output contract: zeroes <out> internally; ...            (it clears the span itself, then accumulates)
//	Output contract: accumulates into <out>; caller zeroes.  (no function here does; the form is reserved so a future one can say so)
//	Output contract: rewrites <out> in place ...             (input and output share a buffer; there is no separate destination)
//
// The probe in outputcontract_test.go proves the statements are TRUE; this test proves they are PRESENT, and that a new writer cannot be added without one.
// It parses source rather than loading the package, so it covers every arch-tagged file (arm64 row4, amd64 split-half) on any host.
var contractForm = regexp.MustCompile(`Output contract: (overwrites .+; do not pre-zero|zeroes .+ internally|accumulates into .+; caller zeroes|rewrites .+ in place)`)

// outputParamNames: a parameter with one of these names is an output buffer.
var outputParamNames = map[string]bool{"dst": true, "sums": true}

// nonDstWriters are writers whose output parameters are not called dst, so the name scan cannot find them. Keep this list and the code agreeing: a name here that no
// longer exists, or no longer carries a contract line, fails.
var nonDstWriters = map[string]bool{
	"AttendTileFused": true, "AttendTileFusedContractExp": true, "GatherVBlockMajor": true,
	"QuantizeActivationsInto": true, "QuantizeActivationsGroupedInto": true, "QuantizeRowInt8": true, "SumActGroupsInto": true, "QuantizeGroupInt4Row": true,
	"RepackInt4SplitHalfInPlace": true, "RepackInt4Row4InPlace": true,
	"MatmulBTW8A8Batch": true, "MatmulBTW4A8Batch": true, // outputs are each op's Dst field
}

type declDoc struct {
	file, key, doc string
	exported       bool
	outParam       bool
}

func parseDecls(t *testing.T, dir string) []declDoc {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no sources under %s (%v)", dir, err)
	}
	var out []declDoc
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
					key = "(" + types(d.Recv.List[0].Type) + ")." + key
				}
				out = append(out, declDoc{filepath.Base(p), key, norm(d.Doc), d.Name.IsExported(), hasOut(d.Type)})
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
						out = append(out, declDoc{filepath.Base(p), ts.Name.Name + "." + m.Names[0].Name, norm(m.Doc), m.Names[0].IsExported(), hasOut(ft)})
					}
				}
			}
		}
	}
	return out
}

func types(e ast.Expr) string {
	switch e := e.(type) {
	case *ast.StarExpr:
		return "*" + types(e.X)
	case *ast.Ident:
		return e.Name
	case *ast.IndexExpr:
		return types(e.X)
	}
	return "?"
}

func TestOutputContract_everyWriterIsDocumented(t *testing.T) {
	seenNonDst := map[string]bool{}
	for _, dir := range []string{".", "../embed", "../encoder", "../gpu/encmetal", "../gpu/enccuda"} {
		for _, d := range parseDecls(t, dir) {
			// unexported helpers carry no contract obligation, and neither do exported methods of unexported types
			unexportedRecv := strings.Contains(d.key, "(") && !ast.IsExported(strings.TrimLeft(d.key[1:strings.Index(d.key, ")")], "*"))
			isWriter := d.exported && !unexportedRecv && (d.outParam || (dir == "." && nonDstWriters[d.key]))
			if dir == "." && nonDstWriters[d.key] {
				seenNonDst[d.key] = true
			}
			if d.exported && strings.Contains(d.doc, "Output contract:") && !contractForm.MatchString(d.doc) {
				t.Errorf("%s/%s %s: 'Output contract:' line is not one of the recognised forms (overwrites …; do not pre-zero | zeroes … internally | accumulates into …; caller zeroes | rewrites … in place)", dir, d.file, d.key)
			}
			// cpuBackend is unexported but is the CPU implementation of an exported interface; it is documented too
			if d.key == "(*cpuBackend).MatmulBT" {
				isWriter = true
			}
			if isWriter && !contractForm.MatchString(d.doc) {
				t.Errorf("%s/%s: %s writes into a caller's buffer but its doc comment has no 'Output contract:' line (R-11) — say which of overwrites / zeroes internally / accumulates it is", dir, d.file, d.key)
			}
		}
	}
	for n := range nonDstWriters {
		if !seenNonDst[n] {
			t.Errorf("nonDstWriters lists %s, which no longer exists in linalg", n)
		}
	}
	// the three embed writers the name scan cannot see (dst is a parameter of a returned closure, or named differently)
	for _, want := range []string{"DequantMXFP4Blocks", "DequantMXFP4Split", "L2Normalize", "(*GGUFFile).RowDequantizer"} {
		found := false
		for _, d := range parseDecls(t, "../embed") {
			if d.key == want {
				found = true
				if !contractForm.MatchString(d.doc) {
					t.Errorf("embed %s has no 'Output contract:' line", want)
				}
			}
		}
		if !found {
			t.Errorf("embed %s no longer exists; update this test", want)
		}
	}
	if _, err := os.Stat("../embed"); err != nil {
		t.Fatal(err)
	}
}
