// Command kernelinventory writes aikit's kernel inventory: every assembly entry point in the packages that hold one,
// with its architecture, its .s file, the production function that dispatches to it (or why none does), and the test
// that holds it equal to its reference. The rows are the kernel registry's (internal/kernreg's Row, OG1), read
// statically from each package's test files, so one run on any host lists every architecture's kernels.
// kernreg.Complete, in each package's TestKernelRegistry_complete, holds those rows to the .s files' TEXT symbols; this
// tool only gathers them, and its test fails when the committed inventory differs from a fresh one.
//
// Usage:
//
//	go run -C tools ./kernelinventory            # print the inventory
//	go run -C tools ./kernelinventory --write    # rewrite docs/internal/kernel-inventory.md
//
// It ends with one VERDICT line: PASS when every package's rows were read and every row names a TEXT symbol of its
// package and arch; FAIL otherwise, with the reason.
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/townsendmerino/aikit/tools/gpumod"
)

// packages are the packages that hold assembly, in the order the inventory lists them.
var packages = []string{"linalg", "linalg/attn", "linalg/elem", "linalg/bitvec", "embed", "internal/cpufeat"}

// InventoryPath is the committed inventory, relative to the repository root.
const InventoryPath = "docs/internal/kernel-inventory.md"

// row is one kernel registry row, with the .s file that defines its symbol.
type row struct {
	Pkg, Symbol, Arch, File, Dispatch, Undispatched, Oracle, NoSlices string
}

func main() {
	write := flag.Bool("write", false, "rewrite "+InventoryPath)
	flag.Parse()
	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Println("VERDICT: FAIL —", err)
		os.Exit(2)
	}
	rows, err := gather(root)
	if err != nil {
		fmt.Println("VERDICT: FAIL —", err)
		os.Exit(1)
	}
	out := render(rows)
	if *write {
		if err := os.WriteFile(filepath.Join(root, InventoryPath), out, 0o644); err != nil {
			fmt.Println("VERDICT: FAIL —", err)
			os.Exit(2)
		}
		fmt.Printf("wrote %s: %d entry points\n", InventoryPath, len(rows))
	} else if _, err := os.Stdout.Write(out); err != nil {
		fmt.Println("VERDICT: FAIL —", err)
		os.Exit(2)
	}
	fmt.Printf("VERDICT: PASS — %d entry points in %d packages\n", len(rows), len(packages))
}

// gather reads every package's registry rows and locates each symbol's .s file.
func gather(root string) ([]row, error) {
	var all []row
	for _, pkg := range packages {
		dir := filepath.Join(root, pkg)
		rows, err := registryRows(dir, pkg)
		if err != nil {
			return nil, err
		}
		if len(rows) == 0 {
			return nil, fmt.Errorf("%s: no kernel registry rows found", pkg)
		}
		text, err := textSymbols(dir)
		if err != nil {
			return nil, err
		}
		for i := range rows {
			f, ok := text[rows[i].Arch+"/"+rows[i].Symbol]
			if !ok {
				return nil, fmt.Errorf("%s: row %s/%s names no TEXT symbol of its arch", pkg, rows[i].Arch, rows[i].Symbol)
			}
			rows[i].File = f
		}
		all = append(all, rows...)
	}
	return all, nil
}

// registryRows returns the elements of every []kernreg.Row composite literal in dir's test files. A field may be a
// string literal or a package-level string constant of the same package.
func registryRows(dir, pkg string) ([]row, error) {
	fset := token.NewFileSet()
	names, err := filepath.Glob(filepath.Join(dir, "*_test.go"))
	if err != nil {
		return nil, err
	}
	var files []*ast.File
	for _, n := range names {
		f, err := parser.ParseFile(fset, n, nil, parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		files = append(files, f)
	}
	consts := map[string]string{}
	for _, f := range files {
		for _, d := range f.Decls {
			gd, ok := d.(*ast.GenDecl)
			if !ok || gd.Tok != token.CONST {
				continue
			}
			for _, sp := range gd.Specs {
				vs := sp.(*ast.ValueSpec)
				for i, id := range vs.Names {
					if i < len(vs.Values) {
						if v, ok := strLit(vs.Values[i], nil); ok {
							consts[id.Name] = v
						}
					}
				}
			}
		}
	}
	var out []row
	var bad error
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			cl, ok := n.(*ast.CompositeLit)
			if !ok || !isRowSlice(cl.Type) {
				return true
			}
			for _, el := range cl.Elts {
				ecl, ok := el.(*ast.CompositeLit)
				if !ok {
					bad = fmt.Errorf("%s: a kernreg.Row element is not a literal", fset.Position(el.Pos()))
					continue
				}
				r := row{Pkg: pkg}
				for _, kv := range ecl.Elts {
					kve, ok := kv.(*ast.KeyValueExpr)
					if !ok {
						bad = fmt.Errorf("%s: an unkeyed kernreg.Row field", fset.Position(kv.Pos()))
						continue
					}
					v, ok := strLit(kve.Value, consts)
					if !ok {
						bad = fmt.Errorf("%s: kernreg.Row field %v is not a string literal or constant", fset.Position(kve.Pos()), kve.Key)
						continue
					}
					switch kve.Key.(*ast.Ident).Name {
					case "Symbol":
						r.Symbol = v
					case "Arch":
						r.Arch = v
					case "Dispatch":
						r.Dispatch = v
					case "Oracle":
						r.Oracle = v
					case "Undispatched":
						r.Undispatched = v
					case "NoSlices":
						r.NoSlices = v
					}
				}
				out = append(out, r)
			}
			return false
		})
	}
	return out, bad
}

func isRowSlice(t ast.Expr) bool {
	at, ok := t.(*ast.ArrayType)
	if !ok {
		return false
	}
	sel, ok := at.Elt.(*ast.SelectorExpr)
	return ok && sel.Sel.Name == "Row" && fmt.Sprint(sel.X) == "kernreg"
}

// strLit is e's string value: a literal, a constant named in consts, or a + of those.
func strLit(e ast.Expr, consts map[string]string) (string, bool) {
	switch x := e.(type) {
	case *ast.BasicLit:
		if x.Kind != token.STRING {
			return "", false
		}
		v, err := strconv.Unquote(x.Value)
		return v, err == nil
	case *ast.Ident:
		v, ok := consts[x.Name]
		return v, ok
	case *ast.BinaryExpr:
		if x.Op != token.ADD {
			return "", false
		}
		l, ok1 := strLit(x.X, consts)
		r, ok2 := strLit(x.Y, consts)
		return l + r, ok1 && ok2
	case *ast.ParenExpr:
		return strLit(x.X, consts)
	}
	return "", false
}

var textRe = regexp.MustCompile(`^TEXT\s+·(\w+)\(SB\)`)

// textSymbols maps arch/symbol to the .s file that defines it, for dir's .s files.
func textSymbols(dir string) (map[string]string, error) {
	names, err := filepath.Glob(filepath.Join(dir, "*.s"))
	if err != nil {
		return nil, err
	}
	out := map[string]string{}
	for _, n := range names {
		base := filepath.Base(n)
		arch := ""
		for _, a := range []string{"amd64", "arm64"} {
			if strings.HasSuffix(base, "_"+a+".s") {
				arch = a
			}
		}
		if arch == "" {
			continue
		}
		b, err := os.ReadFile(n)
		if err != nil {
			return nil, err
		}
		sc := bufio.NewScanner(bytes.NewReader(b))
		for sc.Scan() {
			if m := textRe.FindStringSubmatch(sc.Text()); m != nil {
				out[arch+"/"+m[1]] = base
			}
		}
	}
	return out, nil
}

// render is the inventory document: one table per package, rows by arch then symbol.
func render(rows []row) []byte {
	var b bytes.Buffer
	b.WriteString(`# Kernel inventory

Generated by ` + "`go run -C tools ./kernelinventory --write`" + ` from the kernel registry (internal/kernreg's rows, one per
assembly entry point, in each package's ` + "`*_kernels_test.go`" + `); ` + "`TestInventory_upToDate`" + ` fails when this file differs
from a fresh run. kernreg.Complete, in each package's ` + "`TestKernelRegistry_complete`" + `, holds the rows to the .s files' TEXT
symbols, so a new kernel without a row fails there. Do not edit by hand.

Each entry point is its Go stub's name. "Dispatched by" is the production function that calls it; a kernel no production
code calls says why instead. "Oracle" is the test that holds it, or its dispatcher, equal to the reference.
`)
	counts := map[string]int{}
	for _, r := range rows {
		counts[r.Arch]++
	}
	fmt.Fprintf(&b, "\n%d entry points: %d amd64, %d arm64.\n", len(rows), counts["amd64"], counts["arm64"])
	var pkgs []string // in the order gather read them (packages' order)
	for _, r := range rows {
		if !slices.Contains(pkgs, r.Pkg) {
			pkgs = append(pkgs, r.Pkg)
		}
	}
	for _, pkg := range pkgs {
		var pr []row
		for _, r := range rows {
			if r.Pkg == pkg {
				pr = append(pr, r)
			}
		}
		slices.SortFunc(pr, func(a, b row) int {
			if c := strings.Compare(a.Arch, b.Arch); c != 0 {
				return c
			}
			return strings.Compare(a.Symbol, b.Symbol)
		})
		fmt.Fprintf(&b, "\n## %s (%d)\n\n| arch | entry point | file | dispatched by | oracle |\n|---|---|---|---|---|\n", pkg, len(pr))
		for _, r := range pr {
			disp := "`" + r.Dispatch + "`"
			switch {
			case r.Dispatch != "":
			case r.Undispatched != "":
				disp = "none: " + r.Undispatched
			default:
				disp = "none (" + r.NoSlices + ")"
			}
			oracle := "`" + r.Oracle + "`"
			if r.Oracle == "" {
				oracle = "none"
			}
			if r.NoSlices != "" {
				oracle += " (not swept: " + r.NoSlices + ")"
			}
			fmt.Fprintf(&b, "| %s | `%s` | %s | %s | %s |\n", r.Arch, r.Symbol, r.File, esc(disp), esc(oracle))
		}
	}
	return b.Bytes()
}

func esc(s string) string { return strings.ReplaceAll(s, "|", `\|`) }
