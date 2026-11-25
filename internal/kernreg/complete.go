package kernreg

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

var textRE = regexp.MustCompile(`(?m)^TEXT\s+·(\w+)\(SB\)`)

// Symbols returns the TEXT symbols of dir's .s files, keyed "arch/symbol", with the file
// each is in, for the two architectures aikit writes assembly for; a .s file for any
// other is an error.
func Symbols(dir string) (map[string]string, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.s"))
	if err != nil {
		return nil, err
	}
	out := map[string]string{}
	for _, f := range files {
		base := strings.TrimSuffix(filepath.Base(f), ".s")
		var arch string
		switch {
		case strings.HasSuffix(base, "_amd64"):
			arch = "amd64"
		case strings.HasSuffix(base, "_arm64"):
			arch = "arm64"
		default:
			return nil, fmt.Errorf("%s: an assembly file without an _amd64 or _arm64 suffix; teach kernreg.Symbols its architecture", f)
		}
		src, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		for _, m := range textRE.FindAllSubmatch(src, -1) {
			out[arch+"/"+string(m[1])] = filepath.Base(f)
		}
	}
	return out, nil
}

// goDecls is what Complete reads from the Go files of a package that build for linux on
// one architecture with no extra tags.
type goDecls struct {
	calls map[string]map[string]bool // non-test function -> names it calls
	tests map[string]string          // Test function -> its body's source
	helps map[string]string          // other top-level test-file func or var -> its source
}

func readDecls(dir, arch string) (goDecls, error) {
	fset := token.NewFileSet()
	files, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		return goDecls{}, err
	}
	ctx := build.Default
	ctx.GOOS, ctx.GOARCH, ctx.BuildTags = "linux", arch, nil
	d := goDecls{calls: map[string]map[string]bool{}, tests: map[string]string{}, helps: map[string]string{}}
	for _, path := range files {
		ok, err := ctx.MatchFile(dir, filepath.Base(path))
		if err != nil {
			return goDecls{}, err
		}
		if !ok {
			continue
		}
		src, err := os.ReadFile(path)
		if err != nil {
			return goDecls{}, err
		}
		f, err := parser.ParseFile(fset, path, src, parser.SkipObjectResolution)
		if err != nil {
			return goDecls{}, err
		}
		isTest := strings.HasSuffix(path, "_test.go")
		srcOf := func(n ast.Node) string {
			return string(src[fset.Position(n.Pos()).Offset:fset.Position(n.End()).Offset])
		}
		for _, decl := range f.Decls {
			if gd, ok := decl.(*ast.GenDecl); ok && isTest && gd.Tok == token.VAR {
				for _, spec := range gd.Specs {
					for _, id := range spec.(*ast.ValueSpec).Names {
						d.helps[id.Name] += srcOf(spec)
					}
				}
				continue
			}
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			name := fn.Name.Name
			if isTest {
				if strings.HasPrefix(name, "Test") && fn.Recv == nil {
					d.tests[name] += srcOf(fn.Body)
				} else if fn.Recv == nil {
					d.helps[name] += srcOf(fn.Body)
				}
				continue
			}
			if fn.Recv != nil {
				name = recvName(fn.Recv) + "." + name
			}
			set := d.calls[name]
			if set == nil {
				set = map[string]bool{}
				d.calls[name] = set
			}
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				if c, ok := n.(*ast.CallExpr); ok {
					switch f := c.Fun.(type) {
					case *ast.Ident:
						set[f.Name] = true
					case *ast.SelectorExpr:
						set[f.Sel.Name] = true // a method, matched by name
					}
				}
				return true
			})
		}
	}
	return d, nil
}

func recvName(fl *ast.FieldList) string {
	e := fl.List[0].Type
	if s, ok := e.(*ast.StarExpr); ok {
		e = s.X
	}
	if ix, ok := e.(*ast.IndexExpr); ok {
		e = ix.X
	}
	if id, ok := e.(*ast.Ident); ok {
		return id.Name
	}
	return "?"
}

var wordRE = regexp.MustCompile(`\w+`)

// oracleHops is how far up the package's call graph from a kernel an oracle may start:
// a test of the exported wrapper over the dispatcher over the kernel counts.
const oracleHops = 4

// callers returns sym and every non-test function that reaches it in at most hops calls,
// by name (a method by its method name).
func (d goDecls) callers(sym string, hops int) []string {
	reach := map[string]bool{sym: true}
	frontier := []string{sym}
	for range hops {
		var next []string
		for fn, calls := range d.calls {
			name := dispatchFunc(fn)
			if reach[name] {
				continue
			}
			for _, f := range frontier {
				if calls[f] {
					reach[name] = true
					next = append(next, name)
					break
				}
			}
		}
		frontier = next
	}
	return slices.Collect(func(yield func(string) bool) {
		for k := range reach {
			if !yield(k) {
				return
			}
		}
	})
}

// reaches reports whether src names one of targets, or names a test-file helper function
// or variable whose source does: one level of indirection, for an oracle that calls
// through a table or a shared helper.
func (d goDecls) reaches(src string, targets ...string) bool {
	words := wordRE.FindAllString(src, -1)
	for _, w := range words {
		if slices.Contains(targets, w) {
			return true
		}
	}
	for _, w := range words {
		if h, ok := d.helps[w]; ok && slices.ContainsFunc(wordRE.FindAllString(h, -1), func(x string) bool { return slices.Contains(targets, x) }) {
			return true
		}
	}
	return false
}

// Complete checks rows against dir's assembly and Go source, and adapters against the
// host architecture's rows:
//   - every TEXT symbol has exactly one row, and every row names a TEXT symbol;
//   - a row's Dispatch is a non-test function that calls Symbol, and its Oracle is a Test
//     function that names Symbol or a function within oracleHops calls of it (directly, or
//     through a test helper or table it names), both in files that build for the row's
//     architecture (rows with NoSlices need neither);
//   - on the host architecture, every row without NoSlices has exactly one adapter and
//     every adapter has a row.
func Complete(t *testing.T, dir string, rows []Row, adapters []Adapter) {
	t.Helper()
	probs, err := Problems(dir, HostArch, rows, adapters)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range probs {
		t.Error(p)
	}
	t.Logf("%d rows; %d adapters on %s", len(rows), len(adapters), HostArch)
}

// Problems is Complete's check for host architecture arch, returning one line per
// problem.
func Problems(dir, arch string, rows []Row, adapters []Adapter) ([]string, error) {
	syms, err := Symbols(dir)
	if err != nil {
		return nil, err
	}
	declsByArch := map[string]goDecls{}
	for _, a := range []string{"amd64", "arm64"} {
		d, err := readDecls(dir, a)
		if err != nil {
			return nil, err
		}
		declsByArch[a] = d
	}
	var probs []string
	errorf := func(format string, args ...any) { probs = append(probs, fmt.Sprintf(format, args...)) }
	seen := map[string]bool{}
	for _, r := range rows {
		key := r.Arch + "/" + r.Symbol
		if seen[key] {
			errorf("%s: two rows", key)
		}
		seen[key] = true
		if _, ok := syms[key]; !ok {
			errorf("%s: a row with no TEXT symbol in %s's %s assembly", key, dir, r.Arch)
		}
		if r.NoSlices != "" {
			continue
		}
		decls := declsByArch[r.Arch]
		switch {
		case r.Dispatch != "" && r.Undispatched != "":
			errorf("%s: both a Dispatch and an Undispatched reason", key)
		case r.Undispatched != "":
			for fn, calls := range decls.calls {
				if calls[r.Symbol] {
					errorf("%s: listed Undispatched, but %s calls it", key, fn)
				}
			}
		case r.Dispatch == "":
			errorf("%s: no Dispatch and no Undispatched reason", key)
		case !decls.calls[r.Dispatch][r.Symbol]:
			errorf("%s: Dispatch %s is not a non-test function of %s that calls %s", key, r.Dispatch, dir, r.Symbol)
		}
		body, ok := decls.tests[r.Oracle]
		switch {
		case r.Oracle == "":
			errorf("%s: no Oracle", key)
		case !ok:
			errorf("%s: Oracle %s is not a Test function in %s", key, r.Oracle, dir)
		default:
			if !decls.reaches(body, decls.callers(r.Symbol, oracleHops)...) {
				errorf("%s: Oracle %s reaches %s neither directly nor through a function within %d calls of it, nor through a test helper or table it names", key, r.Oracle, r.Symbol, oracleHops)
			}
		}
	}
	var missing []string
	for key, file := range syms {
		if !seen[key] {
			missing = append(missing, fmt.Sprintf("%s (%s)", key, file))
		}
	}
	slices.Sort(missing)
	for _, m := range missing {
		errorf("%s: a TEXT symbol with no row: add one beside its kernel's test file, with an adapter or a NoSlices reason", m)
	}

	have := map[string]int{}
	for _, ad := range adapters {
		have[ad.Symbol]++
	}
	for _, r := range rows {
		if r.Arch != arch {
			continue
		}
		n := have[r.Symbol]
		switch {
		case r.NoSlices != "" && n > 0:
			errorf("%s/%s: has an adapter and a NoSlices reason", r.Arch, r.Symbol)
		case r.NoSlices == "" && n == 0:
			errorf("%s/%s: no adapter: the over-read sweep cannot reach it", r.Arch, r.Symbol)
		case n > 1:
			errorf("%s/%s: %d adapters", r.Arch, r.Symbol, n)
		}
		delete(have, r.Symbol)
	}
	for sym := range have {
		errorf("%s/%s: an adapter with no row", arch, sym)
	}
	slices.Sort(probs)
	return probs, nil
}

// dispatchFunc is the function name of a Dispatch, without a receiver.
func dispatchFunc(d string) string {
	if _, after, ok := strings.Cut(d, "."); ok {
		return after
	}
	return d
}
