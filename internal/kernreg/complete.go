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
}

func readDecls(dir, arch string) (goDecls, error) {
	fset := token.NewFileSet()
	files, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		return goDecls{}, err
	}
	ctx := build.Default
	ctx.GOOS, ctx.GOARCH, ctx.BuildTags = "linux", arch, nil
	d := goDecls{calls: map[string]map[string]bool{}, tests: map[string]string{}}
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
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			name := fn.Name.Name
			if isTest {
				if strings.HasPrefix(name, "Test") && fn.Recv == nil {
					d.tests[name] += string(src[fset.Position(fn.Body.Pos()).Offset:fset.Position(fn.Body.End()).Offset])
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
					if id, ok := c.Fun.(*ast.Ident); ok {
						set[id.Name] = true
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

// Complete checks rows against dir's assembly and Go source, and adapters against the
// host architecture's rows:
//   - every TEXT symbol has exactly one row, and every row names a TEXT symbol;
//   - a row's Dispatch is a non-test function that calls Symbol, and its Oracle is a Test
//     function whose body names Symbol or Dispatch, both in files that build for the row's
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
		if r.Dispatch == "" {
			errorf("%s: no Dispatch", key)
		} else if !decls.calls[r.Dispatch][r.Symbol] {
			errorf("%s: Dispatch %s is not a non-test function of %s that calls %s", key, r.Dispatch, dir, r.Symbol)
		}
		body, ok := decls.tests[r.Oracle]
		switch {
		case r.Oracle == "":
			errorf("%s: no Oracle", key)
		case !ok:
			errorf("%s: Oracle %s is not a Test function in %s", key, r.Oracle, dir)
		default:
			words := wordRE.FindAllString(body, -1)
			if !slices.Contains(words, r.Symbol) && !slices.Contains(words, dispatchFunc(r.Dispatch)) {
				errorf("%s: Oracle %s names neither %s nor %s", key, r.Oracle, r.Symbol, r.Dispatch)
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
	slices.Sort(probs[len(probs)-len(have):])
	return probs, nil
}

// dispatchFunc is the function name of a Dispatch, without a receiver.
func dispatchFunc(d string) string {
	if _, after, ok := strings.Cut(d, "."); ok {
		return after
	}
	return d
}
