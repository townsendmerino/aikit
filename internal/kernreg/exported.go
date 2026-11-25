package kernreg

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"path/filepath"
	"slices"
	"strings"
)

// Exported returns the exported functions and the exported methods of exported types
// ("Type.Method") declared in dir's non-test files that build for linux on amd64 or
// arm64 with no extra tags.
func Exported(dir string) ([]string, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		return nil, err
	}
	fset := token.NewFileSet()
	set := map[string]bool{}
	for _, path := range files {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		builds := false
		for _, arch := range []string{"amd64", "arm64"} {
			ctx := build.Default
			ctx.GOOS, ctx.GOARCH, ctx.BuildTags = "linux", arch, nil
			ok, err := ctx.MatchFile(dir, filepath.Base(path))
			if err != nil {
				return nil, err
			}
			builds = builds || ok
		}
		if !builds {
			continue
		}
		f, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		for _, d := range f.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok || !fn.Name.IsExported() {
				continue
			}
			name := fn.Name.Name
			if fn.Recv != nil {
				recv := recvName(fn.Recv)
				if !ast.IsExported(recv) {
					continue
				}
				name = recv + "." + name
			}
			set[name] = true
		}
	}
	return slices.Sorted(func(yield func(string) bool) {
		for k := range set {
			if !yield(k) {
				return
			}
		}
	}), nil
}

// ExportedProblems checks that every exported name in dir is covered by exactly one of
// swept (names with a tiny-shape adapter) and exempt (name → why it needs none), and
// that every name in either still exists.
func ExportedProblems(dir string, swept []string, exempt map[string]string) ([]string, error) {
	names, err := Exported(dir)
	if err != nil {
		return nil, err
	}
	var probs []string
	count := map[string]int{}
	for _, s := range swept {
		count[s]++
	}
	for _, n := range names {
		_, ex := exempt[n]
		switch c := count[n]; {
		case c == 0 && !ex:
			probs = append(probs, fmt.Sprintf("%s: an exported entry point with no tiny-shape adapter and no listed reason", n))
		case c > 0 && ex:
			probs = append(probs, fmt.Sprintf("%s: has an adapter and a listed reason", n))
		case c > 1:
			probs = append(probs, fmt.Sprintf("%s: %d adapters", n, c))
		}
	}
	for _, s := range slices.Concat(swept, slices.Collect(func(yield func(string) bool) {
		for k := range exempt {
			if !yield(k) {
				return
			}
		}
	})) {
		if !slices.Contains(names, s) {
			probs = append(probs, fmt.Sprintf("%s: listed but not an exported name of %s", s, dir))
		}
	}
	slices.Sort(probs)
	return slices.Compact(probs), nil
}
