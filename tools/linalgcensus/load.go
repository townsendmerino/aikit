package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// listed is one package as `go list -deps -test -json` reports it. A test variant ("p [p.test]") lists its test files
// in GoFiles; ImportMap maps an import path as written to the variant the build uses.
type listed struct {
	ImportPath string
	Name       string
	Dir        string
	ForTest    string
	Standard   bool
	GoFiles    []string
	CgoFiles   []string
	Imports    []string
	ImportMap  map[string]string
	Module     *struct {
		Path      string
		Dir       string
		GoVersion string
	}
	Error *struct{ Err string }
}

// loaded is one type-checked package of a configuration.
type loaded struct {
	*listed
	files []*ast.File
	pkg   *types.Package
	info  *types.Info // nil unless the package is ours and has bodies to read
	errs  []string    // type errors, kept for our packages only
	err   error       // could not be checked at all
	done  bool
}

// world is one configuration's package graph, type-checked from source. Packages outside our repositories are checked
// without function bodies; ours are checked in full with use and type records.
type world struct {
	cfg   config
	fset  *token.FileSet
	pkgs  map[string]*loaded
	order []string
	ours  func(dir string) bool
	sizes types.Sizes
}

// goList runs `go list -e -deps -test -json` for patterns in dir under env and decodes its stream.
func goList(dir string, env []string, patterns []string) ([]*listed, error) {
	args := append([]string{"list", "-e", "-deps", "-test", "-json=ImportPath,Name,Dir,ForTest,Standard,GoFiles,CgoFiles,Imports,ImportMap,Module,Error"}, patterns...)
	cmd := exec.Command("go", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), env...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("go list (%s): %v: %s", strings.Join(env, " "), err, firstLines(stderr.String(), 5))
	}
	var ps []*listed
	dec := json.NewDecoder(bytes.NewReader(out))
	for {
		var p listed
		if err := dec.Decode(&p); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, fmt.Errorf("decoding go list output: %v", err)
		}
		ps = append(ps, &p)
	}
	return ps, nil
}

func firstLines(s string, n int) string {
	ls := strings.Split(strings.TrimSpace(s), "\n")
	if len(ls) > n {
		ls = append(ls[:n], "...")
	}
	return strings.Join(ls, "\n")
}

// plainPath strips a test variant's " [p.test]" suffix.
func plainPath(id string) string {
	p, _, _ := strings.Cut(id, " ")
	return p
}

func newWorld(cfg config, ps []*listed, ours func(string) bool) *world {
	w := &world{cfg: cfg, fset: token.NewFileSet(), pkgs: map[string]*loaded{}, ours: ours, sizes: types.SizesFor("gc", cfg.goarch)}
	for _, p := range ps {
		if _, dup := w.pkgs[p.ImportPath]; dup {
			continue
		}
		w.pkgs[p.ImportPath] = &loaded{listed: p}
		w.order = append(w.order, p.ImportPath)
	}
	return w
}

// isTestMain reports the generated "p.test" main package, which has no source of ours.
func (l *loaded) isTestMain() bool {
	return l.Name == "main" && strings.HasSuffix(l.ImportPath, ".test") && l.ForTest == ""
}

// hasTestFiles reports whether a variant compiles a _test.go file; a variant without one repeats its plain package.
func (l *loaded) hasTestFiles() bool {
	for _, f := range l.GoFiles {
		if strings.HasSuffix(f, "_test.go") {
			return true
		}
	}
	return false
}

// checkAll type-checks every package that one of ours depends on, and ours.
func (w *world) checkAll() {
	for _, id := range w.order {
		l := w.pkgs[id]
		if l.isTestMain() || !w.ours(l.Dir) {
			continue
		}
		w.check(id)
	}
}

func (w *world) check(id string) (*types.Package, error) {
	if id == "unsafe" {
		return types.Unsafe, nil
	}
	l := w.pkgs[id]
	if l == nil {
		return nil, fmt.Errorf("package %q is not in go list's graph", id)
	}
	if l.done {
		return l.pkg, l.err
	}
	l.done = true
	if l.Error != nil && len(l.GoFiles)+len(l.CgoFiles) == 0 {
		l.err = errors.New(l.Error.Err)
		return nil, l.err
	}
	mine := w.ours(l.Dir)
	bodies := mine && (l.ForTest == "" || l.hasTestFiles())
	for _, f := range append(append([]string{}, l.GoFiles...), l.CgoFiles...) {
		if !filepath.IsAbs(f) {
			f = filepath.Join(l.Dir, f)
		}
		af, err := parser.ParseFile(w.fset, f, nil, parser.SkipObjectResolution)
		if err != nil {
			if mine {
				l.errs = append(l.errs, err.Error())
			}
			if af == nil {
				continue
			}
		}
		l.files = append(l.files, af)
	}
	conf := types.Config{
		Importer:         importerFunc(func(path string) (*types.Package, error) { return w.check(w.resolve(l, path)) }),
		FakeImportC:      true,
		Sizes:            w.sizes,
		IgnoreFuncBodies: !bodies,
		Error: func(err error) {
			if mine {
				l.errs = append(l.errs, err.Error())
			}
		},
	}
	if l.Module != nil && l.Module.GoVersion != "" {
		conf.GoVersion = "go" + l.Module.GoVersion
	}
	if bodies {
		l.info = &types.Info{Uses: map[*ast.Ident]types.Object{}, Defs: map[*ast.Ident]types.Object{}, Types: map[ast.Expr]types.TypeAndValue{}}
	}
	// The package path is the plain one, so a test variant's objects key the same as its plain package's.
	pkg, _ := conf.Check(plainPath(l.ImportPath), w.fset, l.files, l.info)
	l.pkg = pkg
	return pkg, nil
}

func (w *world) resolve(l *loaded, path string) string {
	if m, ok := l.ImportMap[path]; ok {
		return m
	}
	return path
}

type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }
