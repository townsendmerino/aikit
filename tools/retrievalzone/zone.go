// Package retrievalzone answers one question about a change to the root module: could
// it change retrieval results or the retrieval API? It is the library behind
// tools/retrievalmark and the release gate's retrieval-mark check.
//
// THE ZONE (docs/architecture.md § Retrieval zone) is the set of packages a retrieval
// consumer's search path runs through. ken is that consumer. Packages lists the ones
// that are in the zone whole; Symbols lists the entry points ken uses in the two
// packages that are in it only in part (embed, linalg), because goinfer drives most of
// their churn and the zone uses a fraction of each.
//
// HOW A FILE IS CLASSIFIED. Every non-test Go file in the root module is parsed. A
// declaration is REACHED if it is in a zone package, is one of Symbols, or is named by
// a reached declaration; a file is in the zone if it holds a reached declaration. The
// walk is by name, not by type, and it over-approximates on purpose:
//
//   - every file of a package is read whatever its build constraints, so the answer
//     covers every GOOS, GOARCH and build tag at once;
//   - an identifier that matches a package-level name counts as a use of it, even
//     where a local variable shadows it;
//   - a reached type brings every method declared on it, in any file, which covers
//     interface dispatch without needing to know which calls are dynamic;
//   - every init function, and every package-level variable whose initializer makes a
//     call, is reached in each package the zone links (side effects run at import);
//   - a reached function with no body brings the assembly file that defines it, and
//     everything that file names. If no assembly file is found for it, it brings every
//     assembly file in its package.
//
// HOW IT FAILS SAFE. Nothing is listed by hand below the package level, so a new file
// needs no entry: it is parsed and walked like the rest. A path no rule recognizes is
// in the zone. A tree that does not parse puts every changed path in the zone. A file
// is judged in each tree it exists in (before and after), and either saying "zone"
// is enough.
//
// WHAT IT DOES NOT SEE. go:linkname and cgo: the root module uses neither, and a file
// that does is an error, which puts every changed path in the zone. A change of Go
// toolchain on the consumer's side. Speed: an unreached file cannot change a result,
// and this tool makes no claim about how fast the result arrives.
package retrievalzone

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path"
	"regexp"
	"sort"
	"strings"
)

// Packages are the root-module packages that are in the retrieval zone whole: any
// change to a non-test file in one of them is a retrieval change. Paths are relative
// to the module root.
//
// ken imports topk, fuse, chunk, chunk/regex, chunk/markdown, bm25, ann and encoder
// (read from ken at 23a635b, 2026-11-10). sparse is in ken's build through encoder.
// hybrid and late are retrieval API that ken does not import today; they are listed so
// the mark stays true for any retrieval consumer, and they cost nothing while they do
// not change.
var Packages = []string{
	"topk", "fuse",
	"chunk", "chunk/regex", "chunk/markdown",
	"bm25", "sparse", "ann", "hybrid", "late",
	"encoder",
}

// Symbols are the entry points ken uses in the packages that are in the zone only in
// part. Everything these reach is in the zone; the rest of the package is not, unless
// a zone package reaches it too (encoder reaches a good deal of both).
//
// Derived by listing every `embed.X` and `linalg.X` in ken's Go files, tests included
// (ken at 23a635b) and keeping the ones aikit declares: ken's `embed.FS` is the standard
// library's. `retrievalmark --ken <dir>` repeats that against a ken checkout and fails
// if ken uses a package or symbol this file does not name.
var Symbols = map[string][]string{
	"embed":  {"StaticModel", "Load", "LoadFromFS", "LoadMmap"},
	"linalg": {"Workspace", "QuantizeRowInt8", "PackSignBitsRow"},
}

// Source is a snapshot of the repository's files: the working tree or a git ref.
type Source interface {
	// Files lists every file path, slash-separated, relative to the repo root.
	Files() []string
	// Read returns one file's content.
	Read(path string) ([]byte, error)
}

// ref names a package-level declaration: dir is the package directory relative to the
// module root ("" means the referring declaration's own package).
type ref struct{ dir, name string }

type decl struct {
	pkg  *pkg
	file string
	name string // "Name", "(T).Method", "init"
	recv string // receiver base type for a method
	refs []ref

	isType     bool
	bodyless   bool // a func with no body: implemented in assembly
	sideEffect bool // init, or a var whose initializer makes a call

	reached bool
	via     *decl
}

func (d *decl) String() string { return d.file + ": " + d.name }

type asmFile struct {
	path    string
	defines map[string]bool
	names   []string
}

type pkg struct {
	dir     string
	byName  map[string][]*decl
	methods map[string][]*decl
	decls   []*decl
	imports map[string]bool // root-module package dirs any of its files import
	asm     []*asmFile
	goFiles map[string]bool
}

// Analysis is the reachability result for one tree.
type Analysis struct {
	module  string
	nested  []string // directories that hold a nested module's go.mod
	pkgs    map[string]*pkg
	zone    map[string]bool   // Packages
	linked  map[string]bool   // packages in the zone's import closure
	reached map[string]string // file → why
	files   map[string]bool
}

var (
	moduleLine = regexp.MustCompile(`(?m)^module\s+(\S+)`)
	asmDefine  = regexp.MustCompile(`(?m)^(?:TEXT|GLOBL|DATA)\s+·([A-Za-z0-9_]+)`)
	asmName    = regexp.MustCompile(`·([A-Za-z0-9_]+)`)
)

// ignoredByGo reports a path the go tool never builds: a directory or file whose name
// starts with "." or "_".
func ignoredByGo(p string) bool {
	for el := range strings.SplitSeq(p, "/") {
		if strings.HasPrefix(el, ".") || strings.HasPrefix(el, "_") {
			return true
		}
	}
	return false
}

func inTestdata(p string) bool {
	return strings.HasPrefix(p, "testdata/") || strings.Contains(p, "/testdata/")
}

// Analyze parses src's root module and walks it from the zone.
func Analyze(src Source) (*Analysis, error) {
	a := &Analysis{pkgs: map[string]*pkg{}, zone: map[string]bool{}, linked: map[string]bool{},
		reached: map[string]string{}, files: map[string]bool{}}
	for _, p := range Packages {
		a.zone[p] = true
	}
	all := src.Files()
	for _, f := range all {
		a.files[f] = true
		if path.Base(f) == "go.mod" && f != "go.mod" {
			a.nested = append(a.nested, path.Dir(f))
		}
	}
	gomod, err := src.Read("go.mod")
	if err != nil {
		return nil, fmt.Errorf("read go.mod: %w", err)
	}
	m := moduleLine.FindSubmatch(gomod)
	if m == nil {
		return nil, fmt.Errorf("go.mod has no module line")
	}
	a.module = string(m[1])

	fset := token.NewFileSet()
	for _, f := range all {
		if a.inNested(f) || ignoredByGo(f) || inTestdata(f) {
			continue
		}
		dir := path.Dir(f)
		if dir == "." {
			dir = ""
		}
		switch {
		case strings.HasSuffix(f, "_test.go"):
		case strings.HasSuffix(f, ".go"):
			data, err := src.Read(f)
			if err != nil {
				return nil, fmt.Errorf("read %s: %w", f, err)
			}
			if bytes.Contains(data, []byte("//go:linkname")) || bytes.Contains(data, []byte("import \"C\"")) {
				return nil, fmt.Errorf("%s uses go:linkname or cgo, which this tool cannot follow", f)
			}
			file, err := parser.ParseFile(fset, f, data, parser.SkipObjectResolution)
			if err != nil {
				return nil, fmt.Errorf("parse %s: %w", f, err)
			}
			if err := a.addGoFile(a.pkgAt(dir), f, file); err != nil {
				return nil, err
			}
		case strings.HasSuffix(f, ".s"):
			data, err := src.Read(f)
			if err != nil {
				return nil, fmt.Errorf("read %s: %w", f, err)
			}
			af := &asmFile{path: f, defines: map[string]bool{}}
			for _, d := range asmDefine.FindAllSubmatch(data, -1) {
				af.defines[string(d[1])] = true
			}
			seen := map[string]bool{}
			for _, n := range asmName.FindAllSubmatch(data, -1) {
				if !seen[string(n[1])] {
					seen[string(n[1])] = true
					af.names = append(af.names, string(n[1]))
				}
			}
			p := a.pkgAt(dir)
			p.asm = append(p.asm, af)
		}
	}
	a.walk()
	return a, nil
}

func (a *Analysis) inNested(f string) bool {
	for _, n := range a.nested {
		if strings.HasPrefix(f, n+"/") {
			return true
		}
	}
	return false
}

func (a *Analysis) pkgAt(dir string) *pkg {
	p := a.pkgs[dir]
	if p == nil {
		p = &pkg{dir: dir, byName: map[string][]*decl{}, methods: map[string][]*decl{},
			imports: map[string]bool{}, goFiles: map[string]bool{}}
		a.pkgs[dir] = p
	}
	return p
}

// addGoFile indexes one file's package-level declarations and what each one names.
func (a *Analysis) addGoFile(p *pkg, name string, file *ast.File) error {
	p.goFiles[name] = true
	// import name → root-module package dir
	imports := map[string]string{}
	for _, im := range file.Imports {
		ipath := strings.Trim(im.Path.Value, `"`)
		dir, ok := strings.CutPrefix(ipath, a.module+"/")
		if !ok {
			continue
		}
		p.imports[dir] = true
		local := path.Base(dir)
		if im.Name != nil {
			switch im.Name.Name {
			case ".":
				return fmt.Errorf("%s dot-imports %s: names cannot be resolved by this tool", name, ipath)
			case "_":
				continue
			}
			local = im.Name.Name
		}
		imports[local] = dir
	}
	collect := func(n ast.Node) (refs []ref, calls bool) {
		var visit func(ast.Node) bool
		visit = func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.CallExpr:
				calls = true
			case *ast.SelectorExpr:
				if id, ok := x.X.(*ast.Ident); ok {
					if dir, ok := imports[id.Name]; ok {
						refs = append(refs, ref{dir: dir, name: x.Sel.Name})
						return false
					}
				}
				// A field or method name is not a package-level name, and the method
				// is reached through its type: walk the operand only.
				ast.Inspect(x.X, visit)
				return false
			case *ast.Ident:
				refs = append(refs, ref{name: x.Name})
			}
			return true
		}
		ast.Inspect(n, visit)
		return refs, calls
	}
	add := func(d *decl) {
		d.pkg, d.file = p, name
		p.decls = append(p.decls, d)
		if d.recv != "" {
			p.methods[d.recv] = append(p.methods[d.recv], d)
			return
		}
		if d.name == "init" || d.name == "_" {
			return
		}
		p.byName[d.name] = append(p.byName[d.name], d)
	}
	for _, n := range file.Decls {
		switch x := n.(type) {
		case *ast.FuncDecl:
			d := &decl{name: x.Name.Name, bodyless: x.Body == nil}
			d.refs, _ = collect(x)
			if x.Recv != nil && len(x.Recv.List) > 0 {
				d.recv = recvBase(x.Recv.List[0].Type)
				d.name = "(" + d.recv + ")." + x.Name.Name
			} else if x.Name.Name == "init" {
				d.sideEffect = true
			}
			add(d)
		case *ast.GenDecl:
			switch x.Tok {
			case token.CONST:
				// One unit: iota and implicit repetition tie a block's specs together.
				refs, _ := collect(x)
				for _, s := range x.Specs {
					for _, id := range s.(*ast.ValueSpec).Names {
						add(&decl{name: id.Name, refs: refs})
					}
				}
			case token.VAR:
				for _, s := range x.Specs {
					vs := s.(*ast.ValueSpec)
					refs, calls := collect(vs)
					for _, id := range vs.Names {
						add(&decl{name: id.Name, refs: refs, sideEffect: calls})
					}
				}
			case token.TYPE:
				for _, s := range x.Specs {
					ts := s.(*ast.TypeSpec)
					refs, _ := collect(ts)
					add(&decl{name: ts.Name.Name, refs: refs, isType: true})
				}
			}
		}
	}
	return nil
}

func recvBase(e ast.Expr) string {
	for {
		switch x := e.(type) {
		case *ast.StarExpr:
			e = x.X
		case *ast.ParenExpr:
			e = x.X
		case *ast.IndexExpr:
			e = x.X
		case *ast.IndexListExpr:
			e = x.X
		case *ast.Ident:
			return x.Name
		default:
			return "?"
		}
	}
}

// walk marks everything the zone reaches.
func (a *Analysis) walk() {
	// The packages the zone links: the zone packages, the partly-in packages, and
	// everything those import, to a fixed point.
	var linkQ []string
	link := func(dir string) {
		if !a.linked[dir] && a.pkgs[dir] != nil {
			a.linked[dir] = true
			linkQ = append(linkQ, dir)
		}
	}
	for dir := range a.zone {
		link(dir)
	}
	for dir := range Symbols {
		link(dir)
	}
	for len(linkQ) > 0 {
		dir := linkQ[0]
		linkQ = linkQ[1:]
		for imp := range a.pkgs[dir].imports {
			link(imp)
		}
	}

	var q []*decl
	reach := func(d, via *decl) {
		if !d.reached {
			d.reached, d.via = true, via
			q = append(q, d)
		}
	}
	for _, dir := range sortedKeys(a.linked) {
		p := a.pkgs[dir]
		for _, d := range p.decls {
			if a.zone[dir] || d.sideEffect {
				reach(d, nil)
			}
		}
		for _, name := range Symbols[dir] {
			for _, d := range p.byName[name] {
				reach(d, nil)
			}
		}
	}
	asmReached := map[string]*decl{}
	for len(q) > 0 {
		d := q[0]
		q = q[1:]
		if _, ok := a.reached[d.file]; !ok {
			a.reached[d.file] = a.why(d)
			// The file is in: so is everything else it declares. That makes the set
			// of reached files closed under "names", which is what lets a compiler
			// check it (closure_test.go builds the zone with every other file deleted).
			for _, t := range d.pkg.decls {
				if t.file == d.file {
					reach(t, d)
				}
			}
		}
		for _, r := range d.refs {
			p := d.pkg
			if r.dir != "" {
				p = a.pkgs[r.dir]
			}
			if p == nil {
				continue
			}
			for _, t := range p.byName[r.name] {
				reach(t, d)
			}
		}
		if d.isType {
			for _, m := range d.pkg.methods[d.name] {
				reach(m, d)
			}
		}
		if d.bodyless {
			var hit []*asmFile
			for _, af := range d.pkg.asm {
				if af.defines[d.name] {
					hit = append(hit, af)
				}
			}
			if len(hit) == 0 {
				hit = d.pkg.asm // no definer found: take every assembly file
			}
			for _, af := range hit {
				if asmReached[af.path] != nil {
					continue
				}
				asmReached[af.path] = d
				a.reached[af.path] = "implements " + a.why(d)
				for _, n := range af.names {
					for _, t := range d.pkg.byName[n] {
						reach(t, d)
					}
				}
			}
		}
	}
}

// why renders the chain from a root to d, nearest first, for the report.
func (a *Analysis) why(d *decl) string {
	var parts []string
	for n := d; n != nil && len(parts) < 4; n = n.via {
		parts = append(parts, n.String())
	}
	s := strings.Join(parts, " <- ")
	root := d
	for root.via != nil {
		root = root.via
	}
	switch {
	case a.zone[root.pkg.dir]:
	case root.sideEffect:
		s += " (runs at import)"
	default:
		s += " (a ken entry point)"
	}
	return s
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// notBuilt are tracked paths that are in no package's build: prose, CI, scripts and
// fixtures. A path outside every rule in Classify, and not listed here, is in the zone.
var notBuilt = struct{ prefixes, files []string }{
	prefixes: []string{"docs/", "scripts/"},
	files:    []string{"LICENSE", "NOTICE", "results.tsv", "go.work", "go.work.sum"},
}

// Classify reports whether a change to path, in this tree, could change retrieval
// results or the retrieval API, and why. go.mod is not handled here (see Mark): its
// answer depends on what changed in it.
func (a *Analysis) Classify(p string) (zone bool, why string) {
	switch {
	case ignoredByGo(p):
		return false, "the go tool ignores paths starting with . or _"
	case a.inNested(p):
		return false, "in a nested module, which has its own tags"
	case p == "go.sum":
		return false, "a consumer's build uses its own go.sum"
	case inTestdata(p):
		return false, "test fixture"
	case strings.HasSuffix(p, "_test.go"):
		return false, "test file"
	case strings.HasSuffix(p, ".md"):
		return false, "prose"
	}
	for _, pre := range notBuilt.prefixes {
		if strings.HasPrefix(p, pre) {
			return false, "not part of any package"
		}
	}
	for _, f := range notBuilt.files {
		if p == f {
			return false, "not part of any package"
		}
	}
	dir := path.Dir(p)
	if dir == "." {
		dir = ""
	}
	pk := a.pkgs[dir]
	if pk == nil {
		return true, "unclassified: not in a package this tool parsed"
	}
	if a.zone[dir] {
		return true, "in zone package " + dir
	}
	switch {
	case strings.HasSuffix(p, ".go"):
		if !pk.goFiles[p] {
			return true, "unclassified: a Go file this tool did not parse"
		}
		if w, ok := a.reached[p]; ok {
			return true, "reached: " + w
		}
		return false, "nothing in the zone reaches it"
	case strings.HasSuffix(p, ".s"):
		if w, ok := a.reached[p]; ok {
			return true, "reached: " + w
		}
		return false, "nothing in the zone reaches it"
	}
	// A header, or any other file beside the sources: in the zone if the zone links
	// the package at all.
	if a.linked[dir] {
		return true, "unclassified file in " + dir + ", a package the zone links"
	}
	return false, "in " + dir + ", a package the zone does not link"
}

// Reached lists the files of one package directory that are in the zone, sorted.
func (a *Analysis) Reached(dir string) (in, out []string) {
	pk := a.pkgs[dir]
	if pk == nil {
		return nil, nil
	}
	var all []string
	for f := range pk.goFiles {
		all = append(all, f)
	}
	for _, af := range pk.asm {
		all = append(all, af.path)
	}
	sort.Strings(all)
	for _, f := range all {
		if z, _ := a.Classify(f); z {
			in = append(in, f)
		} else {
			out = append(out, f)
		}
	}
	return in, out
}

// Linked lists the non-zone packages the zone links, sorted.
func (a *Analysis) Linked() []string {
	var out []string
	for _, dir := range sortedKeys(a.linked) {
		if !a.zone[dir] {
			out = append(out, dir)
		}
	}
	return out
}

// HasPackage reports whether the tree has a root-module package at dir.
func (a *Analysis) HasPackage(dir string) bool {
	return a.pkgs[dir] != nil && len(a.pkgs[dir].goFiles) > 0
}

// HasSymbol reports whether package dir declares the package-level name.
func (a *Analysis) HasSymbol(dir, name string) bool {
	return a.pkgs[dir] != nil && len(a.pkgs[dir].byName[name]) > 0
}
