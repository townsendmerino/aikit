// Command linalgcensus answers, by the type checker: which of linalg's exported names does each consumer reach? A name
// is a package-level function, type, constant or variable, an exported method of an exported type, or an exported field
// of an exported struct type. The consumers are linalg itself, its tests, aikit's other root-module packages, goinfer,
// ken and tack, each split into production and test files.
//
// HOW. It writes a go.work that uses aikit's root module from this tree and every module of each consumer checkout, so
// every consumer is compiled against this linalg. It lists every Go file of those modules and picks build
// configurations (GOOS/GOARCH, build tags, GOEXPERIMENTs) until each file that any candidate builds is built by one:
// the candidates are each platform with no tags, every tag, and each tag alone, where the tags are those the files'
// //go:build lines name. In each configuration `go list -deps -test` gives the package graph, and go/types checks it
// from source: packages outside the checkouts without bodies, ours in full.
//
// WHAT COUNTS AS REACHING. "named": an identifier resolves to the name (go/types' Uses), which covers calls, selectors,
// method values and expressions, type names, embedded fields and composite-literal keys. "value": an expression has the
// type though nothing names it (a type reached only through a return value). "iface": an interface declared in the file
// is satisfied by a linalg type, so the interface's methods may dispatch to it; this over-approximates on purpose.
//
// WHAT IT DOES NOT SEE. go:linkname, reflection, and aikit's nested modules (they build against a released aikit).
// A file no candidate builds (an `ignore` tag, a contradiction) is listed, not checked.
//
// Usage:
//
//	go run -C tools ./linalgcensus --goinfer ~/goinfer --ken ~/ken --tack ~/tack --md census.md --json census.json
//
// A package that fails to type-check in a configuration does not build there (a CUDA package off linux); another
// configuration checks it. It ends with one VERDICT line: PASS when every consumer was given, every configuration
// listed, and every file that imports the package type-checked cleanly in at least one configuration; INCONCLUSIVE
// otherwise, with the reason. Exit status 0 on PASS, 2 otherwise.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"go/version"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/townsendmerino/aikit/tools/gpumod"
	"github.com/townsendmerino/aikit/tools/retrievalzone"
)

// repo is one checkout the census reads.
type repo struct {
	name    string // column prefix: aikit, goinfer, ken, tack
	root    string // absolute, symlinks resolved
	modules []string
	commit  string
}

const defaultPlatforms = "linux/amd64,linux/arm64,darwin/arm64,darwin/amd64,windows/amd64,windows/arm64"

func main() { os.Exit(run(os.Args[1:])) }

func run(args []string) int {
	fs := flag.NewFlagSet("linalgcensus", flag.ContinueOnError)
	aikitDir := fs.String("aikit", "", "aikit checkout (default: this repository)")
	pkgRel := fs.String("pkg", "linalg", "the package to census, relative to aikit's root module")
	goinfer := fs.String("goinfer", "", "goinfer checkout")
	ken := fs.String("ken", "", "ken checkout")
	tack := fs.String("tack", "", "tack checkout")
	platforms := fs.String("platforms", defaultPlatforms, "GOOS/GOARCH pairs to choose configurations from")
	mdOut := fs.String("md", "", "write the table as markdown to this file")
	jsonOut := fs.String("json", "", "write every use as JSON to this file")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	var incon []string
	root := *aikitDir
	if root == "" {
		r, err := gpumod.RepoRoot()
		if err != nil {
			fmt.Fprintln(os.Stderr, "linalgcensus: cannot locate the repo root:", err)
			return 2
		}
		root = r
	}
	aikit, err := openRepo("aikit", root, true)
	if err != nil {
		fmt.Fprintln(os.Stderr, "linalgcensus:", err)
		return 2
	}
	repos := []*repo{aikit}
	for _, c := range []struct{ name, dir string }{{"goinfer", *goinfer}, {"ken", *ken}, {"tack", *tack}} {
		if c.dir == "" {
			incon = append(incon, "no --"+c.name+" checkout given")
			continue
		}
		r, err := openRepo(c.name, c.dir, false)
		if err != nil {
			fmt.Fprintln(os.Stderr, "linalgcensus:", err)
			return 2
		}
		repos = append(repos, r)
	}
	var plats [][2]string
	for p := range strings.SplitSeq(*platforms, ",") {
		goos, goarch, ok := strings.Cut(strings.TrimSpace(p), "/")
		if !ok {
			fmt.Fprintln(os.Stderr, "linalgcensus: bad platform", p)
			return 2
		}
		plats = append(plats, [2]string{goos, goarch})
	}
	res, err := censusOf(repos, aikit.root, *pkgRel, plats, os.Stdout)
	if err != nil {
		fmt.Fprintln(os.Stderr, "linalgcensus:", err)
		fmt.Println("VERDICT: INCONCLUSIVE —", err)
		return 2
	}
	incon = append(incon, res.incon...)
	if zone, err := retrievalzone.WorkTree(aikit.root); err == nil {
		if a, err := retrievalzone.Analyze(zone); err == nil {
			res.zone = func(f string) bool { z, _ := a.Classify(f); return z }
		}
	}
	if res.zone == nil {
		incon = append(incon, "the retrieval-zone analysis failed")
	}
	if *mdOut != "" {
		if err := os.WriteFile(*mdOut, []byte(res.markdown(repos)), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "linalgcensus:", err)
			return 2
		}
	}
	if *jsonOut != "" {
		b, _ := json.MarshalIndent(res.jsonUses(), "", " ")
		if err := os.WriteFile(*jsonOut, b, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "linalgcensus:", err)
			return 2
		}
	}
	fmt.Print(res.summary())
	if len(incon) > 0 {
		fmt.Println("VERDICT: INCONCLUSIVE —", strings.Join(incon, "; "))
		return 2
	}
	fmt.Printf("VERDICT: PASS — %d exported names of %s across %d configurations, every file importing it type-checked cleanly\n", len(res.c.names), *pkgRel, len(res.configs))
	return 0
}

func openRepo(name, dir string, rootOnly bool) (*repo, error) {
	abs, err := filepath.Abs(dir)
	if err == nil {
		abs, err = filepath.EvalSymlinks(abs)
	}
	if err != nil {
		return nil, fmt.Errorf("%s checkout %s: %v", name, dir, err)
	}
	r := &repo{name: name, root: abs}
	out, err := exec.Command("git", "-C", abs, "rev-parse", "--short=9", "HEAD").Output()
	if err != nil {
		return nil, fmt.Errorf("%s checkout %s is not a git checkout: %v", name, abs, err)
	}
	r.commit = strings.TrimSpace(string(out))
	if rootOnly {
		r.modules = []string{abs}
		return r, nil
	}
	for _, f := range gitFiles(abs) {
		if filepath.Base(f) == "go.mod" && !skippedPath(f) {
			r.modules = append(r.modules, filepath.Join(abs, filepath.Dir(f)))
		}
	}
	if len(r.modules) == 0 {
		return nil, fmt.Errorf("%s checkout %s has no go.mod", name, abs)
	}
	return r, nil
}

func gitFiles(root string) []string {
	out, err := exec.Command("git", "-C", root, "ls-files", "-z").Output()
	if err != nil {
		return nil
	}
	var fs []string
	for f := range strings.SplitSeq(string(out), "\x00") {
		if f != "" {
			fs = append(fs, f)
		}
	}
	return fs
}

// skippedPath reports a path the go tool ignores: under testdata, or under a directory starting with . or _.
func skippedPath(rel string) bool {
	for el := range strings.SplitSeq(filepath.ToSlash(rel), "/") {
		if el == "testdata" || strings.HasPrefix(el, ".") || strings.HasPrefix(el, "_") {
			return true
		}
	}
	return false
}

// goFiles lists the Go files of r's used modules, absolute, leaving out any module the workspace does not use.
func (r *repo) goFiles() []string {
	all := gitFiles(r.root)
	var modDirs []string
	for _, f := range all {
		if filepath.Base(f) == "go.mod" {
			modDirs = append(modDirs, filepath.Join(r.root, filepath.Dir(f)))
		}
	}
	var out []string
	for _, f := range all {
		if !strings.HasSuffix(f, ".go") || skippedPath(f) {
			continue
		}
		abs := filepath.Join(r.root, f)
		// The innermost module owns the file; keep it if that module is used.
		owner := ""
		for _, m := range modDirs {
			if (abs == m || strings.HasPrefix(abs, m+string(filepath.Separator))) && len(m) > len(owner) {
				owner = m
			}
		}
		if slices.Contains(r.modules, owner) {
			out = append(out, abs)
		}
	}
	return out
}

type result struct {
	c       *census
	configs []config
	never   []string
	unclean []string // files importing the target that no configuration checked without a type error
	tracked map[string]bool
	strays  []string // Go files the go tool built that git does not track; their uses are not counted
	incon   []string
	repos   []*repo
	target  string // relative dir of the target package
	aikit   string
	zone    func(string) bool
}

func censusOf(repos []*repo, aikitRoot, pkgRel string, plats [][2]string, log *os.File) (*result, error) {
	modPath, err := modulePath(filepath.Join(aikitRoot, "go.mod"))
	if err != nil {
		return nil, err
	}
	target := modPath + "/" + filepath.ToSlash(pkgRel)
	res := &result{c: newCensus(target), repos: repos, target: filepath.ToSlash(pkgRel), aikit: aikitRoot}

	var files, tags, exps []string
	for _, r := range repos {
		for _, f := range r.goFiles() {
			files = append(files, f)
			t, e := constraintTerms(f)
			tags, exps = append(tags, t...), append(exps, e...)
		}
	}
	tags, exps = sortedUnique(tags), sortedUnique(exps)
	res.tracked = map[string]bool{}
	for _, f := range files {
		res.tracked[f] = true
	}
	cands := candidates(plats, tags, exps)
	res.configs, res.never = cover(files, cands)
	fmt.Fprintf(log, "census: %d Go files, tags %v, experiments %v: %d of %d candidate configurations build every buildable file; %d files build in none\n",
		len(files), tags, exps, len(res.configs), len(cands), len(res.never))

	work, err := os.MkdirTemp("", "linalgcensus-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(work)
	var gw strings.Builder
	var mods, patterns []string
	for _, r := range repos {
		mods = append(mods, r.modules...)
	}
	gw.WriteString("go " + workGoVersion(mods) + "\n\nuse (\n")
	for _, m := range mods {
		fmt.Fprintf(&gw, "\t%s\n", m)
		patterns = append(patterns, m+"/...")
	}
	gw.WriteString(")\n")
	if err := os.WriteFile(filepath.Join(work, "go.work"), []byte(gw.String()), 0o644); err != nil {
		return nil, err
	}
	ours := func(dir string) bool {
		for _, r := range repos {
			for _, m := range r.modules {
				if dir == m || strings.HasPrefix(dir, m+string(filepath.Separator)) {
					return true
				}
			}
		}
		return false
	}
	checked, clean, importers := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for i, cfg := range res.configs {
		start := time.Now()
		ps, err := goList(work, append(cfg.env(), "GOWORK="+filepath.Join(work, "go.work")), patterns)
		if err != nil {
			res.incon = append(res.incon, fmt.Sprintf("%s: %v", cfg, err))
			fmt.Fprintf(log, "config %d/%d %s: go list failed: %v\n", i+1, len(res.configs), cfg, err)
			continue
		}
		w := newWorld(cfg, ps, ours)
		w.checkAll()
		var plain *types.Package
		if l := w.pkgs[target]; l != nil && l.pkg != nil {
			plain = l.pkg
			res.c.declare(w, l.pkg, aikitRoot, cfg.String())
		} else {
			res.incon = append(res.incon, fmt.Sprintf("%s: %s was not checked", cfg, target))
		}
		nOurs, nErr := 0, 0
		for _, id := range w.order {
			l := w.pkgs[id]
			if l.isTestMain() || !ours(l.Dir) {
				continue
			}
			nOurs++
			if l.err != nil {
				res.c.errs[id+" @ "+cfg.String()] = []string{l.err.Error()}
				nErr++
				continue
			}
			if len(l.errs) > 0 {
				res.c.errs[id+" @ "+cfg.String()] = l.errs
				nErr++
			}
			for _, af := range l.files {
				f := w.fset.File(af.Pos()).Name()
				if !res.tracked[f] && !slices.Contains(res.strays, f) {
					res.strays = append(res.strays, f)
				}
				checked[f] = true
				if len(l.errs) == 0 {
					clean[f] = true
				}
				if importsPath(af, target) {
					importers[f] = true
				}
			}
			res.c.record(w, l, plain, cfg.String())
		}
		fmt.Fprintf(log, "config %d/%d %s: %d packages listed, %d of ours checked, %d with errors, %.0fs\n",
			i+1, len(res.configs), cfg, len(ps), nOurs, nErr, time.Since(start).Seconds())
	}
	// A package with type errors in a configuration does not build there (a CUDA package off linux, a unix-only one on
	// windows), and another configuration checks it. What must hold is that every file importing the target was checked
	// cleanly somewhere, and that no file importing it went unchecked.
	for _, f := range files {
		if !checked[f] && fileImports(f, target) {
			importers[f] = true
		}
	}
	for f := range importers {
		if !clean[f] {
			res.unclean = append(res.unclean, f)
		}
	}
	slices.Sort(res.unclean)
	if len(res.unclean) > 0 {
		res.incon = append(res.incon, fmt.Sprintf("%d files that import %s never type-checked cleanly (listed above)", len(res.unclean), pkgRel))
	}
	return res, nil
}

func modulePath(gomod string) (string, error) {
	f, err := os.Open(gomod)
	if err != nil {
		return "", err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if p, ok := strings.CutPrefix(strings.TrimSpace(sc.Text()), "module "); ok {
			return strings.Trim(strings.TrimSpace(p), `"`), nil
		}
	}
	return "", fmt.Errorf("%s has no module line", gomod)
}

// workGoVersion is the highest go directive among the modules, which a go.work must be at least.
func workGoVersion(mods []string) string {
	best := "1.21"
	for _, m := range mods {
		f, err := os.Open(filepath.Join(m, "go.mod"))
		if err != nil {
			continue
		}
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			if v, ok := strings.CutPrefix(strings.TrimSpace(sc.Text()), "go "); ok {
				v = strings.TrimSpace(v)
				if version.Compare("go"+v, "go"+best) > 0 {
					best = v
				}
				break
			}
		}
		f.Close()
	}
	return best
}

func importsPath(af *ast.File, path string) bool {
	for _, im := range af.Imports {
		if strings.Trim(im.Path.Value, "`\"") == path {
			return true
		}
	}
	return false
}

func fileImports(f, path string) bool {
	af, err := parser.ParseFile(token.NewFileSet(), f, nil, parser.ImportsOnly)
	return err == nil && importsPath(af, path)
}
