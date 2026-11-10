package importrules

import (
	"bytes"
	"os"
	"os/exec"
	"regexp"
	"slices"
	"sort"
	"strings"
	"testing"
)

// ext is one external (non-stdlib, non-aikit) package a core package imports. Empty
// goos/goarch mean "every platform"; a set value means the import exists only there,
// because the importing file carries that build constraint.
type ext struct {
	path   string
	goos   string
	goarch string
}

func (e ext) on(p platform) bool {
	return (e.goos == "" || e.goos == p.goos) && (e.goarch == "" || e.goarch == p.goarch)
}

// rule is what one package imports beyond the stdlib: aikit packages (paths relative
// to the module root) and external packages.
type rule struct {
	aikit []string
	ext   []ext
}

// rules is the core module's import graph, one entry per package, non-test imports
// only. It is EXACT, not an upper bound: an import missing from here fails the test,
// and so does an entry here that no package has any more.
//
// Changing it is a design decision, not a chore. A new edge deepens the DAG that
// docs/architecture.md § Package DAG draws, so change that diagram and its prose in
// the same commit.
var rules = map[string]rule{
	// Leaves: stdlib only.
	"topk":                 {},
	"fuse":                 {},
	"chunk":                {},
	"internal/accum":       {},
	"internal/cursor":      {},
	"internal/importrules": {},

	// mmap is a leaf among aikit packages. Its one external import is madvise on darwin.
	"mmap": {ext: []ext{{path: "golang.org/x/sys/unix", goos: "darwin"}}},

	// Kernels and loaders.
	"linalg": {aikit: []string{"mmap"}},
	"embed": {
		aikit: []string{"internal/cursor", "mmap"},
		ext: []ext{
			{path: "golang.org/x/text/unicode/norm"},
			{path: "golang.org/x/sys/cpu", goarch: "amd64"},
		},
	},

	// Chunkers.
	"chunk/markdown": {aikit: []string{"chunk"}},
	"chunk/regex":    {aikit: []string{"chunk"}},

	// Indexes and rerankers.
	"ann":    {aikit: []string{"internal/cursor", "linalg", "mmap", "topk"}},
	"bm25":   {aikit: []string{"internal/accum", "internal/cursor", "mmap", "topk"}},
	"sparse": {aikit: []string{"internal/accum", "topk"}},
	"late":   {aikit: []string{"linalg", "topk"}},
	"hybrid": {aikit: []string{"ann", "bm25", "fuse"}},

	// Model towers.
	"encoder": {aikit: []string{"embed", "linalg", "sparse"}},
	"vision":  {aikit: []string{"embed", "linalg"}},
	"audio":   {aikit: []string{"embed", "linalg"}},

	// Harness and programs.
	"bench":                 {aikit: []string{"ann"}},
	"bench/cmd/benchreport": {aikit: []string{"bench"}},
	"examples/colbert":      {aikit: []string{"ann", "bm25", "chunk", "chunk/regex", "embed", "encoder", "fuse", "late"}},
	"examples/rag":          {aikit: []string{"ann", "bm25", "chunk", "chunk/regex", "embed", "encoder", "fuse", "topk"}},
	"examples/splade":       {aikit: []string{"chunk", "chunk/regex", "encoder", "sparse"}},
	"examples/vision":       {aikit: []string{"ann", "bm25", "chunk", "chunk/regex", "embed", "fuse", "vision"}},
}

// quarantined are the nested modules the core must never reach, directly or through
// a dependency (docs/architecture.md § Quarantines). Paths relative to the module root.
var quarantined = []string{"gpu", "chunk/treesitter"}

// quarantinedNames mirrors CI's "no cgo deps in core graph" grep, so an out-of-repo
// copy of either dependency is caught by name as well.
var quarantinedNames = []string{"webgpu", "gotreesitter"}

type platform struct{ goos, goarch string }

func (p platform) String() string { return p.goos + "/" + p.goarch }

// platforms is every GOOS/GOARCH pair CI builds the core module on (ci.yml: the
// ubuntu-latest, ubuntu-24.04-arm, windows-2025 and windows-11-arm jobs), plus darwin,
// where development happens and where mmap's one external import lives. Build
// constraints change the graph, so the rules are checked on each.
var platforms = []platform{
	{"linux", "amd64"},
	{"linux", "arm64"},
	{"windows", "amd64"},
	{"windows", "arm64"},
	{"darwin", "arm64"},
	{"darwin", "amd64"},
}

// pkg is one line of `go list -deps` output.
type pkg struct {
	path       string
	standard   bool
	mainModule bool
	modulePath string
	incomplete bool
	imports    []string
}

// goList returns every package in the core module and everything they depend on, as
// the go command resolves them for p. It runs with the workspace off and cgo off, the
// way the core is built in CI, and with -e so a broken import is reported by this
// test, by edge, instead of as a go list failure.
func goList(t *testing.T, p platform) []pkg {
	t.Helper()
	const format = `{{.ImportPath}}|{{.Standard}}|{{if .Module}}{{.Module.Main}}|{{.Module.Path}}{{else}}false|{{end}}|{{.Incomplete}}|{{join .Imports ","}}`
	cmd := exec.Command("go", "list", "-e", "-deps", "-f", format, "./...")
	cmd.Dir = "../.." // the module root, from internal/importrules
	cmd.Env = append(os.Environ(),
		"GOOS="+p.goos, "GOARCH="+p.goarch, "CGO_ENABLED=0", "GOWORK=off", "GOFLAGS=")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("%s: go list: %v\n%s", p, err, stderr.String())
	}
	var pkgs []pkg
	for line := range strings.Lines(string(out)) {
		f := strings.Split(strings.TrimSpace(line), "|")
		if len(f) != 6 {
			t.Fatalf("%s: unexpected go list line %q", p, line)
		}
		q := pkg{path: f[0], standard: f[1] == "true", mainModule: f[2] == "true", modulePath: f[3], incomplete: f[4] == "true"}
		if f[5] != "" {
			q.imports = strings.Split(f[5], ",")
		}
		pkgs = append(pkgs, q)
	}
	return pkgs
}

func TestImportRules(t *testing.T) {
	for _, p := range platforms {
		t.Run(p.goos+"_"+p.goarch, func(t *testing.T) { checkPlatform(t, p) })
	}
}

func checkPlatform(t *testing.T, p platform) {
	pkgs := goList(t, p)

	standard := map[string]bool{}
	module := ""
	for _, q := range pkgs {
		standard[q.path] = q.standard
		if q.mainModule {
			module = q.modulePath
		}
	}
	if module == "" {
		t.Fatalf("%s: go list returned no package of the core module", p)
	}
	rel := func(path string) (string, bool) {
		if path == module {
			return ".", true
		}
		return strings.CutPrefix(path, module+"/")
	}
	inQuarantine := func(path string) bool {
		if r, ok := rel(path); ok {
			for _, q := range quarantined {
				if r == q || strings.HasPrefix(r, q+"/") {
					return true
				}
			}
		}
		lower := strings.ToLower(path)
		for _, n := range quarantinedNames {
			if strings.Contains(lower, n) {
				return true
			}
		}
		return false
	}

	// The modules the table lets the core reach. A dependency's own dependencies must
	// stay inside them too (x/text/unicode/norm pulls x/text/transform, for one).
	var extModules []string
	for _, r := range rules {
		for _, e := range r.ext {
			f := strings.Split(e.path, "/")
			if len(f) < 3 {
				t.Fatalf("rules: external import %q is not host/org/module/...", e.path)
			}
			extModules = append(extModules, strings.Join(f[:3], "/"))
		}
	}

	seen := map[string]bool{}
	for _, q := range pkgs {
		if q.standard {
			continue
		}
		if inQuarantine(q.path) {
			t.Errorf("%s: the core module reaches quarantined package %s", p, q.path)
		}
		name, ours := rel(q.path)
		if !q.mainModule {
			if ours {
				// An aikit path outside the core module: a nested module (gpu/*,
				// chunk/treesitter, tools, ...). The edge check below names the importer.
				continue
			}
			if !slices.ContainsFunc(extModules, func(m string) bool { return q.path == m || strings.HasPrefix(q.path, m+"/") }) {
				t.Errorf("%s: the core module reaches %s, which is in no external module the table allows (%s)",
					p, q.path, strings.Join(slices.Compact(slices.Sorted(slices.Values(extModules))), ", "))
			}
			continue
		}

		seen[name] = true
		want, ok := rules[name]
		if !ok {
			t.Errorf("%s: package %s has no entry in the rules table; add one listing exactly what it imports", p, name)
			continue
		}
		var wantExt []string
		for _, e := range want.ext {
			if e.on(p) {
				wantExt = append(wantExt, e.path)
			}
		}
		var gotAikit, gotExt []string
		for _, imp := range q.imports {
			if standard[imp] {
				continue
			}
			if r, ok := rel(imp); ok {
				gotAikit = append(gotAikit, r)
			} else {
				gotExt = append(gotExt, imp)
			}
		}
		diff(t, p, name, "aikit package", gotAikit, want.aikit)
		diff(t, p, name, "external package", gotExt, wantExt)
		if q.incomplete && !t.Failed() {
			t.Errorf("%s: go list reports package %s (or a dependency) as broken; run `GOOS=%s GOARCH=%s go list -deps ./%s`",
				p, name, p.goos, p.goarch, name)
		}
	}

	var stale []string
	for name := range rules {
		if !seen[name] {
			stale = append(stale, name)
		}
	}
	sort.Strings(stale)
	for _, name := range stale {
		t.Errorf("%s: the rules table lists package %s, which the module does not have; delete the entry", p, name)
	}
}

// diff reports every edge that is in the graph but not the table, and every edge
// that is in the table but not the graph, by name.
func diff(t *testing.T, p platform, from, kind string, got, want []string) {
	t.Helper()
	for _, g := range got {
		if !slices.Contains(want, g) {
			t.Errorf("%s: forbidden edge %s -> %s: this %s is not in %s's entry in the rules table (internal/importrules)",
				p, from, g, kind, from)
		}
	}
	for _, w := range want {
		if !slices.Contains(got, w) {
			t.Errorf("%s: stale rule %s -> %s: the table allows this %s but %s no longer imports it; delete it from the entry",
				p, from, w, kind, from)
		}
	}
}

var backticked = regexp.MustCompile("`([^`]+)`")

// The README's package table has a "Deps (beyond stdlib)" column. It went stale the
// same way the architecture doc did, so it is held to the rules table: each row's deps
// must be exactly the package's aikit imports (internal/ packages aside: an importer
// never sees them) plus the external modules it reaches.
func TestREADMEPackageTable(t *testing.T) {
	readme, err := os.ReadFile("../../README.md")
	if err != nil {
		t.Fatal(err)
	}
	// A Windows checkout may store CRLF.
	_, sec, ok := strings.Cut(strings.ReplaceAll(string(readme), "\r\n", "\n"), "\n## Packages\n")
	if !ok {
		t.Fatal("README.md has no '## Packages' section")
	}
	if i := strings.Index(sec, "\n## "); i >= 0 {
		sec = sec[:i]
	}
	rows := 0
	for line := range strings.Lines(sec) {
		cells := strings.Split(strings.Trim(strings.TrimSpace(line), "|"), "|")
		if len(cells) != 3 {
			continue
		}
		name := backticked.FindStringSubmatch(cells[0])
		if name == nil {
			continue
		}
		r, ok := rules[name[1]]
		if !ok {
			continue // a row for something outside the core module (gpu)
		}
		rows++
		var want []string
		for _, a := range r.aikit {
			if !strings.HasPrefix(a, "internal/") {
				want = append(want, a)
			}
		}
		for _, e := range r.ext {
			want = append(want, strings.Join(strings.Split(e.path, "/")[:3], "/"))
		}
		var got []string
		for _, m := range backticked.FindAllStringSubmatch(cells[2], -1) {
			got = append(got, m[1])
		}
		sort.Strings(want)
		sort.Strings(got)
		want, got = slices.Compact(want), slices.Compact(got)
		if !slices.Equal(got, want) {
			t.Errorf("README.md package table, row `%s`: deps column lists %v, the rules table says %v", name[1], got, want)
		}
	}
	if rows < 10 {
		t.Errorf("matched only %d rows of the README package table; the table moved or changed shape", rows)
	}
}
