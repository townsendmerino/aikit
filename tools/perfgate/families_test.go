package main

import (
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/tools/gpumod"
)

// sourceBenchmarks lists the top-level Benchmark functions a test build of dir defines for
// goos/goarch, honouring file-name suffixes and //go:build lines the way `go test -c` does.
func sourceBenchmarks(t *testing.T, dir, goos, goarch string) []string {
	t.Helper()
	ctx := build.Default
	ctx.GOOS, ctx.GOARCH = goos, goarch
	ctx.CgoEnabled = false
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	fset := token.NewFileSet()
	for _, e := range ents {
		if e.IsDir() || !strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		ok, err := ctx.MatchFile(dir, e.Name())
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			continue
		}
		f, err := parser.ParseFile(fset, filepath.Join(dir, e.Name()), nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv == nil && strings.HasPrefix(fd.Name.Name, "Benchmark") {
				names = append(names, fd.Name.Name)
			}
		}
	}
	return names
}

// Every default family resolves to exactly one listed package on each arch it claims, and to
// none on an arch it does not claim, so a box never finds a gap in the table at release time and
// the arches column cannot drift narrower than the source.
func TestDefaultFamilies_resolveInSource(t *testing.T) {
	root, err := gpumod.RepoRoot()
	if err != nil {
		t.Fatal(err)
	}
	for _, target := range [][2]string{{"linux", "amd64"}, {"linux", "arm64"}, {"darwin", "arm64"}, {"windows", "amd64"}} {
		goos, goarch := target[0], target[1]
		listed := map[string][]string{}
		for _, pkg := range defaultPackages {
			listed[pkg] = sourceBenchmarks(t, filepath.Join(root, pkg), goos, goarch)
		}
		for _, f := range defaultFamilies {
			res, err := resolveFamilies([]string{f.name}, listed)
			if err != nil {
				t.Errorf("%s/%s: %v", goos, goarch, err)
				continue
			}
			found := len(res.missing) == 0
			switch {
			case f.appliesTo(goarch) && !found:
				t.Errorf("%s/%s: %s is listed for this arch and no package in %v defines it", goos, goarch, f.name, defaultPackages)
			case !f.appliesTo(goarch) && found:
				t.Errorf("%s/%s: %s is defined here but its arches are %v; widen them", goos, goarch, f.name, f.arches)
			}
		}
	}
}

// The kernel families this gate must cover: the K-quant kinds and the f16 attention kernels
// among them.
func TestDefaultFamilies_coverKQuantAndF16(t *testing.T) {
	for _, want := range []string{
		"BenchmarkQ56KvsInt8", "BenchmarkQ4KvsSplitHalf",
		"BenchmarkMatmulQKAcc64F16", "BenchmarkMatmulAVAcc64F16",
	} {
		if !slices.ContainsFunc(defaultFamilies, func(f family) bool { return f.name == want }) {
			t.Errorf("%s is not a default family", want)
		}
	}
}

func TestResolveFamilies(t *testing.T) {
	// A family that moved package across the tag boundary resolves in each arm by name.
	prev, err := resolveFamilies([]string{"BenchmarkA", "BenchmarkB"},
		map[string][]string{"./linalg": {"BenchmarkA", "BenchmarkB", "BenchmarkOther"}})
	if err != nil {
		t.Fatal(err)
	}
	cur, err := resolveFamilies([]string{"BenchmarkA", "BenchmarkB"},
		map[string][]string{"./linalg": {"BenchmarkA"}, "./linalg/attn": {"BenchmarkB"}})
	if err != nil {
		t.Fatal(err)
	}
	if got := prev.byPkg["./linalg"]; !slices.Equal(got, []string{"BenchmarkA", "BenchmarkB"}) {
		t.Errorf("prev ./linalg = %v", got)
	}
	if got := cur.byPkg["./linalg/attn"]; !slices.Equal(got, []string{"BenchmarkB"}) {
		t.Errorf("cur ./linalg/attn = %v", got)
	}
	if len(prev.missing)+len(cur.missing) != 0 {
		t.Errorf("missing: prev %v cur %v", prev.missing, cur.missing)
	}

	// A family no package defines is reported missing.
	res, err := resolveFamilies([]string{"BenchmarkA", "BenchmarkGone"}, map[string][]string{"./linalg": {"BenchmarkA"}})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(res.missing, []string{"BenchmarkGone"}) {
		t.Errorf("missing = %v, want [BenchmarkGone]", res.missing)
	}

	// A family two packages define is an error.
	if _, err := resolveFamilies([]string{"BenchmarkA"},
		map[string][]string{"./linalg": {"BenchmarkA"}, "./linalg/attn": {"BenchmarkA"}}); err == nil {
		t.Error("a family defined twice resolved without an error")
	}
}

func TestFamilyNames_byArch(t *testing.T) {
	fams := []family{{name: "BenchmarkAll"}, {name: "BenchmarkX86", arches: amd64Only}, {name: "BenchmarkArm", arches: arm64Only}}
	if got := familyNames(fams, "amd64"); !slices.Equal(got, []string{"BenchmarkAll", "BenchmarkX86"}) {
		t.Errorf("amd64: %v", got)
	}
	if got := familyNames(fams, "arm64"); !slices.Equal(got, []string{"BenchmarkAll", "BenchmarkArm"}) {
		t.Errorf("arm64: %v", got)
	}
}

func TestParseFamilies(t *testing.T) {
	got, err := parseFamilies("BenchmarkA, BenchmarkB")
	if err != nil || len(got) != 2 || got[0].name != "BenchmarkA" || got[1].name != "BenchmarkB" || got[1].arches != nil {
		t.Errorf("parseFamilies = %v, %v", got, err)
	}
	for _, bad := range []string{"", " , ", "^(BenchmarkA|BenchmarkB)$", "TestA"} {
		if _, err := parseFamilies(bad); err == nil {
			t.Errorf("parseFamilies(%q) accepted a value that names no benchmark function", bad)
		}
	}
}

func TestBenchRegexAndTestList(t *testing.T) {
	if got, want := benchRegex([]string{"BenchmarkA", "BenchmarkB_x"}), "^(BenchmarkA|BenchmarkB_x)$"; got != want {
		t.Errorf("benchRegex = %q, want %q", got, want)
	}
	names := parseTestList("TestFoo\nBenchmarkA\nBenchmarkB\nExampleC\n")
	if !slices.Equal(names, []string{"BenchmarkA", "BenchmarkB"}) {
		t.Errorf("parseTestList = %v", names)
	}
}

func TestSilentFamilies(t *testing.T) {
	visits := []map[string]float64{{"BenchmarkA/K1_N1": 1, "BenchmarkAB/x": 1}}
	if got := silentFamilies([]string{"BenchmarkA", "BenchmarkB"}, visits); !slices.Equal(got, []string{"BenchmarkB"}) {
		t.Errorf("silentFamilies = %v, want [BenchmarkB]", got)
	}
	if got := silentFamilies([]string{"BenchmarkA"}, []map[string]float64{{"BenchmarkAB/x": 1}}); !slices.Equal(got, []string{"BenchmarkA"}) {
		t.Errorf("a family whose name prefixes another's took that one's rows: %v", got)
	}
}

// writePlanted writes a module at dir with one package per entry of pkgs (path → benchmark
// names), each benchmark an empty loop.
func writePlanted(t *testing.T, dir string, pkgs map[string][]string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module planted\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for pkg, benches := range pkgs {
		pdir := filepath.Join(dir, pkg)
		if err := os.MkdirAll(pdir, 0o755); err != nil {
			t.Fatal(err)
		}
		name := filepath.Base(pdir)
		src := "package " + name + "\n\nimport \"testing\"\n"
		for _, b := range benches {
			src += "\nfunc " + b + "(b *testing.B) {\n\tfor b.Loop() {\n\t}\n}\n"
		}
		if err := os.WriteFile(filepath.Join(pdir, name+"_test.go"), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// Through buildArm on planted packages: a listed family no package defines fails the gate
// (exit 6), a family is found in whichever package defines it, and a package the previous tag
// does not have is skipped in that arm.
func TestBuildArm_missingFamilyFailsTheGate(t *testing.T) {
	dir := t.TempDir()
	writePlanted(t, dir, map[string][]string{"a": {"BenchmarkA", "BenchmarkOther"}, "b": {"BenchmarkB"}})
	pkgs := []string{"./a", "./b", "./c"}

	cur, cleanup, err := buildArm(dir, pkgs, "test-cur", []string{"BenchmarkA", "BenchmarkB", "BenchmarkGone"}, true)
	defer cleanup()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(cur.res.missing, []string{"BenchmarkGone"}) {
		t.Fatalf("missing = %v, want [BenchmarkGone]", cur.res.missing)
	}
	line, rc := missingVerdict(config{}, cur.res.missing, "amd64")
	if rc != 6 || !strings.Contains(line, "VERDICT: FAIL") || !strings.Contains(line, "BenchmarkGone") {
		t.Errorf("missingVerdict = %q, %d; want a FAIL naming BenchmarkGone, exit 6", line, rc)
	}
	if len(cur.runs) != 2 || cur.runs[0].bench != "^(BenchmarkA)$" || cur.runs[1].bench != "^(BenchmarkB)$" {
		t.Errorf("runs = %+v, want one per package naming only its family", cur.runs)
	}
	rows, err := runVisit(cur, config{benchtime: "1x"})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := rows["BenchmarkA"]; !ok || len(rows) != 2 {
		t.Errorf("rows = %v, want BenchmarkA and BenchmarkB only", rows)
	}

	if _, rc := missingVerdict(config{}, nil, "amd64"); rc != 0 {
		t.Errorf("nothing missing: rc = %d, want 0", rc)
	}
	// The working tree does not skip an absent package: it fails to build.
	if _, cleanup, err := buildArm(dir, pkgs, "test-cur2", []string{"BenchmarkA"}, false); err == nil {
		t.Error("an absent package in the working tree built without an error")
	} else {
		cleanup()
	}
}
