package retrievalzone

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/tools/gpumod"
)

func realTree(t *testing.T) (string, *Analysis) {
	t.Helper()
	root, err := gpumod.RepoRoot()
	if err != nil {
		t.Fatalf("repo root: %v", err)
	}
	src, err := WorkTree(root)
	if err != nil {
		t.Fatalf("WorkTree: %v", err)
	}
	a, err := Analyze(src)
	if err != nil {
		t.Fatalf("the real tree does not analyze, so every release would be marked yes: %v", err)
	}
	return root, a
}

// The zone names real things: a package or symbol that is gone is a stale list.
func TestZoneNamesExist(t *testing.T) {
	_, a := realTree(t)
	for _, p := range Packages {
		if !a.HasPackage(p) {
			t.Errorf("Packages lists %q, which the root module does not have", p)
		}
	}
	for p, syms := range Symbols {
		for _, s := range syms {
			if !a.HasSymbol(p, s) {
				t.Errorf("Symbols lists %s.%s, which %s does not declare", p, s, p)
			}
		}
	}
}

// docs/architecture.md lists the zone for people; this list is the one the gate uses.
func TestArchitectureDocListsTheZone(t *testing.T) {
	root, _ := realTree(t)
	doc, err := os.ReadFile(filepath.Join(root, "docs", "architecture.md"))
	if err != nil {
		t.Fatal(err)
	}
	_, sec, ok := strings.Cut(strings.ReplaceAll(string(doc), "\r\n", "\n"), "\n## Retrieval zone\n")
	if !ok {
		t.Fatal("docs/architecture.md has no '## Retrieval zone' section")
	}
	if i := strings.Index(sec, "\n## "); i >= 0 {
		sec = sec[:i]
	}
	for _, p := range Packages {
		if !regexp.MustCompile("`" + regexp.QuoteMeta(p) + "`").MatchString(sec) {
			t.Errorf("docs/architecture.md § Retrieval zone does not name zone package `%s`", p)
		}
	}
	for p, syms := range Symbols {
		for _, s := range syms {
			if !strings.Contains(sec, "`"+p+"."+s+"`") {
				t.Errorf("docs/architecture.md § Retrieval zone does not name entry point `%s.%s`", p, s)
			}
		}
	}
}

// The compiler's check on the walk. Copy the zone's packages into an empty module,
// leaving out every file the walk calls unreached, and type-check the result for each
// platform and build tag the core is built with. If the walk missed a name, something
// the zone uses is undefined and this fails.
//
// It proves the reached set is closed under names. It does not prove the other two
// rules (init functions, methods of a reached type); those have their own tests above.
func TestZoneBuildsWithoutUnreachedFiles(t *testing.T) {
	root, a := realTree(t)
	tmp := t.TempDir()
	copyFile := func(rel string) {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			t.Fatal(err)
		}
		dst := filepath.Join(tmp, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dst, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	copyFile("go.mod")
	copyFile("go.sum")
	kept, dropped := 0, 0
	for dir := range a.linked {
		entries, err := os.ReadDir(filepath.Join(root, filepath.FromSlash(dir)))
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range entries {
			name := e.Name()
			rel := dir + "/" + name
			switch {
			case e.IsDir() || strings.HasSuffix(name, "_test.go"):
			case strings.HasSuffix(name, ".h"):
				copyFile(rel)
			case strings.HasSuffix(name, ".go") || strings.HasSuffix(name, ".s"):
				if zone, _ := a.Classify(rel); zone {
					copyFile(rel)
					kept++
				} else {
					dropped++
				}
			}
		}
	}
	if dropped == 0 {
		t.Fatal("no file was left out: the test would pass whatever the walk did")
	}
	t.Logf("kept %d files, left out %d", kept, dropped)

	configs := []struct{ goos, goarch, tags, experiment string }{
		{"linux", "amd64", "", ""}, {"linux", "arm64", "", ""},
		{"windows", "amd64", "", ""}, {"windows", "arm64", "", ""},
		{"darwin", "arm64", "", ""}, {"darwin", "amd64", "", ""},
		{"linux", "amd64", "aikit_checks", ""},
		{"linux", "amd64", "aikit_noavx512", ""}, {"linux", "amd64", "aikit_noavx2", ""},
		{"linux", "amd64", "aikit_nopopcnt", ""}, {"linux", "arm64", "aikit_nodotprod", ""},
		{"linux", "amd64", "", "simd"}, {"linux", "arm64", "", "simd"},
		{"linux", "riscv64", "", ""}, // a GOARCH with no assembly: the pure-Go fallbacks
	}
	for _, c := range configs {
		name := c.goos + "/" + c.goarch
		if c.tags != "" {
			name += " -tags " + c.tags
		}
		if c.experiment != "" {
			name += " GOEXPERIMENT=" + c.experiment
		}
		cmd := exec.Command("go", "vet", "-tags", c.tags, "./...")
		cmd.Dir = tmp
		cmd.Env = append(os.Environ(), "GOOS="+c.goos, "GOARCH="+c.goarch, "CGO_ENABLED=0",
			"GOWORK=off", "GOFLAGS=-mod=mod", "GOEXPERIMENT="+c.experiment)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Errorf("%s: the zone does not type-check without the files the walk left out: %v\n%s", name, err, out)
		}
	}
}
