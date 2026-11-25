package main

import (
	"go/build"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// plant writes files under dir and makes it a git checkout with them committed.
func plant(t *testing.T, dir string, files map[string]string) {
	t.Helper()
	for p, body := range files {
		f := filepath.Join(dir, p)
		if err := os.MkdirAll(filepath.Dir(f), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(f, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, args := range [][]string{{"init", "-q"}, {"add", "."}, {"-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"}} {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
}

// TestCensus_planted runs the census over a planted library and consumer and checks each way of reaching a name, the
// production/test split, a name declared on one architecture only, a tag-gated consumer file, and the categories.
func TestCensus_planted(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("no git")
	}
	tmp := t.TempDir()
	lib, app := filepath.Join(tmp, "lib"), filepath.Join(tmp, "app")
	plant(t, lib, map[string]string{
		"go.mod": "module example.com/lib\n\ngo 1.27\n",
		"la/la.go": `package la

func Used() int { return helper() }
func helper() int { return Internal() }
func Internal() int { return 1 }
func TestOnly() int { return 2 }
func Nobody() int { return 3 }
func Tagged() int { return 4 }

type Kernels struct{ Names []string; Hidden int }
func Active() Kernels { return Kernels{} }

type Closer struct{}
func (Closer) Close() error { return nil }
func (Closer) Other() {}

const Pi = 3
var Width = 4
`,
		"la/arm_arm64.go": "package la\n\nfunc ArmOnly() int { return 5 }\n",
		"la/la_test.go":   "package la\n\nimport \"testing\"\n\nfunc TestNobody(t *testing.T) { _ = Width }\n",
	})
	plant(t, app, map[string]string{
		"go.mod": "module example.com/app\n\ngo 1.27\n\nrequire example.com/lib v0.0.0\n",
		"app.go": `package app

import (
	"io"

	"example.com/lib/la"
)

type closer interface{ Close() error }

func Run() (int, io.Closer) {
	k := la.Active()
	_ = k.Names
	return la.Used() + la.Pi, la.Closer{}
}
`,
		"tagged.go": "//go:build special\n\npackage app\n\nimport \"example.com/lib/la\"\n\nfunc T() int { return la.Tagged() }\n",
		"arm.go":    "//go:build arm64\n\npackage app\n\nimport \"example.com/lib/la\"\n\nfunc A() int { return la.ArmOnly() }\n",
		"app_test.go": `package app

import (
	"testing"

	"example.com/lib/la"
)

func TestX(t *testing.T) { _ = la.TestOnly() }
`,
	})
	// An untracked package that uses Nobody: built by the go tool, not counted.
	if err := os.MkdirAll(filepath.Join(app, "stray"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(app, "stray", "s.go"), []byte("package stray\n\nimport \"example.com/lib/la\"\n\nvar _ = la.Nobody()\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	libRepo, err := openRepo("aikit", lib, true)
	if err != nil {
		t.Fatal(err)
	}
	appRepo, err := openRepo("goinfer", app, false)
	if err != nil {
		t.Fatal(err)
	}
	devnull, _ := os.Open(os.DevNull)
	defer devnull.Close()
	t.Setenv("GOPROXY", "off")
	res, err := censusOf([]*repo{libRepo, appRepo}, libRepo.root, "la", [][2]string{{"linux", "amd64"}, {"linux", "arm64"}}, devnull)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.strays) != 1 || res.rel(res.strays[0]) != "goinfer/stray/s.go" {
		t.Errorf("strays = %v, want the untracked goinfer/stray/s.go", res.strays)
	}
	if len(res.incon) > 0 {
		t.Fatalf("inconclusive: %v", res.incon)
	}
	// special must be chosen for tagged.go, arm64 for arm.go.
	var cfgs []string
	for _, c := range res.configs {
		cfgs = append(cfgs, c.String())
	}
	if !slices.ContainsFunc(cfgs, func(s string) bool { return strings.Contains(s, "special") }) ||
		!slices.ContainsFunc(cfgs, func(s string) bool { return strings.HasPrefix(s, "linux/arm64") }) {
		t.Errorf("configurations %v do not cover the special tag and arm64", cfgs)
	}
	bc := res.byCategory()
	want := map[string][]string{
		catProd:      {"Active", "Closer", "Kernels", "Kernels.Names", "Pi", "Used", "ArmOnly", "Tagged"},
		catTests:     {"TestOnly"},
		catInside:    {"Internal"},
		catInsideTst: {"Width"},
		catNothing:   {"Closer.Close", "Closer.Other", "Kernels.Hidden", "Nobody"},
	}
	for c, names := range want {
		slices.Sort(names)
		if got := bc[c]; !slices.Equal(got, names) {
			t.Errorf("%s = %v, want %v", c, got, names)
		}
	}
	rs := res.reaches()
	if got := rs["Kernels"].cell("goinfer"); got != "1v" {
		t.Errorf("Kernels in goinfer = %q, want 1v (reached through Active's result, never named)", got)
	}
	if got := rs["Closer.Close"].cell("goinfer"); got != "1i" {
		t.Errorf("Closer.Close in goinfer = %q, want 1i (the closer interface)", got)
	}
	if got := res.platforms(res.c.names["ArmOnly"]); got != "linux/arm64" {
		t.Errorf("ArmOnly platforms = %q, want linux/arm64", got)
	}
	if res.c.names["Nobody"].Decl != "la/la.go" {
		t.Errorf("Nobody declared in %q", res.c.names["Nobody"].Decl)
	}
}

// TestFileRule_matchesGoBuild holds readRule's decision equal to go/build's MatchFile on every Go file of this
// repository, for a few configurations.
func TestFileRule_matchesGoBuild(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	var files []string
	for _, f := range gitFiles(root) {
		if strings.HasSuffix(f, ".go") {
			files = append(files, filepath.Join(root, f))
		}
	}
	if len(files) < 100 {
		t.Fatalf("only %d Go files found under %s", len(files), root)
	}
	for _, c := range []config{
		{goos: "linux", goarch: "amd64"},
		{goos: "darwin", goarch: "arm64", tags: []string{"aikit_nodotprod"}},
		{goos: "windows", goarch: "arm64", tags: []string{"aikit_checks", "race"}},
		{goos: "linux", goarch: "amd64", experiments: []string{"simd"}},
	} {
		ctx := build.Default
		ctx.GOOS, ctx.GOARCH, ctx.CgoEnabled, ctx.BuildTags = c.goos, c.goarch, true, c.tags
		ctx.ToolTags = nil
		for _, e := range c.experiments {
			ctx.ToolTags = append(ctx.ToolTags, "goexperiment."+e)
		}
		tag := c.tagFunc()
		for _, f := range files {
			want, err := ctx.MatchFile(filepath.Dir(f), filepath.Base(f))
			if err != nil {
				continue
			}
			if got := readRule(f).builds(tag); got != want {
				t.Errorf("%s: %s: rule says %v, go/build says %v", c, f, got, want)
			}
		}
	}
}

// TestCensus_uncleanImporterIsInconclusive: a package that fails to type-check in one configuration but not another is
// fine; a file that imports the target and never checks cleanly makes the census inconclusive and is named.
func TestCensus_uncleanImporterIsInconclusive(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("no git")
	}
	tmp := t.TempDir()
	lib, app := filepath.Join(tmp, "lib"), filepath.Join(tmp, "app")
	plant(t, lib, map[string]string{"go.mod": "module example.com/lib\n\ngo 1.27\n", "la/la.go": "package la\n\nfunc F() int { return 1 }\n"})
	plant(t, app, map[string]string{
		"go.mod": "module example.com/app\n\ngo 1.27\n\nrequire example.com/lib v0.0.0\n",
		// Fails only on arm64: the package builds on amd64, so this is "does not build there", not a census failure.
		"ok/a.go":       "package ok\n\nimport \"example.com/lib/la\"\n\nfunc A() int { return la.F() }\n",
		"ok/b_arm64.go": "package ok\n\nvar _ int = \"not an int\"\n",
		// Fails everywhere and imports the target.
		"bad/bad.go": "package bad\n\nimport \"example.com/lib/la\"\n\nvar _ string = la.F()\n",
	})
	libRepo, err := openRepo("aikit", lib, true)
	if err != nil {
		t.Fatal(err)
	}
	appRepo, err := openRepo("goinfer", app, false)
	if err != nil {
		t.Fatal(err)
	}
	devnull, _ := os.Open(os.DevNull)
	defer devnull.Close()
	t.Setenv("GOPROXY", "off")
	res, err := censusOf([]*repo{libRepo, appRepo}, libRepo.root, "la", [][2]string{{"linux", "amd64"}, {"linux", "arm64"}}, devnull)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, f := range res.unclean {
		got = append(got, res.rel(f))
	}
	if !slices.Equal(got, []string{"goinfer/bad/bad.go"}) {
		t.Errorf("unclean = %v, want [goinfer/bad/bad.go]", got)
	}
	if len(res.incon) != 1 {
		t.Errorf("incon = %v, want one reason", res.incon)
	}
}
