package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/tools/canary"
	"github.com/townsendmerino/aikit/tools/gate"
)

func TestFirstNonEmpty(t *testing.T) {
	cases := []struct{ in, want string }{
		{"hello", "hello"},
		{"\n\nhello\nworld", "hello"},
		{"  padded  \nworld", "padded"},
		{"\n \n\t\n", ""},
		{"", ""},
		{strings.Repeat("x", 100), strings.Repeat("x", 70)},
	}
	for _, c := range cases {
		if got := firstNonEmpty(c.in); got != c.want {
			t.Errorf("firstNonEmpty(%.20q...) = %q (len %d), want %q (len %d)", c.in, got, len(got), c.want, len(c.want))
		}
	}
}

func writeGoMod(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %q: %v", dir, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module x\n"), 0o644); err != nil {
		t.Fatalf("write go.mod in %q: %v", dir, err)
	}
}

func TestAllModuleDirs_excludesGitAndVenvAndCanaryFixture(t *testing.T) {
	root := t.TempDir()
	writeGoMod(t, root)                                        // "."
	writeGoMod(t, filepath.Join(root, "gpu"))                  // "gpu"
	writeGoMod(t, filepath.Join(root, ".git", "modules", "x")) // must be skipped
	writeGoMod(t, filepath.Join(root, ".venv", "lib"))         // must be skipped
	writeGoMod(t, filepath.Join(root, canary.VulnFixtureDir))  // the deliberate positive control — must be excluded from the real scan

	dirs := allModuleDirs(root)
	want := map[string]bool{".": true, "gpu": true}
	if len(dirs) != len(want) {
		t.Fatalf("allModuleDirs = %v, want exactly %v", dirs, want)
	}
	for _, d := range dirs {
		if !want[d] {
			t.Errorf("allModuleDirs returned unexpected dir %q", d)
		}
	}
}

func TestAllModuleDirs_sorted(t *testing.T) {
	root := t.TempDir()
	writeGoMod(t, filepath.Join(root, "zzz"))
	writeGoMod(t, filepath.Join(root, "aaa"))
	writeGoMod(t, filepath.Join(root, "mmm"))
	dirs := allModuleDirs(root)
	if len(dirs) != 3 || dirs[0] != "aaa" || dirs[1] != "mmm" || dirs[2] != "zzz" {
		t.Errorf("allModuleDirs = %v, want sorted [aaa mmm zzz]", dirs)
	}
}

// fakeGvc writes an executable shell script standing in for govulncheck, printing out
// verbatim and exiting 0 — scanModule classifies purely from the printed text, matching
// how the real govulncheck also reports vulnerabilities via exit 0 (never used as the
// classification signal).
func fakeGvc(t *testing.T, out string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake govulncheck script is a POSIX shell script; vulncheck's own CI runs ubuntu-latest only")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "fake-govulncheck")
	script := "#!/bin/sh\ncat <<'EOF'\n" + out + "\nEOF\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("write fake govulncheck: %v", err)
	}
	return path
}

func TestScanModule_clean(t *testing.T) {
	gvc := fakeGvc(t, "No vulnerabilities found.\n")
	root := t.TempDir()
	writeGoMod(t, filepath.Join(root, "mod"))
	c := scanModule(root, gvc, "mod")
	if c.Outcome != gate.OK {
		t.Fatalf("Outcome = %v, want OK", c.Outcome)
	}
	if got := c.Field("status"); got != "CLEAN" {
		t.Errorf("status = %q, want CLEAN", got)
	}
}

func TestScanModule_vulnerable(t *testing.T) {
	out := strings.Join([]string{
		"Vulnerability #1: GO-2021-0113",
		"  x/text is affected by ...",
		"  Found in: golang.org/x/text@v0.3.5",
		"  Fixed in: golang.org/x/text@v0.3.8",
		"Vulnerability #2: GO-2022-9999",
		"  Found in: example.com/other@v1.0.0",
		"  Fixed in: example.com/other@v1.0.1",
	}, "\n")
	gvc := fakeGvc(t, out)
	root := t.TempDir()
	writeGoMod(t, filepath.Join(root, "mod"))
	c := scanModule(root, gvc, "mod")
	if c.Outcome != gate.Fail {
		t.Fatalf("Outcome = %v, want Fail", c.Outcome)
	}
	if got := c.Field("status"); got != "VULNERABLE" {
		t.Errorf("status = %q, want VULNERABLE", got)
	}
	if got := c.Field("count"); got != "2" {
		t.Errorf("count = %q, want 2 (two distinct Vulnerability #N: lines)", got)
	}
	var foundLine, fixedLine bool
	for _, f := range c.Fields {
		if f.Key == "line" && strings.Contains(f.State, "Found in:") {
			foundLine = true
		}
		if f.Key == "line" && strings.Contains(f.State, "Fixed in:") {
			fixedLine = true
		}
	}
	if !foundLine || !fixedLine {
		t.Errorf("expected both a Found-in and a Fixed-in line field; fields=%v", c.Fields)
	}
}

// TestScanModule_unscannedNeverReadsAsClean is the false-clean guard this gate exists for:
// a scan that produced neither "No vulnerabilities found" nor a Vulnerability line (a
// crashed/errored govulncheck) must be UNSCANNED, a FAIL outcome — never silently treated
// as clean just because it also didn't report a vulnerability.
func TestScanModule_unscannedNeverReadsAsClean(t *testing.T) {
	gvc := fakeGvc(t, "govulncheck: loading packages: package example.com/mod/... matched no packages\n")
	root := t.TempDir()
	writeGoMod(t, filepath.Join(root, "mod"))
	c := scanModule(root, gvc, "mod")
	if c.Outcome != gate.Fail {
		t.Fatalf("Outcome = %v, want Fail (unscanned is never a pass)", c.Outcome)
	}
	if got := c.Field("status"); got != "UNSCANNED" {
		t.Errorf("status = %q, want UNSCANNED", got)
	}
	if got := c.Field("detail"); got == "" {
		t.Error("detail is empty, want the first line of the failure output")
	}
}

func TestScanModule_emptyOutputIsUnscanned(t *testing.T) {
	gvc := fakeGvc(t, "")
	root := t.TempDir()
	writeGoMod(t, filepath.Join(root, "mod"))
	c := scanModule(root, gvc, "mod")
	if c.Outcome != gate.Fail {
		t.Errorf("Outcome = %v, want Fail — empty output is not evidence of cleanliness", c.Outcome)
	}
	if got := c.Field("status"); got != "UNSCANNED" {
		t.Errorf("status = %q, want UNSCANNED", got)
	}
}

func TestFindGovulncheck_envOverride(t *testing.T) {
	fake := fakeGvc(t, "irrelevant")
	t.Setenv("GOVULNCHECK", fake)
	if got := findGovulncheck(); got != fake {
		t.Errorf("findGovulncheck() = %q, want the GOVULNCHECK env override %q", got, fake)
	}
}

func TestScannerVersion_unknownOnMissingBinary(t *testing.T) {
	if got, want := scannerVersion(filepath.Join(t.TempDir(), "does-not-exist")), "unknown"; got != want {
		t.Errorf("scannerVersion(missing binary) = %q, want %q", got, want)
	}
}

func TestPrecondition(t *testing.T) {
	for _, tc := range []struct {
		name, goos, ambient, pinned, want string
	}{
		{"linux on the pin", "linux", "go1.27.3", "go1.27.3", ""},
		{"a Mac", "darwin", "go1.27.3", "go1.27.3", "must run on linux"},
		{"an older Go", "linux", "go1.27.0", "go1.27.3", "set GOTOOLCHAIN=go1.27.3"},
		{"a newer Go is not the pin either", "linux", "go1.27.4", "go1.27.3", "pinned toolchain go1.27.3"},
		{"no toolchain line", "linux", "go1.27.3", "", "no toolchain line"},
	} {
		got := precondition(tc.goos, tc.ambient, tc.pinned)
		if (tc.want == "") != (got == "") || !strings.Contains(got, tc.want) {
			t.Errorf("%s: precondition = %q, want it to contain %q", tc.name, got, tc.want)
		}
	}
}

// The pin is read from the real root go.mod, and every module that names a toolchain names
// the same one: a module left on an older line would be scanned against an older stdlib.
func TestToolchainLinesAgree(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	pin := pinnedToolchain(root)
	if pin == "" {
		t.Fatal("the root go.mod has no toolchain line")
	}
	for _, m := range allModuleDirs(root) {
		data, err := os.ReadFile(filepath.Join(root, m, "go.mod"))
		if err != nil {
			t.Fatal(err)
		}
		if mm := reToolchain.FindSubmatch(data); mm != nil && string(mm[1]) != pin {
			t.Errorf("%s/go.mod says toolchain %s, the root says %s", m, mm[1], pin)
		}
	}
}
