package gpumod

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// chdir changes to dir for the duration of the test, restoring the original working
// directory on cleanup — RepoRoot's walk-up fallback reads os.Getwd(), so exercising it
// means actually being in the target directory.
func chdir(t *testing.T, dir string) {
	t.Helper()
	orig, err := os.Getwd()
	if err != nil {
		t.Fatalf("os.Getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("os.Chdir(%q): %v", dir, err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(orig); err != nil {
			t.Fatalf("restoring cwd to %q: %v", orig, err)
		}
	})
}

// TestRepoRoot_gitFastPath runs RepoRoot from inside this real checkout (the tools module,
// a subdirectory of the repo) and confirms it resolves to the actual repo root — the `git
// rev-parse --show-toplevel` path, exercised for real rather than mocked.
func TestRepoRoot_gitFastPath(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("os.Getwd: %v", err)
	}
	// Confirm we're actually inside a git repo before trusting the result — otherwise this
	// test would trivially pass by hitting the walk-up path instead of the one it names.
	toplevel, gitErr := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if gitErr != nil {
		t.Skip("not running inside a git checkout")
	}
	wantRoot := strings.TrimSpace(string(toplevel))

	got, err := RepoRoot()
	if err != nil {
		t.Fatalf("RepoRoot() error = %v", err)
	}
	if got != wantRoot {
		t.Errorf("RepoRoot() = %q, want %q (git rev-parse --show-toplevel, from cwd %q)", got, wantRoot, wd)
	}
}

// TestRepoRoot_walkUpFallback exercises the walk-up loop directly: a temp directory tree,
// outside any real git repo, with a marker ".git" directory partway up. RepoRoot must climb
// from a nested child to find it — the fallback this package's doc comment promises for "a
// detached tree" (no working git, or none on PATH).
func TestRepoRoot_walkUpFallback(t *testing.T) {
	base := t.TempDir()
	// A bare ".git" directory (no HEAD/objects/refs) is not a repository git itself will
	// recognize — `git rev-parse --show-toplevel` run here fails, so this also exercises
	// the fallback even when git IS on PATH, not just the "git missing" case.
	if err := os.Mkdir(filepath.Join(base, ".git"), 0o755); err != nil {
		t.Fatalf("mkdir .git: %v", err)
	}
	nested := filepath.Join(base, "a", "b", "c")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatalf("mkdir nested: %v", err)
	}
	chdir(t, nested)

	got, err := RepoRoot()
	if err != nil {
		t.Fatalf("RepoRoot() error = %v", err)
	}
	// Resolve symlinks on both sides — on macOS, t.TempDir() lives under /var, which is
	// itself a symlink to /private/var, and RepoRoot returns whatever os.Getwd()'s walk
	// resolved to.
	wantResolved, err := filepath.EvalSymlinks(base)
	if err != nil {
		t.Fatalf("EvalSymlinks(%q): %v", base, err)
	}
	gotResolved, err := filepath.EvalSymlinks(got)
	if err != nil {
		t.Fatalf("EvalSymlinks(%q): %v", got, err)
	}
	if gotResolved != wantResolved {
		t.Errorf("RepoRoot() = %q (resolved %q), want %q (resolved %q)", got, gotResolved, base, wantResolved)
	}
}

// TestRepoRoot_notFound confirms the walk-up gives up (rather than looping forever or
// panicking) when there is no ".git" anywhere above cwd.
func TestRepoRoot_notFound(t *testing.T) {
	base := t.TempDir()
	nested := filepath.Join(base, "x", "y")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatalf("mkdir nested: %v", err)
	}
	chdir(t, nested)

	// t.TempDir() is rooted under the OS temp dir, which is not itself inside a git
	// checkout on any CI/dev box this runs on — if that ever stops holding, this test's
	// failure will say so directly rather than silently passing for the wrong reason.
	if _, err := RepoRoot(); err == nil {
		t.Error("RepoRoot() returned nil error in a tree with no .git anywhere above it")
	}
}

func TestModuleDirs(t *testing.T) {
	root := t.TempDir()
	mustGoMod := func(rel string) {
		t.Helper()
		dir := filepath.Join(root, rel)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("mkdir %q: %v", dir, err)
		}
		if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module x\n"), 0o644); err != nil {
			t.Fatalf("write go.mod in %q: %v", dir, err)
		}
	}
	mustGoMod("gpu")
	mustGoMod("gpu/anncuda")
	mustGoMod("gpu/annmetal")
	// A file that happens to be named go.mod but isn't a directory boundary marker anyone
	// else cares about, and a decoy elsewhere in gpu/ with no go.mod, should not confuse
	// the walk.
	if err := os.MkdirAll(filepath.Join(root, "gpu", "not_a_module"), 0o755); err != nil {
		t.Fatalf("mkdir decoy: %v", err)
	}
	// Outside gpu/ entirely — must not be picked up.
	mustGoMod("chunk/treesitter")

	dirs, err := ModuleDirs(root)
	if err != nil {
		t.Fatalf("ModuleDirs error = %v", err)
	}
	want := []string{"gpu", "gpu/anncuda", "gpu/annmetal"}
	if len(dirs) != len(want) {
		t.Fatalf("ModuleDirs = %v, want %v", dirs, want)
	}
	for i, w := range want {
		if dirs[i] != w {
			t.Errorf("dirs[%d] = %q, want %q (got full: %v)", i, dirs[i], w, dirs)
		}
	}
}

func TestModuleDirs_noGpuDir(t *testing.T) {
	root := t.TempDir() // no gpu/ subdirectory at all
	if _, err := ModuleDirs(root); err == nil {
		t.Error("ModuleDirs on a root with no gpu/ directory returned nil error")
	}
}

// TestProvenance_realRepo runs Provenance for real against this checkout — the fields it
// reports (commit, host, OS, arch) are exactly what a VERDICT line names, so this is the
// integration test matching how every gate actually uses it, not a mock of git/uname.
func TestProvenance_realRepo(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root, err := RepoRoot()
	if err != nil {
		t.Skipf("cannot resolve repo root: %v", err)
	}
	p := Provenance(root)
	if p.Commit == "" || p.Commit == "?" {
		t.Errorf("Commit = %q, want a real short hash (are we in a git repo with at least one commit?)", p.Commit)
	}
	if len(p.Commit) > 1 && !isHexish(p.Commit) {
		t.Errorf("Commit = %q, does not look like a hex short-hash", p.Commit)
	}
	if p.Date == "" {
		t.Error("Date is empty")
	}
	if !strings.HasSuffix(p.Date, "Z") {
		t.Errorf("Date = %q, want a UTC (Z-suffixed) timestamp", p.Date)
	}
	if p.Host == "" {
		t.Error("Host is empty — uname -n produced nothing usable")
	}
	if p.OSName == "" {
		t.Error("OSName is empty — uname -s produced nothing usable")
	}
	if p.Arch == "" {
		t.Error("Arch is empty — uname -m produced nothing usable")
	}
	// Dirty is either "" or exactly " +dirty" — never anything else.
	if p.Dirty != "" && p.Dirty != " +dirty" {
		t.Errorf("Dirty = %q, want \"\" or \" +dirty\"", p.Dirty)
	}
}

func isHexish(s string) bool {
	for _, r := range s {
		if !strings.ContainsRune("0123456789abcdefABCDEF", r) {
			return false
		}
	}
	return true
}

// TestExec_pinsGOWORKAndCGO is the regression test for the property tools/preflight,
// tools/gpupins, and tools/releasegate all now depend on having centralized here: every
// subprocess Exec runs gets GOWORK=off and CGO_ENABLED=0, regardless of the ambient shell
// environment — verified by asking `go env` itself, in the child process, rather than
// inspecting cmd.Env before the fact.
func TestExec_pinsGOWORKAndCGO(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not on PATH")
	}
	root, err := RepoRoot()
	if err != nil {
		t.Skipf("cannot resolve repo root: %v", err)
	}
	out, rc := Exec(root, "tools", nil, "go", "env", "GOWORK")
	if rc != 0 {
		t.Fatalf("go env GOWORK: rc=%d out=%q", rc, out)
	}
	if got := strings.TrimSpace(out); got != "off" {
		t.Errorf("go env GOWORK = %q, want %q", got, "off")
	}

	out, rc = Exec(root, "tools", nil, "go", "env", "CGO_ENABLED")
	if rc != 0 {
		t.Fatalf("go env CGO_ENABLED: rc=%d out=%q", rc, out)
	}
	if got := strings.TrimSpace(out); got != "0" {
		t.Errorf("go env CGO_ENABLED = %q, want %q", got, "0")
	}
}

// TestExec_extraEnvOverrides confirms a caller's extraEnv can override Exec's own defaults
// (os/exec keeps the LAST occurrence of a duplicate key) — the seam every caller that needs
// something other than CGO_ENABLED=0 relies on.
func TestExec_extraEnvOverrides(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not on PATH")
	}
	root, err := RepoRoot()
	if err != nil {
		t.Skipf("cannot resolve repo root: %v", err)
	}
	out, rc := Exec(root, "tools", []string{"CGO_ENABLED=1"}, "go", "env", "CGO_ENABLED")
	if rc != 0 {
		t.Fatalf("go env CGO_ENABLED: rc=%d out=%q", rc, out)
	}
	if got := strings.TrimSpace(out); got != "1" {
		t.Errorf("go env CGO_ENABLED with extraEnv override = %q, want %q", got, "1")
	}
}

// TestExec_relDirJoinsRoot confirms Exec actually runs in root/relDir, not just root — the
// distinction every caller passing a non-empty relDir depends on.
func TestExec_relDirJoinsRoot(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not on PATH")
	}
	root, err := RepoRoot()
	if err != nil {
		t.Skipf("cannot resolve repo root: %v", err)
	}
	// `go list .` in tools/gate should report the gate package specifically — proof the
	// command actually ran there, not at root. relDir is relative to the REPO root (root
	// here), so this is "tools/gate", not "gate" — tools is its own module one level down.
	out, rc := Exec(root, "tools/gate", nil, "go", "list", ".")
	if rc != 0 {
		t.Fatalf("go list . in tools/gate: rc=%d out=%q", rc, out)
	}
	if got := strings.TrimSpace(out); got != "github.com/townsendmerino/aikit/tools/gate" {
		t.Errorf("go list . in relDir=gate = %q, want the gate package path", got)
	}
}

func TestExec_exitCodePropagates(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not on PATH")
	}
	root, err := RepoRoot()
	if err != nil {
		t.Skipf("cannot resolve repo root: %v", err)
	}
	// An unrecognized go subcommand reliably exits non-zero.
	_, rc := Exec(root, "tools", nil, "go", "not-a-real-subcommand-xyz")
	if rc == 0 {
		t.Error("Exec of an invalid go subcommand returned rc=0")
	}
}

func TestExec_missingBinaryReturnsNegativeOne(t *testing.T) {
	root := t.TempDir()
	_, rc := Exec(root, "", nil, "definitely-not-a-real-binary-xyz")
	if rc != -1 {
		t.Errorf("Exec of a nonexistent binary: rc=%d, want -1", rc)
	}
}

func TestExitCode(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want int
	}{
		{"nil error", nil, 0},
		{"non-exec error", errors.New("boom"), -1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := exitCode(c.err); got != c.want {
				t.Errorf("exitCode(%v) = %d, want %d", c.err, got, c.want)
			}
		})
	}

	// A real *exec.ExitError, for the branch the table above can't construct directly.
	cmd := exec.Command("go", "not-a-real-subcommand-xyz")
	runErr := cmd.Run()
	if runErr == nil {
		t.Skip("expected the bogus go subcommand to fail")
	}
	ee, ok := runErr.(*exec.ExitError)
	if !ok {
		t.Skipf("error was not *exec.ExitError: %T", runErr)
	}
	if got := exitCode(runErr); got != ee.ExitCode() {
		t.Errorf("exitCode(ExitError) = %d, want %d", got, ee.ExitCode())
	}
	if got := exitCode(runErr); got == 0 {
		t.Errorf("exitCode(ExitError from a failing command) = 0, want nonzero")
	}
}

// TestGoSep_separatesStdoutAndStderr reproduces the exact scenario GoSep's own doc comment
// says it exists for: `go list ./...` in a module with no .go files puts its
// "matched no packages" note on STDERR and leaves stdout empty. CombinedOutput would fold
// the note into stdout and make an empty package list look non-empty — this is the
// regression test for GoSep existing at all.
func TestGoSep_separatesStdoutAndStderr(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not on PATH")
	}
	empty := t.TempDir()
	if err := os.WriteFile(filepath.Join(empty, "go.mod"), []byte("module scratch\n\ngo 1.21\n"), 0o644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
	// rc is not asserted: modern `go list ./...` on a package-less module exits 0 with an
	// informational stderr note, not an error — GoSep's own doc comment only promises the
	// stdout/stderr split, which is what this test actually checks.
	stdout, stderr, _ := GoSep(empty, "", nil, "list", "./...")
	if strings.TrimSpace(stdout) != "" {
		t.Errorf("stdout = %q, want empty (no packages)", stdout)
	}
	if !strings.Contains(stderr, "no Go files") && !strings.Contains(stderr, "matched no packages") && strings.TrimSpace(stderr) == "" {
		t.Errorf("stderr = %q, want a note about missing packages", stderr)
	}
}

// RepoRoot under a pre-push hook in a linked worktree: git exports GIT_DIR (the worktree's
// own git directory) to the hook, and `git rev-parse --show-toplevel` with GIT_DIR set and no
// GIT_WORK_TREE answers the cwd, which for `go run -C tools ./preflight` is tools/.
func TestRepoRoot_linkedWorktreeUnderHookEnv(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("no git on PATH")
	}
	base := t.TempDir()
	main, wt := filepath.Join(base, "main"), filepath.Join(base, "wt")
	git := func(dir string, args ...string) string {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		// This test may itself run under a hook: the fixture's git must not inherit its GIT_DIR.
		cmd.Env = append(withoutGitLocation(os.Environ()), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	if err := os.MkdirAll(main, 0o755); err != nil {
		t.Fatal(err)
	}
	git(main, "init", "-q")
	git(main, "commit", "-q", "--allow-empty", "-m", "root")
	git(main, "worktree", "add", "-q", wt)
	if err := os.MkdirAll(filepath.Join(wt, "tools"), 0o755); err != nil {
		t.Fatal(err)
	}
	gitDir := git(wt, "rev-parse", "--absolute-git-dir")
	wantRoot, err := filepath.EvalSymlinks(wt)
	if err != nil {
		t.Fatal(err)
	}

	chdir(t, filepath.Join(wt, "tools"))
	t.Setenv("GIT_DIR", gitDir)
	got, err := RepoRoot()
	if err != nil {
		t.Fatal(err)
	}
	if got, _ = filepath.EvalSymlinks(got); got != wantRoot {
		t.Fatalf("RepoRoot() with the hook's GIT_DIR = %s, want the worktree root %s", got, wantRoot)
	}
}

// The walk-up fallback stops at a linked worktree's .git, which is a file, not a directory.
func TestRepoRoot_walkUpStopsAtGitFile(t *testing.T) {
	base := t.TempDir()
	if err := os.WriteFile(filepath.Join(base, ".git"), []byte("gitdir: /nonexistent\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	nested := filepath.Join(base, "tools")
	if err := os.Mkdir(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	chdir(t, nested)
	got, err := RepoRoot()
	if err != nil {
		t.Fatalf("RepoRoot() error = %v", err)
	}
	want, _ := filepath.EvalSymlinks(base)
	if got, _ = filepath.EvalSymlinks(got); got != want {
		t.Fatalf("RepoRoot() = %s, want %s", got, want)
	}
}
