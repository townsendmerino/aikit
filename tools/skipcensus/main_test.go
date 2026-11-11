package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

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

// TestRun_repoRootFailure confirms skipcensus fails loudly (exit 2), rather than running
// `go test` against the wrong tree or panicking, when it cannot resolve a repo root at all.
func TestRun_repoRootFailure(t *testing.T) {
	dir := t.TempDir() // no .git anywhere above it
	chdir(t, dir)
	if got := run(nil); got != 2 {
		t.Errorf("run(nil) = %d, want 2 (cannot locate repo root)", got)
	}
}

// TestRun_endToEnd exercises the real wrapper: default args ("./..." when none given), the
// summary line skipcensus itself appends after skips.Run returns, and that a clean run's
// exit code is 0 — against a real, tiny, throwaway Go module (skips.Run itself shells to a
// real `go test -json`, so this is the only way to test skipcensus's OWN plumbing around
// it: repo-root resolution, arg defaulting, and the summary/exit-code glue).
func TestRun_endToEnd(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not on PATH")
	}
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatalf("mkdir .git: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module scratch\n\ngo 1.21\n"), 0o644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
	testSrc := `package scratch

import "testing"

func TestPasses(t *testing.T) {}

func TestSkips(t *testing.T) { t.Skip("no fixture present") }
`
	if err := os.WriteFile(filepath.Join(root, "scratch_test.go"), []byte(testSrc), 0o644); err != nil {
		t.Fatalf("write scratch_test.go: %v", err)
	}
	chdir(t, root)

	var buf bytes.Buffer
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stdout = w
	done := make(chan struct{})
	go func() {
		buf.ReadFrom(r) //nolint:errcheck // a pipe read in a test: a short read shows up as a failed assertion below
		close(done)
	}()

	code := run(nil) // nil args must default to ./...

	w.Close()
	os.Stdout = orig
	<-done

	if code != 0 {
		t.Errorf("run(nil) = %d, want 0 (one passing test, one skip — nothing failed)", code)
	}
	out := buf.String()
	if !strings.Contains(out, "skip census") {
		t.Errorf("output missing the census summary line; got:\n%s", out)
	}
	if !strings.Contains(out, "1 passed") {
		t.Errorf("output missing the expected pass count (1 passed); got:\n%s", out)
	}
	if !strings.Contains(out, "1 skipped") {
		t.Errorf("output missing the expected skip count (1 skipped); got:\n%s", out)
	}
}
