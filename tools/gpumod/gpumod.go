// Package gpumod holds the helpers the two gpu pre-tag gates (gpugate, gpudevice) share:
// enumerating the nine gpu modules, reading the commit provenance a VERDICT line names, and
// running a `go` subprocess under the environment those gates pin. It is HELPERS, not a
// runner — the runner is tools/gate; this package decides nothing.
package gpumod

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

func nowUTC() string { return time.Now().UTC().Format("2006-01-02T15:04:05Z") }

// GolangciLint is the PINNED linter, shared by every gate that lints. `go run pkg@ver`
// compiles golangci-lint with the caller's Go toolchain, so its bundled staticcheck is
// identical on every box and in CI — pinning the version alone does not pin that. CI's
// core and gpu jobs, and the preflight mirror, all resolve to this one string.
const GolangciLint = "github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.14.0"

// RepoRoot finds the checkout root regardless of the caller's cwd (the tools run as
// `go run -C tools ./gpugate`, whose cwd is tools/). git is the authority; a walk up to a
// .git directory is the fallback for a detached tree.
func RepoRoot() (string, error) {
	if out, err := exec.Command("git", "rev-parse", "--show-toplevel").Output(); err == nil {
		if p := strings.TrimSpace(string(out)); p != "" {
			return p, nil
		}
	}
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if fi, err := os.Stat(filepath.Join(dir, ".git")); err == nil && fi.IsDir() {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", os.ErrNotExist
		}
		dir = parent
	}
}

// ModuleDirs returns the nine gpu module directories relative to root ("gpu",
// "gpu/anncuda", …), sorted — the same enumeration the shells did with
// `find gpu -name go.mod -exec dirname`. A break in one module is invisible from the
// others, and none of them is in the root module's build, so every gate walks all nine.
func ModuleDirs(root string) ([]string, error) {
	var dirs []string
	gpuRoot := filepath.Join(root, "gpu")
	err := filepath.WalkDir(gpuRoot, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && d.Name() == "go.mod" {
			rel, rerr := filepath.Rel(root, filepath.Dir(p))
			if rerr != nil {
				return rerr
			}
			dirs = append(dirs, rel)
		}
		return nil
	})
	sort.Strings(dirs)
	return dirs, err
}

// Prov is the commit provenance a VERDICT line names: a gate's verdict describes a specific
// commit, so a dirty tree is flagged rather than quietly attributed to that commit.
type Prov struct {
	Commit string // short hash, or "?"
	Dirty  string // " +dirty" or ""
	Date   string // UTC, RFC3339-ish
	Host   string // uname -n
	OSName string // uname -s: Darwin / Linux
	Arch   string // uname -m: arm64 / x86_64
}

// Provenance reads the commit/dirty/host/OS/arch/date the shells put in their VERDICT lines.
func Provenance(root string) Prov {
	p := Prov{Commit: "?"}
	if out, err := gitIn(root, "rev-parse", "--short", "HEAD"); err == nil {
		p.Commit = strings.TrimSpace(out)
	}
	if out, err := gitIn(root, "status", "--porcelain"); err == nil && strings.TrimSpace(out) != "" {
		p.Dirty = " +dirty"
	}
	p.Date = nowUTC()
	p.Host = uname("-n")
	p.OSName = uname("-s")
	p.Arch = uname("-m")
	return p
}

func gitIn(dir string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.Output()
	return string(out), err
}

func uname(flag string) string {
	out, err := exec.Command("uname", flag).Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

// Exec runs `name <args>` in root/relDir under the environment the gpu gates require:
// GOWORK=off (a workspace overrides module resolution with local directories and would make
// a release gate measure the developer's overrides rather than what will be published —
// 1809a2c) and CGO_ENABLED=0 (the cgo-free invariant every gate enforces), plus any extra
// assignments the caller passes EXPLICITLY per command — nothing gate-relevant is left to
// an inherited shell export. Returns combined output and the process exit code (-1 if it
// could not start).
func Exec(root, relDir string, extraEnv []string, name string, args ...string) (string, int) {
	cmd := exec.Command(name, args...)
	cmd.Dir = filepath.Join(root, relDir)
	cmd.Env = append(os.Environ(), "GOWORK=off", "CGO_ENABLED=0")
	cmd.Env = append(cmd.Env, extraEnv...)
	out, err := cmd.CombinedOutput()
	return string(out), exitCode(err)
}

// Go is Exec for a `go` subcommand.
func Go(root, relDir string, extraEnv []string, args ...string) (string, int) {
	return Exec(root, relDir, extraEnv, "go", args...)
}

// GoSep runs a `go` subcommand and returns stdout and stderr SEPARATELY, plus the exit code.
// It exists for `go list ./...`, whose "no buildable packages here" case is stdout-empty
// with the note "matched no packages" on STDERR — so n/a detection must read stdout alone,
// exactly as the shells did with `go list ./... 2>/dev/null`. CombinedOutput would fold the
// note into stdout and make an empty package list look non-empty.
func GoSep(root, relDir string, extraEnv []string, args ...string) (stdout, stderr string, rc int) {
	cmd := exec.Command("go", args...)
	cmd.Dir = filepath.Join(root, relDir)
	cmd.Env = append(os.Environ(), "GOWORK=off", "CGO_ENABLED=0")
	cmd.Env = append(cmd.Env, extraEnv...)
	var so, se bytes.Buffer
	cmd.Stdout = &so
	cmd.Stderr = &se
	rc = exitCode(cmd.Run()) // MUST run before reading the buffers — return args evaluate left to right
	return so.String(), se.String(), rc
}

func exitCode(err error) int {
	if err == nil {
		return 0
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return ee.ExitCode()
	}
	return -1
}
