// Command vulncheck is the vulnerability STATEMENT for every module in this repo — the
// release-notes deliverable RELEASING.md step 4 pastes into the notes, not hygiene. It
// replaces scripts/vulncheck.sh, expressed as a gate.Check matrix over tools/gate.
//
// WHY A DELIVERABLE. aikit's pitch is a single static binary scp'd somewhere and run
// offline; whoever deploys it cannot patch it in place. "What is in it, and is any of it
// known-vulnerable" is part of that claim, the same way cgo-free and dependency-light are.
// govulncheck earns a gate because it is REACHABILITY-FILTERED: it reports a vulnerability
// only when a symbol the build actually reaches is affected, so "no vulnerabilities found"
// is a statement about this binary, not about the dependency list.
//
// CONTRACTS CARRIED FROM THE SHELL:
//   - EVERY module is enumerated from the tree (now 14, including tools/ itself), never a
//     list — a scanner that silently covers half a repo is worse than none.
//   - UNSCANNABLE IS A FAILURE, distinct from clean. "Could not load packages" must never
//     read as "no vulnerabilities". Both a reachable vulnerability and an unscannable module
//     make the STATEMENT INCOMPLETE.
//   - The four *metal modules are scanned under GOOS=darwin (their sources are entirely
//     behind //go:build darwin and would otherwise contribute zero packages and a green
//     tick on Linux).
//   - govulncheck itself missing is INCONCLUSIVE — could not judge — never a pass.
//
// It inherits the ambient toolchain environment exactly as the shell did — no GOWORK
// override — so a scan on the Linux/CI target (no go.work) is byte-identical to
// scripts/vulncheck.sh; only the *metal GOOS is forced.
//
// Usage:  go run -C tools ./vulncheck
// Env:    GOVULNCHECK   path to the binary (default: govulncheck on PATH, else
//
//	$(go env GOPATH)/bin/govulncheck)
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"

	"github.com/townsendmerino/aikit/tools/canary"
	"github.com/townsendmerino/aikit/tools/gate"
	"github.com/townsendmerino/aikit/tools/gpumod"
)

// reVuln matches govulncheck's per-vulnerability header, e.g. "Vulnerability #1: GO-2024-…".
var reVuln = regexp.MustCompile(`Vulnerability #[0-9]+:`)

func main() { os.Exit(run()) }

// reportCells prints one line per module — plus the reachable-advisory detail
// lines under a VULNERABLE one — and returns the tally the STATEMENT is built
// from. Split out of run() because it was the only nesting in an otherwise flat
// precondition-then-scan-then-report flow: range → switch → case → range → if,
// five deep, against run()'s own two.
func reportCells(cells []gate.Cell) (clean, vuln, unscanned int) {
	for _, c := range cells {
		switch c.Field("status") {
		case "CLEAN":
			clean++
			fmt.Printf("  %-30s CLEAN\n", c.Name)
		case "VULNERABLE":
			vuln++
			fmt.Printf("  %-30s VULNERABLE (%s reachable)\n", c.Name, c.Field("count"))
			for _, f := range c.Fields {
				if f.Key == "line" {
					fmt.Printf("      %s\n", f.State)
				}
			}
		default: // UNSCANNED
			unscanned++
			fmt.Printf("  %-30s UNSCANNED — %s\n", c.Name, c.Field("detail"))
		}
	}
	return clean, vuln, unscanned
}

// precondition says why a scan here could not produce the STATEMENT, or "" if it can.
//
// Two things made a scan on a development Mac INCOMPLETE without saying why (found by the
// v1.76.1 dry run, 2026-11-10), and both are refused up front now rather than discovered in
// the report:
//
//   - NOT LINUX. The *cuda modules have packages only on linux, and unlike the *metal ones
//     (scanned under GOOS=darwin from anywhere) their sources need the host to be linux, so
//     elsewhere they come back UNSCANNED.
//   - NOT THE PINNED TOOLCHAIN. govulncheck judges the standard library of the Go that runs
//     it. Only modules with a `toolchain` line switch to the pinned one by themselves; the
//     rest are scanned with whatever is installed, and an older one re-reports advisories the
//     pin already fixed (Go 1.27.0 reported nine against a tree pinned to 1.27.3).
func precondition(goos, ambient, pinned string) string {
	switch {
	case goos != "linux":
		return "vulncheck must run on linux, and this is " + goos + ": the *cuda modules have no packages here and would be UNSCANNED"
	case pinned == "":
		return "the root go.mod has no toolchain line to scan against"
	case ambient != pinned:
		return "vulncheck must run on the pinned toolchain " + pinned + ", and `go` here is " + ambient +
			": set GOTOOLCHAIN=" + pinned + " (modules without their own toolchain line are scanned with the Go that is installed)"
	}
	return ""
}

var reToolchain = regexp.MustCompile(`(?m)^toolchain\s+(go\S+)`)

// pinnedToolchain is the root go.mod's toolchain line, e.g. "go1.27.3", or "".
func pinnedToolchain(root string) string {
	data, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		return ""
	}
	if m := reToolchain.FindSubmatch(data); m != nil {
		return string(m[1])
	}
	return ""
}

// ambientGoVersion is the Go a module WITHOUT a toolchain line gets: `go env GOVERSION`
// asked from outside any module, so no go.mod switches it.
func ambientGoVersion() string {
	cmd := exec.Command("go", "env", "GOVERSION")
	cmd.Dir = os.TempDir()
	cmd.Env = append(os.Environ(), "GOWORK=off")
	out, err := cmd.Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

func run() int {
	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Println("STATEMENT: INCONCLUSIVE — cannot locate repo root: " + err.Error())
		return 2
	}
	gvc := findGovulncheck()
	if gvc == "" {
		fmt.Println("STATEMENT: INCONCLUSIVE — govulncheck not found (go install golang.org/x/vuln/cmd/govulncheck@latest); nothing scanned")
		return 2
	}
	if why := precondition(runtime.GOOS, ambientGoVersion(), pinnedToolchain(root)); why != "" {
		fmt.Println("STATEMENT: INCONCLUSIVE — " + why + "; nothing scanned")
		return 2
	}
	p := gpumod.Provenance(root)
	prov := p.Commit + p.Dirty
	fmt.Printf("aikit vulnerability scan — %s — %s\n", prov, p.Date)
	fmt.Printf("scanner: %s\n\n", scannerVersion(gvc))

	// CANARY: prove govulncheck actually scans and reports, with the SAME binary the shipped
	// scan uses, before any "no reachable vulnerabilities" is trusted. The vulnfixture module
	// reaches a symbol with a known advisory; if it is not reported, govulncheck examined
	// nothing (missing binary, no vuln DB, empty graph) and the STATEMENT means nothing. This
	// catches the case the per-module UNSCANNED=FAIL cannot: a scan that exits 0 with "No
	// vulnerabilities found" over a graph it never really analysed.
	cout := scanRaw(root, gvc, canary.VulnFixtureDir)
	if res := canary.CheckGovulncheck(cout); !res.Fired {
		fmt.Println("STATEMENT: INCONCLUSIVE — CANNOT-EVALUATE: " + res.Reason + "; nothing trusted")
		return 2
	}
	fmt.Printf("canary: govulncheck reports %s in %s ✓\n\n", canary.CanaryAdvisory, canary.VulnFixtureDir)

	mods := allModuleDirs(root)
	checks := make([]gate.Check, 0, len(mods))
	for _, m := range mods {
		checks = append(checks, gate.Check{Name: m, Run: func() gate.Cell { return scanModule(root, gvc, m) }})
	}
	cells := gate.RunAll(checks)

	clean, vuln, unscanned := reportCells(cells)
	total := clean + vuln + unscanned

	fmt.Println()
	if vuln == 0 && unscanned == 0 {
		fmt.Printf("STATEMENT: no reachable vulnerabilities in %d/%d modules at %s (%s)\n", clean, total, prov, p.Date)
		return 0
	}
	if unscanned > 0 {
		fmt.Printf("  %d module(s) could not be scanned — that is not a clean result.\n", unscanned)
	}
	if vuln > 0 {
		fmt.Printf("  %d module(s) have reachable vulnerabilities.\n", vuln)
	}
	fmt.Printf("STATEMENT: INCOMPLETE — %d clean, %d vulnerable, %d unscanned, of %d at %s\n", clean, vuln, unscanned, total, prov)
	return 1
}

// scanRaw runs govulncheck ./... in one module dir and returns its combined output — the raw
// form the canary inspects, without the CLEAN/VULNERABLE/UNSCANNED classification.
func scanRaw(root, gvc, m string) string {
	cmd := exec.Command(gvc, "./...")
	cmd.Dir = filepath.Join(root, m)
	cmd.Env = os.Environ()
	out, _ := cmd.CombinedOutput() //nolint:errcheck // the canary judges the output text; a failed run has none and does not fire
	return string(out)
}

// scanModule runs govulncheck in one module and classifies the result the way the shell
// did: a "No vulnerabilities found" with exit 0 is CLEAN; any "Vulnerability #N:" is
// VULNERABLE; anything else is UNSCANNED (the scan itself failed) — never mistaken for a pass.
func scanModule(root, gvc, m string) gate.Cell {
	cmd := exec.Command(gvc, "./...")
	cmd.Dir = filepath.Join(root, m)
	cmd.Env = os.Environ()
	if strings.HasSuffix(m, "metal") {
		// darwin-only sources contribute zero packages under the host GOOS on Linux;
		// scanning them under GOOS=darwin is what the shell did to avoid a vacuous pass.
		cmd.Env = append(cmd.Env, "GOOS=darwin", "GOARCH=arm64")
	}
	outB, _ := cmd.CombinedOutput() //nolint:errcheck // judged by the output below: anything that is not CLEAN or VULNERABLE is UNSCANNED
	out := string(outB)

	switch {
	case strings.Contains(out, "No vulnerabilities found"):
		return gate.Cell{Name: m, Outcome: gate.OK, Fields: []gate.Field{{Key: "status", State: "CLEAN"}}}
	case reVuln.MatchString(out):
		n := len(reVuln.FindAllString(out, -1))
		c := gate.Cell{Name: m, Outcome: gate.Fail, Fields: []gate.Field{
			{Key: "status", State: "VULNERABLE"},
			{Key: "count", State: fmt.Sprintf("%d", n)},
		}}
		for ln := range strings.SplitSeq(out, "\n") {
			t := strings.TrimSpace(ln)
			if reVuln.MatchString(ln) || strings.HasPrefix(t, "Found in:") || strings.HasPrefix(t, "Fixed in:") {
				c.Fields = append(c.Fields, gate.Field{Key: "line", State: t})
			}
		}
		return c
	default:
		return gate.Cell{Name: m, Outcome: gate.Fail, Fields: []gate.Field{
			{Key: "status", State: "UNSCANNED"},
			{Key: "detail", State: firstNonEmpty(out)},
		}}
	}
}

// allModuleDirs enumerates every SHIPPED module in the tree (root as ".", others relative),
// sorted — the same set `find . -name go.mod` produces, minus .git and .venv, and minus the
// canary fixture module (tools/canary/vulnfixture pins a deliberately-vulnerable dependency as
// a govulncheck positive control; it is required by nothing and must never enter this scan).
func allModuleDirs(root string) []string {
	var dirs []string
	//nolint:errcheck // the callback never returns an error other than SkipDir, so WalkDir cannot
	filepath.WalkDir(root, func(pth string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() && (d.Name() == ".git" || d.Name() == ".venv") {
			return filepath.SkipDir
		}
		if d.IsDir() {
			if rel, _ := filepath.Rel(root, pth); rel == canary.VulnFixtureDir { //nolint:errcheck // pth is under root
				return filepath.SkipDir // the intentionally-vulnerable canary module
			}
		}
		if !d.IsDir() && d.Name() == "go.mod" {
			rel, _ := filepath.Rel(root, filepath.Dir(pth)) //nolint:errcheck // pth is under root
			dirs = append(dirs, rel)
		}
		return nil
	})
	sort.Strings(dirs)
	return dirs
}

func findGovulncheck() string {
	if v := os.Getenv("GOVULNCHECK"); v != "" {
		return v
	}
	if p, err := exec.LookPath("govulncheck"); err == nil {
		return p
	}
	if out, err := exec.Command("go", "env", "GOPATH").Output(); err == nil {
		cand := filepath.Join(strings.TrimSpace(string(out)), "bin", "govulncheck")
		if fi, err := os.Stat(cand); err == nil && !fi.IsDir() {
			return cand
		}
	}
	return ""
}

func scannerVersion(gvc string) string {
	out, err := exec.Command(gvc, "-version").Output()
	if err != nil {
		return "unknown"
	}
	return strings.Join(strings.Fields(string(out)), " ")
}

func firstNonEmpty(s string) string {
	for ln := range strings.SplitSeq(s, "\n") {
		if strings.TrimSpace(ln) != "" {
			t := strings.TrimSpace(ln)
			if len(t) > 70 {
				t = t[:70]
			}
			return t
		}
	}
	return ""
}
