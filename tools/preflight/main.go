// Command preflight runs what CI's core (root-module) job runs, locally, before pushing —
// the pre-push mirror of CI, ported from scripts/preflight.sh onto tools/gate.
//
// WHY IT EXISTS. Using CI as a first linter rather than a last check costs a full round trip
// per mistake (a revert once left an asm kernel with no caller; golangci-lint reports that
// in three seconds, but it was found from CI, after pushing, twice — 2026-08-12). So this runs
// every one of ci.yml's core-job steps, in the same order, with exactly one deliberate
// weakening: `go test` runs UNRACED (CI's own `-race` pass costs a real multiple of both time
// and memory — too slow for a pre-push gate). Every other core step is mirrored exactly, including
// "no cgo deps in core graph" and "test (aikit_checks)" — both fast, neither needs network or a
// GPU, and both used to be silent gaps here with no actual justification.
//
// THE EXCLUSIONS ARE ENFORCED, NOT JUST STATED. buildChecks's names and main_test.go's
// notImplemented map are checked against every ci.yml core-job step name
// (TestCoreStepsAccountedFor): a step landing in neither fails that test, so an exclusion has
// to be a reviewed, named entry — not a stale prose claim nobody's checking against reality
// (which is exactly how "no cgo deps in core graph" and "test (aikit_checks)" went unmirrored
// for a while despite fitting none of the reasons this comment used to give).
//
// preflight covers the core job, PLUS one thing the core job cannot cover: a cross-GOOS/
// GOARCH vet. arm64, treesitter, vulncheck, perf-smoke, scripts-boundary, gpu-pins, and
// gpu-kernels are separate ci.yml jobs that genuinely need network, a GPU, another OS, or
// another module, and aren't attempted here; nor is fuzz-smoke (10 targets x 15s x up to 2
// retries — too slow for a pre-push gate, and its own job so it stays off CI's critical path); gpugate/gpudevice/scriptsguard/gpupins are the
// local tools for the pieces that have one.
//
// THE CROSS-VET IS THE ONE DELIBERATE ADDITION, and it is not a core-job mirror — it is the
// only check here that front-runs a DIFFERENT ci.yml job. The reason "another OS" excuses
// the windows job's `go build` and `go test` but not its `go vet`: vet cross-compiles, so
// the check needs no runner, no network, and about three seconds. Meanwhile every local
// build on this repo's development box (darwin/arm64) type-checks only the arm64 file set,
// which makes "compiles for me, undefined symbol everywhere else" a class of mistake that
// CANNOT be found locally without this — and it is not hypothetical: 9724289 landed a test
// file referencing arm64-only symbols with no build tag, went green locally, and reddened
// four non-arm64 jobs at once (windows, perf, both GOEXPERIMENT=simd legs) for one missing
// `//go:build arm64` line, costing a full round trip plus e4613bb to fix.
//
// Two targets, not one: windows/amd64 covers the windows job's own vet step, and linux/amd64
// covers the amd64-only breakage the windows leg can miss (a linux-tagged file, a cgo-free
// path that differs). Together they cover every non-arm64 job 9724289 broke. Note these vet
// the arm64 host's NON-arm64 file sets — the arm64 files themselves are already covered by
// the plain "vet" step above, so the pair genuinely widens coverage rather than repeating it.
//
// ENVIRONMENT PARITY IS PART OF EACH CHECK. Reproducing a command without its environment is
// not reproducing the check. Every go step runs through gpumod.Exec, which pins GOWORK=off
// and CGO_ENABLED=0 explicitly on its own exec.Cmd for every check, not just the one step
// literally named for it — a developer's go.work must not decide what this reports, and the
// dedicated "build (cgo-free)" step is now a label matching ci.yml's own step name, not the
// only place the pin applies. The env each check ran under is printed next to it. This
// CORRECTS the shell, which set GOWORK=off once as a global export (correct in effect, but
// invisible and not per-command) and used whatever golangci-lint was on PATH — a build that
// drifts by the Go it was compiled with (see gpumod.GolangciLint). This runs the linter
// build-pinned via `go run @v2.13.0`, the same one CI's goinstall v2.13.0 and the gpu gate
// resolve to.
//
// THE RELATIONSHIP TO ci.yml IS GUARDED. This check list mirrors the core job; a silent
// second copy drifts. tools/ is stdlib-only (no YAML parser), so a full derivation is out of
// reach — but main_test.go's tripwire reads ci.yml as text and fails if the core job's step
// names change, so the next check CI gains cannot silently go un-mirrored (and
// TestCoreStepsAccountedFor, above, catches it going un-mirrored on the implementation side).
//
// Usage:
//
//	go run -C tools ./preflight            # everything
//	go run -C tools ./preflight --fast     # skip go test / aikit_checks (fmt/vet/lint/build only)
package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/townsendmerino/aikit/tools/canary"
	"github.com/townsendmerino/aikit/tools/gate"
	"github.com/townsendmerino/aikit/tools/gpumod"
	"github.com/townsendmerino/aikit/tools/skips"
)

func main() { os.Exit(run(os.Args[1:])) }

// buildChecks constructs the ordered check list run() executes, in ci.yml's core-job
// step order. Split out from run() so main_test.go's TestCoreStepsAccountedFor can
// enumerate check NAMES without actually running go build / golangci-lint / etc — a
// gate.Check's Run closure only executes inside gate.RunAll, so calling buildChecks
// itself does no I/O.
func buildChecks(root string, fast bool) []gate.Check {
	checks := []gate.Check{
		{Name: "gofmt", Run: func() gate.Cell { return checkGofmt(root) }},
		{Name: "build", Run: func() gate.Cell { return goStep(root, "build", nil, "build", "./...") }},
		{Name: "vet", Run: func() gate.Cell { return goStep(root, "vet", nil, "vet", "./...") }},
		// Not core-job steps — see main.go's CROSS-VET paragraph. Placed here because they
		// are seconds-cheap and catch a whole-package build break, so failing now beats
		// failing after golangci-lint and go test have run.
		{Name: "cross-vet windows", Run: func() gate.Cell {
			return goStep(root, "cross-vet windows", []string{"GOOS=windows", "GOARCH=amd64"}, "vet", "./...")
		}},
		{Name: "cross-vet linux", Run: func() gate.Cell {
			return goStep(root, "cross-vet linux", []string{"GOOS=linux", "GOARCH=amd64"}, "vet", "./...")
		}},
		{Name: "golangci-lint", Run: func() gate.Cell { return checkLint(root) }},
		{Name: "no cgo deps in core graph", Run: func() gate.Cell { return checkNoCgoDeps(root) }},
		// Redundant with "build" above now that gpumod.Exec pins CGO_ENABLED=0 for every
		// step — kept as its own labeled cell so the report names the specific ci.yml
		// step it mirrors ("build with cgo disabled"), not because it needs its own
		// env override anymore.
		{Name: "build (cgo-free)", Run: func() gate.Cell { return goStep(root, "build (cgo-free)", nil, "build", "./...") }},
	}
	if !fast {
		checks = append(checks,
			gate.Check{Name: "go test", Run: func() gate.Cell { return goTest(root) }},
			gate.Check{Name: "test (aikit_checks)", Run: func() gate.Cell {
				return goStep(root, "test (aikit_checks)", nil, "test", "-tags", "aikit_checks", "./linalg/")
			}},
		)
	}
	return checks
}

func run(args []string) int {
	fast := len(args) > 0 && args[0] == "--fast"
	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Println(gate.Verdict(gate.Inconclusive, "cannot locate repo root: "+err.Error()))
		return 2
	}
	p := gpumod.Provenance(root)
	fmt.Printf("aikit preflight — %s%s — root module\n\n", p.Commit, p.Dirty)

	if fast {
		fmt.Println("  (--fast: skipping go test / aikit_checks; CI runs go test with -race)")
	}
	cells := gate.RunAll(buildChecks(root, fast))

	for _, c := range cells {
		fmt.Printf("  %-25s %-52s %s\n", c.Name, "["+c.Field("env")+"]", statusWord(c.Outcome))
		for _, f := range c.Fields {
			if f.Key == "line" {
				fmt.Printf("      %s\n", f.State)
			}
		}
	}
	rep := gate.ReconcileWith(cells, gate.FailWins)

	fmt.Println()
	switch rep.Outcome {
	case gate.Fail:
		fmt.Println(gate.Verdict(gate.Fail, fmt.Sprintf("%d check(s) failed — fix before pushing", rep.Fail)))
		return 1
	case gate.Inconclusive:
		fmt.Println(gate.Verdict(gate.Inconclusive, fmt.Sprintf("%d check(s) could not be run (external tool unavailable)", rep.Incon)))
		return 2
	}
	fmt.Println(gate.Verdict(gate.OK, fmt.Sprintf("%d/%d clean", rep.Pass, rep.Total)))
	fmt.Println("         CI also runs go test WITH -race; fuzz-smoke/arm64/treesitter/vulncheck/")
	fmt.Println("         perf-smoke/scripts-boundary/gpu-pins/gpu-kernels are separate CI jobs")
	fmt.Println("         (windows: its vet step IS covered above by cross-vet windows; its build/test are not).")
	return 0
}

// checkNoCgoDeps mirrors ci.yml's "no cgo deps in core graph": the core module must stay
// pure-Go — no cgo WebGPU, no tree-sitter (migration-plan §7/§9). Those live in the
// goinfer/gpu and aikit/chunk/treesitter modules respectively.
func checkNoCgoDeps(root string) gate.Cell {
	out, rc := gpumod.Exec(root, "", nil, "go", "list", "-deps", "./...")
	if rc != 0 {
		c := cell("no cgo deps in core graph", gate.Fail, "GOWORK=off CGO_ENABLED=0")
		c.Fields = append(c.Fields, headLines(out, 12)...)
		return c
	}
	var leaked []string
	for ln := range strings.SplitSeq(out, "\n") {
		low := strings.ToLower(ln)
		if strings.Contains(low, "webgpu") || strings.Contains(low, "gotreesitter") {
			leaked = append(leaked, ln)
		}
	}
	if len(leaked) > 0 {
		c := cell("no cgo deps in core graph", gate.Fail, "GOWORK=off CGO_ENABLED=0")
		for _, l := range leaked {
			c.Fields = append(c.Fields, gate.Field{Key: "line", State: l})
		}
		return c
	}
	return cell("no cgo deps in core graph", gate.OK, "GOWORK=off CGO_ENABLED=0")
}

// checkGofmt: gofmt reports by printing filenames and still exits 0, so EMPTINESS is the
// check. .venv is not Go source.
func checkGofmt(root string) gate.Cell {
	out, _ := gpumod.Exec(root, "", nil, "gofmt", "-l", ".")
	var bad []string
	for ln := range strings.SplitSeq(out, "\n") {
		ln = strings.TrimSpace(ln)
		if ln != "" && !strings.HasPrefix(ln, ".venv/") {
			bad = append(bad, ln)
		}
	}
	if len(bad) > 0 {
		c := cell("gofmt", gate.Fail, "none")
		for _, b := range bad {
			c.Fields = append(c.Fields, gate.Field{Key: "line", State: b})
		}
		return c
	}
	return cell("gofmt", gate.OK, "none")
}

// goStep runs a `go` subcommand via gpumod.Exec (GOWORK=off + CGO_ENABLED=0, plus any extra
// env), recording the env it ran under. A non-zero exit is a FAIL with the first lines of
// output.
func goStep(root, name string, extraEnv []string, args ...string) gate.Cell {
	// gpumod.Exec sets GOWORK=off and CGO_ENABLED=0 itself; env here is for the reported
	// cell field, not a second copy passed to it.
	env := append([]string{"GOWORK=off", "CGO_ENABLED=0"}, extraEnv...)
	out, rc := gpumod.Exec(root, "", extraEnv, "go", args...)
	if rc != 0 {
		c := cell(name, gate.Fail, strings.Join(env, " "))
		c.Fields = append(c.Fields, headLines(out, 12)...)
		return c
	}
	return cell(name, gate.OK, strings.Join(env, " "))
}

// goTest runs `go test ./...` through the skip census (AK3), so a green here reports the skip
// tally by reason as a denominator rather than folding skips into a bare pass. Output is
// captured (preflight stays concise); on failure the failing lines are surfaced, and either
// way the census summary is printed under the row.
func goTest(root string) gate.Cell {
	var buf bytes.Buffer
	res, _ := skips.Run(root, []string{"GOWORK=off"}, []string{"./..."}, &buf)
	outcome := gate.OK
	if res.ExitCode != 0 {
		outcome = gate.Fail
	}
	c := cell("go test", outcome, "GOWORK=off")
	if outcome == gate.Fail {
		for ln := range strings.SplitSeq(buf.String(), "\n") {
			if strings.HasPrefix(ln, "---") || strings.HasPrefix(ln, "FAIL") || strings.HasPrefix(ln, "panic") {
				c.Fields = append(c.Fields, gate.Field{Key: "line", State: strings.TrimRight(ln, "\r")})
				if countLines(c) >= 8 {
					break
				}
			}
		}
	}
	for ln := range strings.SplitSeq("skip census — "+res.Summary(), "\n") {
		c.Fields = append(c.Fields, gate.Field{Key: "line", State: ln})
	}
	return c
}

// checkLint runs the pinned golangci-lint on the ROOT module. A preflight `version` run
// separates "the linter could not be built/run" (INCONCLUSIVE — an external tool the check
// depends on) from "the linter found issues" (FAIL). Build-pinned via `go run @v2.13.0`.
func checkLint(root string) gate.Cell {
	if _, rc := gpumod.Exec(root, "", nil, "go", "run", gpumod.GolangciLint, "version"); rc != 0 {
		return cell("golangci-lint", gate.Inconclusive, "GOWORK=off CGO_ENABLED=0")
	}
	// CANARY: a "clean" is trusted only after the linter flags its fixture (see tools/canary).
	cout, _ := gpumod.Exec(filepath.Join(root, canary.FixturesDir), "", nil, "go", "run", gpumod.GolangciLint, "run", "--build-tags", "canaryfixture")
	if res := canary.CheckGolangci(cout); !res.Fired {
		c := cell("golangci-lint", gate.Inconclusive, "GOWORK=off CGO_ENABLED=0")
		c.Fields = append(c.Fields, gate.Field{Key: "line", State: "CANNOT-EVALUATE — " + res.Reason})
		return c
	}
	out, rc := gpumod.Exec(root, "", nil, "go", "run", gpumod.GolangciLint, "run")
	if rc != 0 {
		c := cell("golangci-lint", gate.Fail, "GOWORK=off CGO_ENABLED=0")
		c.Fields = append(c.Fields, headLines(out, 12)...)
		return c
	}
	return cell("golangci-lint", gate.OK, "GOWORK=off CGO_ENABLED=0")
}

func cell(name string, o gate.Outcome, env string) gate.Cell {
	return gate.Cell{Name: name, Outcome: o, Fields: []gate.Field{{Key: "env", State: env}}}
}

func statusWord(o gate.Outcome) string {
	switch o {
	case gate.OK:
		return "ok"
	case gate.Fail:
		return "FAIL"
	case gate.Inconclusive:
		return "INCONCLUSIVE (tool unavailable)"
	default:
		return string(o)
	}
}

func headLines(out string, n int) []gate.Field {
	var fs []gate.Field
	for ln := range strings.SplitSeq(out, "\n") {
		if strings.TrimSpace(ln) == "" {
			continue
		}
		fs = append(fs, gate.Field{Key: "line", State: strings.TrimRight(ln, "\r")})
		if len(fs) >= n {
			break
		}
	}
	return fs
}

func countLines(c gate.Cell) int {
	n := 0
	for _, f := range c.Fields {
		if f.Key == "line" {
			n++
		}
	}
	return n
}
