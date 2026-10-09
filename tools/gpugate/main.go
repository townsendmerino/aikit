// Command gpugate is the mechanical, no-GPU half of the gpu/ pre-tag ritual. It replaces
// scripts/gpu_gate.sh: six declared groups, each expressed as a gate.Check over the shared
// runner (tools/gate), reconciled once. Run it, paste the VERDICT line into the tag message.
//
// WHY THIS IS THE GATE AND NOT CI. The device tests need a GPU and no CI runner has one, so
// the release ritual is the enforcement; this is the half that needs no hardware. The six
// groups: all nine gpu modules build cgo-free; gofmt+vet on gpu/; golangci-lint across the
// nine modules (the root CI job lints only the root module — `./...` stops at module
// boundaries); and the three kernel assertions (FMA lint, fma histogram, PTX reproducibility).
//
// CONTRACTS CARRIED FROM THE SHELL, verbatim in intent:
//   - A SKIP IS NOT A PASS. `go test` prints ok for a package whose tests all skipped; the
//     three kernel groups fail on a skip. The one legitimate skip — PTX reproducibility when
//     NVRTC is absent — is reported INCONCLUSIVE (could not judge), never PASS: still not
//     green, but distinguished from "reproducibility is broken" (FAIL).
//   - ...EXCEPT WHERE THE TOOL CANNOT EXIST. NVIDIA ships no NVRTC for Apple Silicon, so on
//     darwin ptx-repro is n/a (platform-keyed), excluded from the tally — not FAIL, or a Mac
//     could never be green. "Absent because unsupported" and "absent because uninstalled"
//     get different answers.
//   - THE TALLY CANNOT SILENTLY SHRINK. The shell reconciled declared groups against emitted
//     verdicts by hand because a group that died mid-block would vanish. tools/gate.RunAll
//     removes the failure mode at the source: a panicking check becomes a Fail cell, so all
//     six groups always emit exactly one cell — the count cannot be lost.
//   - PROVENANCE. The verdict names a commit; a dirty tree is INCONCLUSIVE, not PASS.
//
// The bash-3.2 portability constraint that shaped the shell (571982f) dies with it — Go has
// no such floor. That is a benefit of the move, not a contract to preserve.
//
// Usage:  go run -C tools ./gpugate
// Env:    NVRTC_LIB / CUDA_INC   override NVRTC discovery for the ptx-repro group (inherited
//
//	by the test's findNVRTC; see gpu/build_ptx.sh)
package main

import (
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/townsendmerino/aikit/tools/canary"
	"github.com/townsendmerino/aikit/tools/gate"
	"github.com/townsendmerino/aikit/tools/gpumod"
)

func main() { os.Exit(run()) }

func run() int {
	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Println(gate.Verdict(gate.Inconclusive, "cannot locate repo root: "+err.Error()))
		return 2
	}
	mods, err := gpumod.ModuleDirs(root)
	if err != nil || len(mods) == 0 {
		fmt.Println(gate.Verdict(gate.Fail, fmt.Sprintf("enumerating gpu modules: %v (found %d)", err, len(mods))))
		return 1
	}
	p := gpumod.Provenance(root)
	box := p.OSName + "/" + p.Arch
	fmt.Printf("aikit gpu gate — %s%s — %s — %s\n\n", p.Commit, p.Dirty, box, p.Date)

	// The six groups, as data. Order matches the shell for a line-by-line diff.
	checks := []gate.Check{
		{Name: "build", Run: func() gate.Cell { return checkBuild(root, mods) }},
		{Name: "fmt-vet", Run: func() gate.Cell { return checkFmtVet(root) }},
		{Name: "fma-lint", Run: func() gate.Cell { return kernelTest(root, "fma-lint", "TestKernelFMALint") }},
		{Name: "fma-histogram", Run: func() gate.Cell { return kernelTest(root, "fma-histogram", "TestGemvQuantFMAHistogram") }},
		{Name: "lint", Run: func() gate.Cell { return checkLint(root, mods) }},
		{Name: "ptx-repro", Run: func() gate.Cell { return checkPTXRepro(root, p.OSName) }},
	}
	cells := gate.RunAll(checks)

	var na []string
	for _, c := range cells {
		fmt.Printf("  %-14s %-4s %s\n", c.Name, word(c.Outcome), detail(c))
		if c.Outcome == gate.NA {
			na = append(na, c.Name)
		}
	}
	rep := gate.ReconcileWith(cells, gate.FailWins)
	applic := rep.Total - rep.NA

	fmt.Println()
	if len(na) > 0 {
		fmt.Printf("NOT APPLICABLE on this platform — %s\n", strings.Join(na, " "))
		fmt.Println("      This verdict does not cover them; a box that can run them must.")
		fmt.Println()
	}

	switch {
	case rep.Outcome == gate.Fail:
		fmt.Println(gate.Verdict(gate.Fail, fmt.Sprintf("%d of %d applicable groups red at %s%s (%s, %s)",
			rep.Fail, applic, p.Commit, p.Dirty, box, p.Date)))
		return 1
	case rep.Outcome == gate.Inconclusive:
		fmt.Println(gate.Verdict(gate.Inconclusive, fmt.Sprintf("%d of %d applicable groups could not be judged (NVRTC unfindable) at %s%s (%s, %s)",
			rep.Incon, applic, p.Commit, p.Dirty, box, p.Date)))
		fmt.Println("Install NVRTC (pip install nvidia-cuda-nvrtc-cu12==12.9.86) or set NVRTC_LIB, then re-run.")
		return 2
	case p.Dirty != "":
		fmt.Println(gate.Verdict(gate.Inconclusive, fmt.Sprintf("%d/%d applicable groups green, but the working tree is DIRTY.", applic, applic)))
		fmt.Printf("The verdict names %s and the tree is not %s. Commit, then re-run before tagging.\n", p.Commit, p.Commit)
		return 2
	}
	fmt.Println(gate.Verdict(gate.OK, fmt.Sprintf("%d/%d applicable groups green at %s (%s, %s)", applic, rep.Total, p.Commit, box, p.Date)))
	if rep.NA > 0 {
		fmt.Printf("         %d of %d not applicable here — %s\n", rep.NA, rep.Total, strings.Join(na, " "))
	}
	fmt.Println("Paste that line into the tag message. Device tests are the hand-run half — see RELEASING.md.")
	return 0
}

// (1) build — every gpu module builds cgo-free. A no-package module on this OS builds
// vacuously (rc 0), which is correct: this group is "nothing is broken", not "everything ran".
func checkBuild(root string, mods []string) gate.Cell {
	var bad []string
	for _, m := range mods {
		if _, rc := gpumod.Go(root, m, nil, "build", "./..."); rc != 0 {
			bad = append(bad, m)
		}
	}
	if len(bad) > 0 {
		return cell("build", gate.Fail, "cgo-free build failed: "+strings.Join(bad, " "))
	}
	return cell("build", gate.OK, fmt.Sprintf("%d modules build with CGO_ENABLED=0", len(mods)))
}

// (2) fmt-vet — gofmt and vet on the gpu module itself.
func checkFmtVet(root string) gate.Cell {
	unformatted, _ := gpumod.Exec(root, "gpu", nil, "gofmt", "-l", ".")
	vetout, vrc := gpumod.Go(root, "gpu", nil, "vet", "./...")
	if strings.TrimSpace(unformatted) != "" || vrc != 0 || strings.TrimSpace(vetout) != "" {
		return cell("fmt-vet", gate.Fail, fmt.Sprintf("gofmt:[%s] vet:[%s]", strings.TrimSpace(unformatted), firstLine(vetout)))
	}
	return cell("fmt-vet", gate.OK, "gofmt clean, go vet clean")
}

// golangciLint is the pinned linter (gpumod.GolangciLint) — see there for why the BUILD,
// not just the version, is pinned.
const golangciLint = gpumod.GolangciLint

// (3) lint — the pinned golangci-lint across the nine modules. A preflight `version` run
// separates "the linter could not be built or run" (INCONCLUSIVE — could not judge, e.g. no
// network to fetch it) from "the linter found issues" (FAIL); it also warms the build cache
// so the per-module runs are fast. Modules with no buildable packages on this OS (go list
// empty) have nothing to lint and are skipped.
func checkLint(root string, mods []string) gate.Cell {
	if out, rc := gpumod.Go(root, ".", nil, "run", golangciLint, "version"); rc != 0 {
		return cell("lint", gate.Inconclusive, "could not build the pinned golangci-lint: "+firstLine(out))
	}
	// CANARY: prove golangci-lint actually runs AND reports, with the SAME invocation the
	// per-module runs use, before any "clean" is trusted. A linter that cannot exec (a
	// cross-GOOS `go run` builds a binary for the wrong platform) or runs but checks nothing
	// flags no module — indistinguishable from a real pass without this positive control
	// (which the false-clean of 2026-08-13, twice in one day, is exactly the case for).
	cout, _ := gpumod.Go(root, canary.FixturesDir, nil, "run", golangciLint, "run", "--build-tags", "canaryfixture")
	if res := canary.CheckGolangci(cout); !res.Fired {
		return cell("lint", gate.Inconclusive, "CANNOT-EVALUATE — "+res.Reason)
	}
	var bad []string
	n := 0
	for _, m := range mods {
		pkgs, _, rc := gpumod.GoSep(root, m, nil, "list", "./...")
		if rc == 0 && strings.TrimSpace(pkgs) == "" {
			continue // n/a on this OS — nothing to lint
		}
		n++
		if _, rc := gpumod.Go(root, m, nil, "run", golangciLint, "run"); rc != 0 {
			bad = append(bad, m)
		}
	}
	if len(bad) > 0 {
		return cell("lint", gate.Fail, "golangci-lint issues in: "+strings.Join(bad, " "))
	}
	return cell("lint", gate.OK, fmt.Sprintf("%d applicable modules clean (pinned v2.14.0)", n))
}

var reTestLine = regexp.MustCompile(`_test\.go:[0-9]+`)

// kernelTest runs one -run assertion in gpu/ and classifies it. classifyKernel holds the
// decision (pure, so it is unit-tested) the way the shell's run_test did.
func kernelTest(root, name, pattern string) gate.Cell {
	out, rc := gpumod.Go(root, "gpu", nil, "test", "-count=1", "-run", pattern, "-v", "./...")
	return classifyKernel(name, pattern, out, rc)
}

// classifyKernel: a non-zero exit, no matching test, or ANY skip is a FAIL — a skip is not a
// pass, and these assertions have no legitimate skip.
func classifyKernel(name, pattern, out string, rc int) gate.Cell {
	nrun, npass, nskip := countTests(out)
	switch {
	case rc != 0:
		return cell(name, gate.Fail, testErrLine(out))
	case nrun == 0:
		return cell(name, gate.Fail, "no tests matched '"+pattern+"' — the assertion is gone or renamed")
	case nskip > 0:
		return cell(name, gate.Fail, fmt.Sprintf("%d skipped — a skip is not a pass", nskip))
	default:
		return cell(name, gate.OK, fmt.Sprintf("%d passed (%d run)", npass, nrun))
	}
}

// checkPTXRepro is the one group with a legitimate skip. On darwin NVRTC cannot exist → n/a
// (platform-keyed). Elsewhere the test is run WITHOUT AIKIT_REQUIRE_PTX_REPRO so its own
// findNVRTC (honoring NVRTC_LIB) decides, and classifyPTX maps the result.
func checkPTXRepro(root, osName string) gate.Cell {
	if osName == "Darwin" {
		return cell("ptx-repro", gate.NA, "no NVRTC for darwin (NVIDIA ships none for Apple Silicon)")
	}
	out, rc := gpumod.Go(root, "gpu", nil, "test", "-count=1", "-run", "TestPTXReproducible", "-v", "./...")
	return classifyPTX(out, rc)
}

// classifyPTX: unlike the other kernel groups, a skip here means NVRTC is unfindable →
// INCONCLUSIVE (could not judge reproducibility), NOT FAIL. Still never PASS — a skip is not
// a pass — but distinguished from "reproducibility is broken" (a real FAIL).
func classifyPTX(out string, rc int) gate.Cell {
	nrun, npass, nskip := countTests(out)
	switch {
	case rc != 0:
		return cell("ptx-repro", gate.Fail, testErrLine(out))
	case nrun == 0:
		return cell("ptx-repro", gate.Fail, "no tests matched 'TestPTXReproducible' — the assertion is gone or renamed")
	case nskip > 0:
		return cell("ptx-repro", gate.Inconclusive, "NVRTC unfindable — reproducibility not checked")
	default:
		return cell("ptx-repro", gate.OK, fmt.Sprintf("%d passed (%d run)", npass, nrun))
	}
}

// --- helpers ----------------------------------------------------------------------

func cell(name string, o gate.Outcome, detail string) gate.Cell {
	return gate.Cell{Name: name, Outcome: o, Fields: []gate.Field{{Key: "detail", State: detail}}}
}

func detail(c gate.Cell) string {
	if len(c.Fields) > 0 {
		return c.Fields[0].State
	}
	return ""
}

func word(o gate.Outcome) string {
	switch o {
	case gate.OK:
		return "PASS"
	case gate.NA:
		return "n/a"
	case gate.Fail:
		return "FAIL"
	default:
		return string(o)
	}
}

func countTests(out string) (nrun, npass, nskip int) {
	for ln := range strings.SplitSeq(out, "\n") {
		t := strings.TrimSpace(ln)
		switch {
		case strings.HasPrefix(ln, "=== RUN"):
			nrun++
		case strings.HasPrefix(t, "--- PASS"):
			npass++
		case strings.HasPrefix(t, "--- SKIP"):
			nskip++
		}
	}
	return
}

func testErrLine(out string) string {
	for ln := range strings.SplitSeq(out, "\n") {
		if reTestLine.MatchString(ln) {
			s := strings.TrimSpace(ln)
			if len(s) > 90 {
				s = s[:90]
			}
			return s
		}
	}
	return firstLine(out)
}

func firstLine(s string) string {
	for ln := range strings.SplitSeq(s, "\n") {
		if strings.TrimSpace(ln) != "" {
			s := strings.TrimSpace(ln)
			if len(s) > 90 {
				s = s[:90]
			}
			return s
		}
	}
	return ""
}
