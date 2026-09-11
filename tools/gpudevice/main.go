// Command gpudevice is the DEVICE half of the gpu/ pre-tag ritual — the half that needs a
// GPU. It replaces scripts/gpu_device.sh: run every gpu module's test suite on hardware and
// print a verdict for the tag message. Run it on BOTH a Mac and an NVIDIA box; a gpu/ tag
// needs both lines.
//
// THE ARITHMETIC IS THE CONTRACT. Each of the nine modules resolves to one of three states,
// and the split is the point:
//   - ok      — tests ran and passed, with the passed/skipped counts shown (a module whose
//     every test skipped also prints `ok` from `go test`, so the counts matter).
//   - n/a     — no buildable packages on this OS (the four *metal on Linux, the four *cuda on
//     darwin). Excluded from the tally, not counted red or green — which is what
//     makes visible that NO SINGLE MACHINE COVERS ALL NINE.
//   - FAIL    — tests failed, or (a distinct case) `go list` could not even load the
//     packages, reported with the loader's own message rather than excused as a
//     platform difference.
//
// Applicability is decided by `go list` — exit 0 with no output means nothing buildable here
// — NOT by grepping `go test` output for English, which broke on darwin once already.
//
// HONESTY, carried from the shell: a SKIP IS NOT A PASS — a module whose every test skipped
// is reported SKIPPED, and if NO backend is present at all (every device test skips) the run
// is INCONCLUSIVE, never PASS. The verdict names a commit, so a dirty tree is INCONCLUSIVE.
// A real test FAILURE outranks both — a broken test is worse news than a dirty tree.
//
// Usage:  go run -C tools ./gpudevice
package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

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
	box := p.Host + " " + p.OSName + "/" + p.Arch
	backend := detectBackend(p.OSName)
	fmt.Printf("aikit gpu device tests — %s%s — %s — backend:%s — %s\n\n", p.Commit, p.Dirty, box, backend, p.Date)

	// The matrix is DATA: one Check per module, run by gate.RunAll so one module's failure
	// never aborts the rest and the count is never lost.
	checks := make([]gate.Check, 0, len(mods))
	for _, m := range mods {
		checks = append(checks, gate.Check{Name: m, Run: func() gate.Cell { return checkModule(root, m) }})
	}
	cells := gate.RunAll(checks)

	var na, skipped []string
	for _, c := range cells {
		status := c.Field("status")
		fmt.Printf("  %-24s %-7s %s\n", c.Name, status, c.Field("inline"))
		for _, f := range c.Fields {
			if f.Key == "line" {
				fmt.Printf("      %s\n", f.State)
			}
		}
		switch {
		case c.Outcome == gate.NA:
			na = append(na, c.Name)
		case status == "SKIPPED":
			skipped = append(skipped, c.Name)
		}
	}

	rep := gate.ReconcileWith(cells, gate.FailWins)
	applic := rep.Total - rep.NA

	fmt.Println()
	if rep.Fail > 0 {
		var failed []string
		for _, c := range cells {
			if c.Outcome == gate.Fail {
				failed = append(failed, c.Name)
			}
		}
		fmt.Println(gate.Verdict(gate.Fail, fmt.Sprintf("device tests red in: %s (at %s%s, %s, backend:%s, %s)",
			strings.Join(failed, " "), p.Commit, p.Dirty, box, backend, p.Date)))
		return 1
	}
	if len(skipped) > 0 {
		fmt.Printf("NOTE: ran but every test skipped — %s\n\n", strings.Join(skipped, " "))
	}
	if len(na) > 0 {
		fmt.Printf("NOT APPLICABLE on this platform — %s\n", strings.Join(na, " "))
		fmt.Println("      Those need a run on the other OS. This verdict does not cover them.")
		fmt.Println()
	}
	if backend == "none" {
		fmt.Println(gate.Verdict(gate.Inconclusive, fmt.Sprintf("%d applicable modules ran green, but NO GPU BACKEND was detected here,", applic)))
		fmt.Println("so every device test skipped. This says the code compiles and the CPU paths agree;")
		fmt.Println("it says nothing about Metal or CUDA. Run it on the hardware before tagging.")
		return 2
	}
	if p.Dirty != "" {
		fmt.Println(gate.Verdict(gate.Inconclusive, fmt.Sprintf("%d/%d applicable modules green on backend:%s, but the working tree is DIRTY.", applic, applic, backend)))
		fmt.Printf("The verdict names %s and the tree is not %s. Commit, then re-run before tagging.\n", p.Commit, p.Commit)
		return 2
	}
	// A SKIP IS NOT A PASS, in the VERDICT LINE too (audit G-07). The per-module
	// row already said SKIPPED and a NOTE already listed them, but the pasted
	// verdict counted them as green — and the verdict is the part that ends up
	// in a tag message and gets read later. A module whose every test skipped
	// for a missing fixture is evidence of nothing about that module.
	//
	// Measured on this repo the day this was written: gpu/visionmetal ran 0
	// tests and gpu/qwenmetal ran 1, both for absent testdata fixtures, while
	// the suite printed ok and this line would have called them green.
	summary, ok := deviceVerdict(applic, skipped, p.Commit, box, backend, p.Date)
	if !ok {
		fmt.Println(gate.Verdict(gate.Inconclusive, summary))
		return 2
	}
	fmt.Println(gate.Verdict(gate.OK, summary))
	fmt.Printf("         %d of %d not applicable on this platform.\n", rep.NA, rep.Total)
	fmt.Printf("Paste that line into the tag message. It covers ONLY backend:%s — a gpu/ tag\n", backend)
	fmt.Println("needs the other platform's verdict line too.")
	return 0
}

// checkModule classifies one module. `go list` decides applicability BEFORE `go test`, so a
// not-applicable module (no packages on this OS) is never confused with a broken one.
func checkModule(root, m string) gate.Cell {
	pkgs, lerr, lrc := gpumod.GoSep(root, m, nil, "list", "./...")
	if lrc == 0 && strings.TrimSpace(pkgs) == "" {
		return devCell(m, gate.NA, "n/a", "(no buildable packages on this platform)")
	}
	if lrc != 0 {
		c := devCell(m, gate.Fail, "FAIL", "(cannot load packages)")
		c.Fields = append(c.Fields, follow(lerr, 4)...)
		return c
	}
	out, rc := gpumod.Go(root, m, nil, "test", "-v", "./...")
	if rc != 0 {
		c := devCell(m, gate.Fail, "FAIL", "")
		c.Fields = append(c.Fields, failLines(out)...)
		return c
	}
	ran, skip := countTopLevel(out)
	if ran == 0 {
		return devCell(m, gate.OK, "SKIPPED", fmt.Sprintf("(0 passed, %d skipped)", skip))
	}
	return devCell(m, gate.OK, "ok", fmt.Sprintf("(%d passed, %d skipped)", ran, skip))
}

func detectBackend(osName string) string {
	if osName == "Darwin" {
		return "metal"
	}
	if err := exec.Command("nvidia-smi", "-L").Run(); err == nil {
		return "cuda"
	}
	return "none"
}

// --- cell plumbing ------------------------------------------------------------------

func devCell(name string, o gate.Outcome, status, inline string) gate.Cell {
	return gate.Cell{Name: name, Outcome: o, Fields: []gate.Field{
		{Key: "status", State: status},
		{Key: "inline", State: inline},
	}}
}

// follow captures the first n non-empty lines of a loader error as indented follow-up.
func follow(out string, n int) []gate.Field {
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

// failLines captures the first five ^--- FAIL / FAIL / panic lines of a test failure.
func failLines(out string) []gate.Field {
	var fs []gate.Field
	for ln := range strings.SplitSeq(out, "\n") {
		if strings.HasPrefix(ln, "--- FAIL") || strings.HasPrefix(ln, "FAIL") || strings.HasPrefix(ln, "panic") {
			fs = append(fs, gate.Field{Key: "line", State: strings.TrimRight(ln, "\r")})
			if len(fs) >= 5 {
				break
			}
		}
	}
	return fs
}

// countTopLevel counts top-level (unindented) --- PASS / --- SKIP lines, matching the shell's
// `grep -c '^--- PASS'`: a module whose every top-level test skipped has ran == 0.
func countTopLevel(out string) (ran, skip int) {
	for ln := range strings.SplitSeq(out, "\n") {
		switch {
		case strings.HasPrefix(ln, "--- PASS"):
			ran++
		case strings.HasPrefix(ln, "--- SKIP"):
			skip++
		}
	}
	return
}

// deviceVerdict builds the pasted verdict line and reports whether it is a pass.
//
// An all-skip module is NOT green (audit G-07). The per-module row already said
// SKIPPED and a NOTE already listed them, but the verdict counted them toward
// "N/N green" — and the verdict is the part that ends up in a tag message and
// gets read months later. A module whose every test skipped for an absent
// fixture is evidence of nothing about that module.
//
// Measured on this repo the day this was written: gpu/visionmetal ran 0 tests
// and gpu/qwenmetal 1, both for missing testdata fixtures, while the suite
// printed ok and this line would have called them green.
func deviceVerdict(applicable int, skipped []string, commit, box, backend, date string) (string, bool) {
	green := applicable - len(skipped)
	if green <= 0 {
		return fmt.Sprintf("0/%d applicable gpu modules actually exercised a device at %s — every test skipped (%s)",
			applicable, commit, strings.Join(skipped, " ")), false
	}
	s := fmt.Sprintf("%d/%d applicable gpu modules green at %s (%s, backend:%s, %s)", green, applicable, commit, box, backend, date)
	if len(skipped) > 0 {
		s += fmt.Sprintf(" — %d ran but skipped every test (%s), NOT counted green", len(skipped), strings.Join(skipped, " "))
	}
	return s, true
}
