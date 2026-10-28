// Command gpupins keeps the four GPU backend modules that stay in aikit (gpu/anncuda, gpu/annmetal, gpu/enccuda,
// gpu/encmetal) frozen (owner decision SD2 of 2026-10-26): each must `require github.com/townsendmerino/aikit/gpu` at the
// latest gpu/v0.35.x tag, and `require github.com/townsendmerino/aikit` at frozenRoot, the tag they were frozen at. They
// no longer track the newest root tag, so a root require that moves is a FAIL, not drift for --fix to follow.
//
// The four tower backends (gpu/qwencuda, gpu/qwenmetal, gpu/visioncuda, gpu/visionmetal) moved into goinfer and are
// retired here with no further tags, so they are outside the invariant, as are chunk/treesitter, the examples and
// benchmarks modules, and gpu/webgpu.
//
// `--fix` rewrites the four gpu requires to the latest gpu/v0.35.x tag, and carries examples/gpu-ann along (it requires
// two of the four, so MVS needs its require to move with them; it is not judged). It never writes a root require.
//
// A require is static go.mod text, so the invariant is gated rather than derived; the replace directives stay, since Go
// ignores them in a dependency's go.mod.
//
// Usage:
//
//	go run -C tools ./gpupins          # check; drift is a FAIL
//	go run -C tools ./gpupins --fix    # rewrite the four requires to the latest gpu/v0.35.x tag
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/townsendmerino/aikit/tools/gate"
	"github.com/townsendmerino/aikit/tools/gpumod"
)

const (
	rootMod = "github.com/townsendmerino/aikit"
	gpuMod  = "github.com/townsendmerino/aikit/gpu"
)

// backends are the GPU backend modules that stay in aikit, enumerated rather than discovered: a module joining the
// invariant is a deliberate edit here.
var backends = []string{
	"gpu/anncuda", "gpu/annmetal",
	"gpu/enccuda", "gpu/encmetal",
}

// dependents are in-tree modules that are NOT part of the pin invariant — never checked,
// never in the verdict — but whose aikit requires `--fix` must carry along anyway, because
// MVS forces them up the moment the backends move. examples/gpu-ann requires anncuda and
// annmetal (`replace`-backed at ../..), so a `--fix` that bumps those to a new root version
// and stops leaves gpu-ann requiring the OLD one, below its own dependencies: go.mod untidy,
// `go list` broken, govulncheck unable to load it. Enumerated for the same reason `backends`
// is — a new example that acquires a backend dependency is a deliberate edit here, not a
// discovery, and getting it wrong is a red CI run, not a silent wrong answer.
var dependents = []string{"examples/gpu-ann"}

// reGpuTag is the frozen device-module series the backends stay on (SD2).
var reGpuTag = regexp.MustCompile(`^gpu/v0\.35\.\d+$`)

// frozenRoot is the root aikit tag the four backends require since SD2 froze them. Changing it is a decision of the
// owner's, the same as SD2 itself, not a release step.
const frozenRoot = "v1.75.0"

func main() { os.Exit(run(os.Args[1:])) }

func run(args []string) int {
	fix := len(args) > 0 && args[0] == "--fix"

	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Println("gpupins: INCONCLUSIVE — cannot locate repo root: " + err.Error())
		return 2
	}

	// The targets the four must name. A tag we cannot resolve is INCONCLUSIVE — the gate could not judge (and --fix has
	// nothing to write) — never a silent pass. The root target is the freeze, not a tag read from git.
	wantRoot := frozenRoot
	wantGpuTag := latestTag(root, "gpu/v0.35.*", reGpuTag)
	if wantGpuTag == "" {
		fmt.Println("gpupins: INCONCLUSIVE — could not resolve the latest gpu/v0.35.x tag from git " +
			"(need `git tag`; CI must checkout with fetch-depth: 0)")
		return 2
	}
	wantGpu := strings.TrimPrefix(wantGpuTag, "gpu/") // the require names the version, not the tag path

	p := gpumod.Provenance(root)
	fmt.Printf("aikit gpu backend pins — %s%s — %s\n", p.Commit, p.Dirty, p.Date)
	fmt.Printf("target: %s %s, %s %s (frozen, SD2)\n\n", gpuMod, wantGpu, rootMod, wantRoot)

	if fix {
		return runFix(root, "", wantGpu) // the root require is frozen: --fix never moves it
	}

	checks := make([]gate.Check, 0, len(backends))
	for _, m := range backends {
		checks = append(checks, gate.Check{Name: m, Run: func() gate.Cell { return checkModule(root, m, wantRoot, wantGpu) }})
	}
	cells := gate.RunAll(checks)

	for _, c := range cells {
		fmt.Printf("  %-22s %s\n", c.Name, statusLine(c))
	}
	rep := gate.ReconcileWith(cells, gate.FailWins)

	fmt.Println()
	switch rep.Outcome {
	case gate.Fail:
		if rootMoved(cells) {
			fmt.Printf("gpupins: a root aikit require moved off %s. The four are frozen there (SD2): restore it by hand; --fix does not move it.\n", frozenRoot)
		} else {
			fmt.Printf("gpupins: %d/%d backend(s) drift from the latest gpu/v0.35.x tag — run `go run -C tools ./gpupins --fix`.\n", rep.Fail, rep.Total)
		}
		fmt.Println(gate.Verdict(gate.Fail, fmt.Sprintf("%d/%d pin aikit %s and the latest gpu/v0.35.x tag", rep.Pass, rep.Total, frozenRoot)))
		return 1
	case gate.Inconclusive:
		fmt.Println(gate.Verdict(gate.Inconclusive, "a backend go.mod could not be read"))
		return 2
	}
	fmt.Println(gate.Verdict(gate.OK, fmt.Sprintf("all %d backends pin aikit %s and gpu %s", rep.Total, wantRoot, wantGpu)))
	return 0
}

// rootMoved reports whether any cell's root require is not the freeze.
func rootMoved(cells []gate.Cell) bool {
	for _, c := range cells {
		for _, f := range c.Fields {
			if f.Key == "root" && f.State != "ok" {
				return true
			}
		}
	}
	return false
}

// checkModule reads one backend's go.mod and compares its two aikit requires to the targets.
func checkModule(root, mod, wantRoot, wantGpu string) gate.Cell {
	path := filepath.Join(root, mod, "go.mod")
	data, err := os.ReadFile(path)
	if err != nil {
		return gate.Cell{Name: mod, Outcome: gate.Inconclusive, Fields: []gate.Field{{Key: "read", State: "n/a", Detail: err.Error()}}}
	}
	gotRoot, gotGpu := requires(string(data))

	c := gate.Cell{Name: mod, Outcome: gate.OK}
	if wantRoot != "" {
		c.Fields = append(c.Fields, pinField("root", gotRoot, wantRoot))
	}
	c.Fields = append(c.Fields, pinField("gpu", gotGpu, wantGpu))
	for _, f := range c.Fields {
		if f.State != "ok" {
			c.Outcome = gate.Fail
		}
	}
	return c
}

// pinField reports one require as ok / missing / a `got != want` drift.
func pinField(key, got, want string) gate.Field {
	switch {
	case got == "":
		return gate.Field{Key: key, State: "MISSING", Detail: "no require " + want}
	case got != want:
		return gate.Field{Key: key, State: "DRIFT", Detail: got + " != " + want}
	default:
		return gate.Field{Key: key, State: "ok"}
	}
}

// requires extracts the versions of the two aikit modules from go.mod text. It reads the
// require lines directly rather than pulling in golang.org/x/mod/modfile — the tools stay
// stdlib-only, and the lines are unambiguous once trimmed. The `replace` lines begin with
// "replace" (not the bare module path), so they are ignored; the aikit/gpu path is checked
// before the aikit path because the former is a prefix-extension of the latter.
func requires(gomod string) (root, gpu string) {
	for ln := range strings.SplitSeq(gomod, "\n") {
		t := strings.TrimSpace(ln)
		switch {
		case strings.HasPrefix(t, gpuMod+" "):
			gpu = versionField(t, gpuMod)
		case strings.HasPrefix(t, rootMod+" "):
			root = versionField(t, rootMod)
		}
	}
	return root, gpu
}

// versionField returns the version token following modPath on a trimmed require line.
func versionField(trimmed, modPath string) string {
	rest := strings.TrimSpace(strings.TrimPrefix(trimmed, modPath))
	if i := strings.IndexAny(rest, " \t"); i >= 0 {
		rest = rest[:i] // drop a trailing // comment or anything after the version
	}
	return rest
}

// runFix rewrites the two aikit requires in each backend's go.mod to the targets, preserving
// indentation and any trailing comment, and reports what it changed. This IS RELEASING.md
// step 3: the hand-edit of eight files becomes one command that reads the tags and writes
// them, so the version is derived from git rather than restated from memory.
func runFix(root, wantRoot, wantGpu string) int {
	changed, failed := 0, 0
	for _, mod := range backends {
		fixModule(root, mod, wantRoot, wantGpu, &changed, &failed)
	}
	// The dependents are printed under their own heading so the report never reads as though
	// the pin invariant grew to nine modules — it did not; these are carried, not judged.
	if len(dependents) > 0 {
		fmt.Println("\n  carried along (not part of the invariant; MVS forces them):")
		for _, mod := range dependents {
			fixModule(root, mod, wantRoot, wantGpu, &changed, &failed)
		}
	}
	fmt.Println()
	if failed > 0 {
		fmt.Println(gate.Verdict(gate.Inconclusive, fmt.Sprintf("%d rewritten, %d could not be written", changed, failed)))
		return 2
	}
	fmt.Println(gate.Verdict(gate.OK, fmt.Sprintf("%d module(s) rewritten to gpu %s", changed, wantGpu)))
	return 0
}

// fixModule rewrites one go.mod's aikit requires in place, reporting one line about what it
// did and folding the result into the caller's counters. A module that has no aikit require
// at all is simply "already current" — rewrite makes no edits and none are needed.
func fixModule(root, mod, wantRoot, wantGpu string, changed, failed *int) {
	path := filepath.Join(root, mod, "go.mod")
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("  %-22s could not read: %v\n", mod, err)
		*failed++
		return
	}
	out, edits := rewrite(string(data), wantRoot, wantGpu)
	if len(edits) == 0 {
		fmt.Printf("  %-22s already current\n", mod)
		return
	}
	if err := os.WriteFile(path, []byte(out), 0o644); err != nil {
		fmt.Printf("  %-22s could not write: %v\n", mod, err)
		*failed++
		return
	}
	fmt.Printf("  %-22s %s\n", mod, strings.Join(edits, "; "))
	*changed++
}

// rewrite returns go.mod text with the two aikit require versions set to the targets (an empty target leaves that
// require alone), and a list of human-readable edits made. Lines are matched by trimmed prefix so indentation and
// trailing comments survive; only the version token is replaced.
func rewrite(gomod, wantRoot, wantGpu string) (string, []string) {
	lines := strings.Split(gomod, "\n")
	var edits []string
	for i, ln := range lines {
		indent := ln[:len(ln)-len(strings.TrimLeft(ln, " \t"))]
		t := strings.TrimSpace(ln)
		switch {
		case strings.HasPrefix(t, gpuMod+" "):
			if got := versionField(t, gpuMod); got != wantGpu {
				lines[i] = setVersion(indent, gpuMod, t, wantGpu)
				edits = append(edits, "gpu "+got+"→"+wantGpu)
			}
		case strings.HasPrefix(t, rootMod+" ") && wantRoot != "":
			if got := versionField(t, rootMod); got != wantRoot {
				lines[i] = setVersion(indent, rootMod, t, wantRoot)
				edits = append(edits, "root "+got+"→"+wantRoot)
			}
		}
	}
	return strings.Join(lines, "\n"), edits
}

// setVersion rebuilds a require line with a new version, keeping indentation and any tokens
// after the version (e.g. a `// indirect` comment — not present on these two, but preserved).
func setVersion(indent, modPath, trimmed, ver string) string {
	rest := strings.TrimSpace(strings.TrimPrefix(trimmed, modPath))
	tail := ""
	if i := strings.IndexAny(rest, " \t"); i >= 0 {
		tail = strings.TrimSpace(rest[i:])
	}
	line := indent + modPath + " " + ver
	if tail != "" {
		line += " " + tail
	}
	return line
}

// latestTag returns the highest tag matching re, using git's version sort. The glob is a
// prefilter (git's fnmatch lets `*` cross `/`, so `gpu/v*` also matches `gpu/<name>/v…`);
// re is the real filter that keeps only the module-root series.
func latestTag(root, glob string, re *regexp.Regexp) string {
	out, _ := gpumod.Exec(root, "", nil, "git", "tag", "--list", glob, "--sort=-v:refname")
	for ln := range strings.SplitSeq(out, "\n") {
		t := strings.TrimSpace(ln)
		if re.MatchString(t) {
			return t
		}
	}
	return ""
}

func statusLine(c gate.Cell) string {
	parts := make([]string, 0, len(c.Fields))
	for _, f := range c.Fields {
		parts = append(parts, f.String())
	}
	word := "ok"
	if c.Outcome != gate.OK {
		word = string(c.Outcome)
	}
	return fmt.Sprintf("%-4s  %s", word, strings.Join(parts, "  "))
}

func orNone(s string) string {
	if s == "" {
		return "(none)"
	}
	return s
}
