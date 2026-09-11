// Command releasegate validates that the repo is ready to release vX.Y.Z — the root-module
// release gate, ported from scripts/release-gate.sh onto tools/gate. release.yml re-runs it
// on the tag before publishing, so it is the last check between a commit and a Go module
// version that is immutable once the proxy fetches it.
//
// FOUR CHECKS:
//  1. CHANGELOG.md has a `## [X.Y.Z]` section AND a `[X.Y.Z]:` compare link — a release
//     deliverable; its failure messages name exactly what is missing.
//  2. golangci-lint clean against .golangci.yml, build-pinned via `go run @v2.13.0` (the
//     same the gpu gate, preflight, and CI resolve to) so a local gate reports identically.
//  3. apidiff shows NO incompatible change in any Hard-tier package vs the previous tag —
//     the 1.0 compatibility bar — with documented-Experimental symbols/members exempted.
//  4. The core module pulls no external dependency beyond golang.org/x/text and
//     golang.org/x/sys (darwin-only madvise) — the cgo-free, dependency-light invariant.
//
// EXTERNAL TOOLS ARE INVOKED, NOT DEPENDED ON. golangci-lint and apidiff are run via
// `go run pkg@ver`; nothing enters tools/go.mod, and either being unbuildable is
// INCONCLUSIVE — could not judge — never a pass. Previous-tag resolution takes the highest
// v* by version, excluding the one being released, so the retracted v1.8.0 is never picked.
//
// Usage:  go run -C tools ./releasegate <version, e.g. 1.4.0 (no leading v)>
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/townsendmerino/aikit/tools/canary"
	"github.com/townsendmerino/aikit/tools/gate"
	"github.com/townsendmerino/aikit/tools/gpumod"
)

const rootMod = "github.com/townsendmerino/aikit"

// hardPkgs are the Hard-tier packages under the 1.0 compatibility guarantee (README).
var hardPkgs = []string{"topk", "ann", "bm25", "fuse", "embed", "encoder", "chunk"}

// experimentalSyms: symbols documented Experimental that LIVE INSIDE a Hard-tier package —
// their leading symbol name is the whole match, so an apidiff line led by one is exempt.
var experimentalSyms = map[string][]string{
	"encoder": {"Backend", "RegisterBackend", "NewBackend", "LoadQ8", "ModelQ8", "WeightsQ8", "LayerWeightsQ8", "LoadBERT", "BERT", "LoadSPLADE", "SPLADE", "LoadCrossEncoder", "CrossEncoder"},
	"ann":     {"FlatBinary", "NewFlatBinary", "NewFlatBinaryOverquery", "DefaultOverquery"},
	"embed":   {"LoadMmap"},
}

// experimentalMembers: Experimental MEMBERS on Hard-tier types — matched as whole apidiff
// symbol paths so listing one member never exempts the rest of its type.
var experimentalMembers = map[string][]string{
	"embed": {"(*SafetensorsFile).ReleaseTensors", "Tensor.SubF32"},
}

const apidiffPkg = "golang.org/x/exp/cmd/apidiff@latest"

func main() { os.Exit(run(os.Args[1:])) }

func run(args []string) int {
	if len(args) < 1 || strings.TrimSpace(args[0]) == "" {
		fmt.Fprintln(os.Stderr, "usage: releasegate <version, e.g. 1.4.0 (no leading v)>")
		return 2
	}
	ver := args[0]
	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Println(gate.Verdict(gate.Inconclusive, "cannot locate repo root: "+err.Error()))
		return 2
	}

	checks := []gate.Check{
		{Name: "changelog", Run: func() gate.Cell { return checkChangelog(root, ver) }},
		{Name: "golangci-lint", Run: func() gate.Cell { return checkLint(root) }},
		{Name: "apidiff", Run: func() gate.Cell { return checkAPIDiff(root, ver) }},
		{Name: "core-deps", Run: func() gate.Cell { return checkCoreDeps(root) }},
		{Name: "perf-evidence", Run: func() gate.Cell { return checkPerfEvidence(root, ver) }},
	}
	cells := gate.RunAll(checks)
	rep := gate.ReconcileWith(cells, gate.FailWins)

	switch rep.Outcome {
	case gate.Fail:
		fmt.Println(gate.Verdict(gate.Fail, fmt.Sprintf("v%s — %d/%d check(s) failed", ver, rep.Fail, rep.Total)))
		return 1
	case gate.Inconclusive:
		fmt.Println(gate.Verdict(gate.Inconclusive, fmt.Sprintf("v%s — a required external tool was unavailable", ver)))
		return 2
	}
	fmt.Println(gate.Verdict(gate.OK, fmt.Sprintf("v%s — %d/%d checks passed", ver, rep.Pass, rep.Total)))
	return 0
}

// (1) CHANGELOG — the messages name exactly what is missing (a release deliverable).
func checkChangelog(root, ver string) gate.Cell {
	data, err := os.ReadFile(filepath.Join(root, "CHANGELOG.md"))
	if err != nil {
		return failMsg("changelog", "cannot read CHANGELOG.md: "+err.Error())
	}
	s := string(data)
	q := regexp.QuoteMeta(ver)
	var missing []string
	if !regexp.MustCompile(`(?m)^## \[` + q + `\]`).MatchString(s) {
		missing = append(missing, fmt.Sprintf("CHANGELOG.md has no '## [%s]' section", ver))
	}
	if !regexp.MustCompile(`(?m)^\[` + q + `\]: `).MatchString(s) {
		missing = append(missing, fmt.Sprintf("CHANGELOG.md has no '[%s]:' compare link", ver))
	}
	if len(missing) > 0 {
		for _, m := range missing {
			fmt.Println("::error::release-gate: " + m)
		}
		return failMsg("changelog", strings.Join(missing, "; "))
	}
	fmt.Println("release-gate: CHANGELOG section + compare link present")
	return okCell("changelog")
}

// (2) golangci-lint — pinned build, on the whole root module.
func checkLint(root string) gate.Cell {
	fmt.Println("release-gate: golangci-lint (.golangci.yml)")
	if _, rc := gpumod.Exec(root, "", nil, "go", "run", gpumod.GolangciLint, "version"); rc != 0 {
		return inconMsg("golangci-lint", "could not build the pinned golangci-lint")
	}
	// CANARY: trust a "clean" only after the linter flags its fixture (see tools/canary).
	cout, _ := gpumod.Exec(filepath.Join(root, canary.FixturesDir), "", nil, "go", "run", gpumod.GolangciLint, "run", "--build-tags", "canaryfixture")
	if res := canary.CheckGolangci(cout); !res.Fired {
		return inconMsg("golangci-lint", "CANNOT-EVALUATE — "+res.Reason)
	}
	out, rc := gpumod.Exec(root, "", nil, "go", "run", gpumod.GolangciLint, "run", "./...")
	if rc != 0 {
		fmt.Println("::error::release-gate: golangci-lint reported issues")
		fmt.Println(strings.TrimRight(out, "\n"))
		return failMsg("golangci-lint", "issues reported")
	}
	return okCell("golangci-lint")
}

// (3) apidiff — Hard-tier compatibility vs the previous tag.
func checkAPIDiff(root, ver string) gate.Cell {
	// CANARY: prove apidiff actually detects a break here, before any "no incompatible
	// changes" — including the no-previous-tag skip — is trusted. A baseline comparison with
	// nothing to compare against reports no breaks, indistinguishable from a real pass; the
	// fixture pair has a known break apidiff must find, or this is cannot-evaluate.
	if res := canary.CheckApidiff(apidiffCanaryOut(root)); !res.Fired {
		return inconMsg("apidiff", "CANNOT-EVALUATE — "+res.Reason)
	}
	prev := prevTag(root, ver)
	if prev == "" {
		fmt.Println("release-gate: no previous tag — skipping apidiff")
		return okCell("apidiff")
	}
	fmt.Printf("release-gate: apidiff Hard tier %s → current tree\n", prev)
	wt, err := os.MkdirTemp("", "releasegate-wt")
	if err != nil {
		return inconMsg("apidiff", "mktemp: "+err.Error())
	}
	if _, rc := gpumod.Exec(root, "", nil, "git", "worktree", "add", "-q", "-f", wt, prev); rc != 0 {
		return inconMsg("apidiff", "git worktree add "+prev+" failed")
	}
	defer func() {
		gpumod.Exec(root, "", nil, "git", "worktree", "remove", "--force", wt)
		gpumod.Exec(root, "", nil, "git", "worktree", "prune")
	}()

	var broken []string
	for _, p := range hardPkgs {
		base := filepath.Join(os.TempDir(), "gate-"+p+".api")
		// apidiff exits non-zero only on a TOOL/LOAD error (it cannot build itself, or
		// cannot load the package) — reporting API changes is exit 0. So a non-zero here is
		// "could not judge", INCONCLUSIVE, never mistaken for a diff. -w writes the previous
		// tag's API from the worktree; -incompatible compares the current tree against it.
		if base_out, rc := gpumod.Exec(wt, "", nil, "go", "run", apidiffPkg, "-w", base, rootMod+"/"+p); rc != 0 {
			return inconMsg("apidiff", "apidiff could not record the "+prev+" baseline for "+p+": "+firstLine(base_out))
		}
		inc, rc := gpumod.Exec(root, "", nil, "go", "run", apidiffPkg, "-incompatible", base, rootMod+"/"+p)
		if rc != 0 {
			return inconMsg("apidiff", "apidiff could not compare "+p+" against "+prev+": "+firstLine(inc))
		}
		inc = filterExperimental(inc, p)
		if strings.TrimSpace(inc) != "" {
			fmt.Printf("::error::release-gate: apidiff: incompatible change in Hard-tier '%s' vs %s:\n%s\n", p, prev, strings.TrimRight(inc, "\n"))
			broken = append(broken, p)
		}
	}
	if len(broken) > 0 {
		return failMsg("apidiff", "Hard-tier breakage vs "+prev+" in: "+strings.Join(broken, " "))
	}
	return okCell("apidiff")
}

// (4) core dependency invariant.
func checkCoreDeps(root string) gate.Cell {
	out, rc := gpumod.Exec(root, "", nil, "go", "list", "-deps", "-f", "{{if not .Standard}}{{.ImportPath}}{{end}}", "./...")
	// A failed `go list` emits little or nothing, which would filter to "no external deps" and
	// read as clean — the same examines-nothing false-clean the canaries guard against. rc is
	// the built-in positive control here: a non-zero go list is cannot-evaluate, not a pass.
	if rc != 0 {
		return inconMsg("core-deps", "CANNOT-EVALUATE — `go list -deps` failed: "+firstLine(out))
	}
	var ext []string
	allowed := regexp.MustCompile(`^golang\.org/x/(text|sys)(/|$)`)
	for ln := range strings.SplitSeq(out, "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" || strings.HasPrefix(ln, rootMod) || allowed.MatchString(ln) {
			continue
		}
		ext = append(ext, ln)
	}
	if len(ext) > 0 {
		fmt.Println("::error::release-gate: core module pulls unexpected external dependency(ies): " + strings.Join(ext, " "))
		return failMsg("core-deps", "unexpected external deps: "+strings.Join(ext, " "))
	}
	return okCell("core-deps")
}

// apidiffCanaryOut writes the old fixture's API and compares the new fixture against it, using
// the SAME `go run apidiff` invocation as the real check, from the tools module dir where the
// fixture import paths resolve. Its output is handed to canary.CheckApidiff.
func apidiffCanaryOut(root string) string {
	toolsDir := filepath.Join(root, "tools")
	base := filepath.Join(os.TempDir(), "canary-apidiff-old.api")
	if _, rc := gpumod.Exec(toolsDir, "", nil, "go", "run", apidiffPkg, "-w", base, canary.ApidiffFixtureOld); rc != 0 {
		return "" // could not even record the baseline → not fired → cannot-evaluate
	}
	out, _ := gpumod.Exec(toolsDir, "", nil, "go", "run", apidiffPkg, "-incompatible", base, canary.ApidiffFixtureNew)
	return out
}

// prevTag is the highest v* tag by version, excluding the one being released — so a retracted
// old tag (v1.8.0) is never selected for a current-era release.
func prevTag(root, ver string) string {
	out, _ := gpumod.Exec(root, "", nil, "git", "tag", "--list", "v*", "--sort=-v:refname")
	for ln := range strings.SplitSeq(out, "\n") {
		t := strings.TrimSpace(ln)
		if t != "" && t != "v"+ver {
			return t
		}
	}
	return ""
}

// filterExperimental drops apidiff incompatible lines whose subject is documented
// Experimental for this package — the leading-symbol cases and the whole-member cases.
func filterExperimental(inc, pkg string) string {
	lines := strings.Split(inc, "\n")
	syms := experimentalSyms[pkg]
	var symRe *regexp.Regexp
	if len(syms) > 0 {
		quoted := make([]string, len(syms))
		for i, s := range syms {
			quoted[i] = regexp.QuoteMeta(s)
		}
		symRe = regexp.MustCompile(`^- \(?\*?(` + strings.Join(quoted, "|") + `)[).: ]`)
	}
	members := experimentalMembers[pkg]
	var kept []string
	for _, ln := range lines {
		if symRe != nil && symRe.MatchString(ln) {
			continue
		}
		drop := false
		for _, m := range members {
			if strings.HasPrefix(ln, "- "+m+":") {
				drop = true
				break
			}
		}
		if !drop {
			kept = append(kept, ln)
		}
	}
	return strings.Join(kept, "\n")
}

// runIn runs a command in dir with the ambient environment (release-gate, like its shell,
// does not override GOWORK — on the Linux/CI target there is no go.work). Returns combined
// output and exit code.

func okCell(name string) gate.Cell { return gate.Cell{Name: name, Outcome: gate.OK} }
func failMsg(name, msg string) gate.Cell {
	return gate.Cell{Name: name, Outcome: gate.Fail, Fields: []gate.Field{{Key: "msg", State: msg}}}
}
func inconMsg(name, msg string) gate.Cell {
	fmt.Println("release-gate: INCONCLUSIVE — " + name + ": " + msg)
	return gate.Cell{Name: name, Outcome: gate.Inconclusive, Fields: []gate.Field{{Key: "msg", State: msg}}}
}

func firstLine(s string) string {
	for ln := range strings.SplitSeq(s, "\n") {
		if strings.TrimSpace(ln) != "" {
			return strings.TrimSpace(ln)
		}
	}
	return ""
}

// (5) PERF EVIDENCE — the release's CHANGELOG section must carry a perfgate
// VERDICT line, or an explicit EXCEPTION saying why it does not (audit G-04).
//
// v1.38.0 — the release that changed BOTH ViT towers' attention schedule —
// shipped with no perfgate, no vulncheck and no recorded exception, and this
// gate passed it, because it only checked that a CHANGELOG section and a
// compare link existed. The "exception recorded in the open" convention
// introduced in v1.35.0 decayed to silence over three releases, which is what
// an unenforced convention does.
//
// perfgateVerdictLine matches the real perfgate result line the CHANGELOG
// convention uses — e.g. "`perfgate` VERDICT: PASS — no regression vs
// v1.40.0 above each shape's floor — 33/45 shapes resolve the 5.0% class"
// (every real entry to date: v1.38.0, v1.39.0, v1.39.1, v1.40.0 all use this
// exact prefix). Anchored to the START of a line so prose that merely
// mentions the words "perfgate" and "verdict" elsewhere in the section
// cannot satisfy it. This is the fix for a real gate-that-cannot-fail bug
// found 2026-09-11: the release-prep author wrote a "measurement pending
// nvidia-rtx2070s" sentence for a still-unmeasured perfgate run that itself
// happened to contain both substrings ("no perfgate VERDICT is recorded
// here... an explicit EXCEPTION"), and the old strings.Contains check —
// "perfgate" anywhere AND ("VERDICT" or "EXCEPTION") anywhere, independently
// — read that as a real result and passed. A prose sentence EXPLAINING that
// no verdict exists yet must not be indistinguishable from one.
var perfgateVerdictLine = regexp.MustCompile("(?m)^`perfgate` VERDICT: (PASS|FAIL|INCONCLUSIVE)\\b")

// perfgateExceptionLine matches the documented-exception form (the PERFGATE
// EXCEPTION blockquote v1.40.0 introduced for a known false-positive VERDICT
// FAIL) — RELEASING.md step 2b's "or record an explicit EXCEPTION". Same
// line-start anchoring, same reasoning.
var perfgateExceptionLine = regexp.MustCompile(`(?m)^>\s*\*\*PERFGATE EXCEPTION:`)

// The gpu module's tag-evidence workflow already requires a VERDICT: line in
// the tag message; this is the root module's equivalent, checked against the
// CHANGELOG section where this repo actually records its gates. Deliberately
// NOT a check that perfgate passed — that is perfgate's job and its exit code.
// This checks only that the question was asked and answered in the open, AT
// the structural position RELEASING.md defines (its own line, the literal
// `perfgate` VERDICT: token or the PERFGATE EXCEPTION blockquote) — not
// merely somewhere in the section's prose. See perfgateVerdictLine's own
// comment for why that distinction is load-bearing, not pedantry.
func checkPerfEvidence(root, ver string) gate.Cell {
	data, err := os.ReadFile(filepath.Join(root, "CHANGELOG.md"))
	if err != nil {
		return failMsg("perf-evidence", "cannot read CHANGELOG.md: "+err.Error())
	}
	sec, ok := changelogSection(string(data), ver)
	if !ok {
		// checkChangelog already fails on this; do not double-report.
		return gate.Cell{Name: "perf-evidence", Outcome: gate.OK,
			Fields: []gate.Field{{Key: "section", State: "absent (reported by changelog check)"}}}
	}
	hasVerdict := perfgateVerdictLine.MatchString(sec) || perfgateExceptionLine.MatchString(sec)
	if !hasVerdict {
		return failMsg("perf-evidence", fmt.Sprintf(
			"CHANGELOG [%s] has no perfgate VERDICT line (its own line, exactly "+
				"\"`perfgate` VERDICT: PASS|FAIL|INCONCLUSIVE — ...\") or PERFGATE EXCEPTION blockquote — "+
				"run `go run -C tools ./perfgate <prev-tag>` and paste its VERDICT line verbatim, or record "+
				"an explicit EXCEPTION saying why this release does not need one", ver))
	}
	fmt.Println("release-gate: perfgate VERDICT/EXCEPTION recorded for v" + ver)
	return gate.Cell{Name: "perf-evidence", Outcome: gate.OK,
		Fields: []gate.Field{{Key: "perfgate", State: "recorded"}}}
}

// changelogSection returns the body of the `## [ver]` section, up to the next
// `## [` heading.
func changelogSection(doc, ver string) (string, bool) {
	head := "## [" + ver + "]"
	i := strings.Index(doc, head)
	if i < 0 {
		return "", false
	}
	rest := doc[i+len(head):]
	if j := strings.Index(rest, "\n## ["); j >= 0 {
		rest = rest[:j]
	}
	return rest, true
}
