package main

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/tools/gpumod"
)

// coreSteps is what preflight KNOWS ci.yml's core (root-module) job runs, in order. This
// list is the TRIPWIRE: tools/ is stdlib-only so preflight cannot parse the YAML and derive
// the mirror, but it can read the step names as text and fail when they change — so the next
// check CI's core job gains cannot silently go un-mirrored. Update this list AND
// buildChecks/implementedAs/notImplemented together, on purpose — TestCoreStepsAccountedFor,
// below, enforces that every name here lands in exactly one of the other two.
var coreSteps = []string{
	"gofmt",
	"build",
	"vet",
	"golangci-lint",
	"no cgo deps in core graph",
	"build with cgo disabled",
	"test (race)",
	"test (aikit_checks)",
}

// implementedAs maps each ci.yml core-job step name preflight mirrors to the Check.Name
// buildChecks runs it under — identical except "test (race)", which preflight runs UNRACED
// (too slow for a pre-push gate; see main.go's doc comment) under the name "go test".
var implementedAs = map[string]string{
	"gofmt":                     "gofmt",
	"build":                     "build",
	"vet":                       "vet",
	"golangci-lint":             "golangci-lint",
	"no cgo deps in core graph": "no cgo deps in core graph",
	"build with cgo disabled":   "build (cgo-free)",
	"test (race)":               "go test",
	"test (aikit_checks)":       "test (aikit_checks)",
}

// notImplemented is every ci.yml core-job step name preflight deliberately does not run at
// all, with why. This is the enforcement point for the finding that prompted it: a prose
// exclusion policy in main.go's doc comment is not checked against reality, so a step could
// go unmirrored with no reason that actually fit the policy (as "no cgo deps in core graph"
// and "test (aikit_checks)" both did, despite needing neither network nor a GPU). A name in
// neither this map nor implementedAs now fails TestCoreStepsAccountedFor instead of just
// being absent from a comment nobody re-checks.
//
// Empty since `fuzz (smoke)` moved out of the core job into its own ci.yml job (it ran as
// core's last step and sat on CI's critical path); like the other separate jobs it is not
// attempted here — too slow for a pre-push gate.
var notImplemented = map[string]string{}

func TestCICoreStepsUnchanged(t *testing.T) {
	root, err := gpumod.RepoRoot()
	if err != nil {
		t.Fatalf("repo root: %v", err)
	}
	got, err := coreJobStepNames(filepath.Join(root, ".github", "workflows", "ci.yml"))
	if err != nil {
		t.Fatalf("reading ci.yml core steps: %v", err)
	}
	if strings.Join(got, "|") != strings.Join(coreSteps, "|") {
		t.Fatalf("ci.yml core-job steps changed — the preflight mirror may be stale.\n"+
			"  got:  %v\n  want: %v\n"+
			"Reconcile: update coreSteps in this file AND decide whether tools/preflight should now run the new/changed check.",
			got, coreSteps)
	}
}

// TestCoreStepsAccountedFor is the check TestCICoreStepsUnchanged does NOT do: it verifies
// every ci.yml core-job step name is either actually implemented by buildChecks (via
// implementedAs) or explicitly, individually excluded (via notImplemented) — never both,
// never neither. buildChecks itself does no I/O (a Check's Run closure only executes inside
// gate.RunAll), so this runs fast without a real build/lint/test pass.
func TestCoreStepsAccountedFor(t *testing.T) {
	root, err := gpumod.RepoRoot()
	if err != nil {
		t.Fatalf("repo root: %v", err)
	}
	implemented := map[string]bool{}
	for _, c := range buildChecks(root, false) {
		implemented[c.Name] = true
	}
	for _, name := range coreSteps {
		wantName, isImplemented := implementedAs[name]
		_, isExcluded := notImplemented[name]
		switch {
		case isImplemented && isExcluded:
			t.Errorf("ci.yml core step %q is listed in BOTH implementedAs and notImplemented — fix the bookkeeping", name)
		case !isImplemented && !isExcluded:
			t.Errorf("ci.yml core step %q is in neither implementedAs nor notImplemented — a silent gap "+
				"(either implement it in buildChecks or add a reason to notImplemented)", name)
		case isImplemented && !implemented[wantName]:
			t.Errorf("ci.yml core step %q claims to run as preflight check %q (implementedAs), "+
				"but buildChecks has no such check", name, wantName)
		}
	}
}

// crossVetChecks are the two checks buildChecks runs that are NOT core-job mirrors: a
// cross-GOOS/GOARCH `go vet ./...` front-running ci.yml's separate `windows` job (and the
// amd64 legs of `simd`/`perf-smoke`). See main.go's CROSS-VET paragraph for why an "another
// OS" exclusion covers that job's build/test steps but not its vet step.
//
// They are asserted here rather than left to buildChecks alone because the two enforcement
// tests above only iterate coreSteps: a check with no ci.yml core step behind it is invisible
// to them, so it could be dropped in a refactor with nothing going red. That is precisely the
// failure mode this package's own doc comment warns about ("a prose exclusion policy is not
// checked against reality"), applied to an addition instead of an omission.
var crossVetChecks = []string{"cross-vet windows", "cross-vet linux"}

// TestCrossVetChecksPresent pins the two cross-compile vet checks into buildChecks, and
// re-verifies the premise that justifies them: ci.yml's `windows` job really does run
// `go vet ./...`, so front-running it locally is mirroring a real check rather than
// inventing one. If CI's windows job stops vetting, this fails and the justification in
// main.go gets re-read instead of quietly becoming false.
func TestCrossVetChecksPresent(t *testing.T) {
	root, err := gpumod.RepoRoot()
	if err != nil {
		t.Fatalf("repo root: %v", err)
	}
	have := map[string]bool{}
	for _, c := range buildChecks(root, false) {
		have[c.Name] = true
	}
	for _, name := range crossVetChecks {
		if !have[name] {
			t.Errorf("buildChecks is missing %q — the cross-GOOS/GOARCH vet that catches "+
				"missing build tags before a push (see main.go's CROSS-VET paragraph; 9724289 "+
				"is the round trip it exists to prevent)", name)
		}
	}

	steps, err := jobRunSteps(filepath.Join(root, ".github", "workflows", "ci.yml"), "windows")
	if err != nil {
		t.Fatalf("reading ci.yml windows job: %v", err)
	}
	if !slices.Contains(steps, "go vet ./...") {
		t.Errorf("ci.yml's `windows` job no longer runs `go vet ./...` (got %v) — the premise for "+
			"preflight's cross-vet checks changed; re-read main.go's CROSS-VET paragraph and "+
			"decide whether the local mirror still makes sense", steps)
	}
}

// jobRunSteps reads ci.yml as TEXT and returns the `- run:` commands inside the named job,
// in order, stopping at the next 2-space-indented job header — the `- run:` counterpart of
// coreJobStepNames, for jobs whose steps are bare commands rather than named blocks.
func jobRunSteps(ymlPath, job string) ([]string, error) {
	b, err := os.ReadFile(ymlPath)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(b), "\n")
	start := -1
	for i, ln := range lines {
		if strings.TrimRight(ln, " ") == "  "+job+":" {
			start = i
			break
		}
	}
	if start < 0 {
		return nil, fmt.Errorf("no `  %s:` job found in %s", job, ymlPath)
	}
	var runs []string
	for _, ln := range lines[start+1:] {
		if len(ln) > 2 && ln[0] == ' ' && ln[1] == ' ' && isJobNameStart(ln[2]) {
			break
		}
		if t := strings.TrimSpace(ln); strings.HasPrefix(t, "- run:") {
			runs = append(runs, strings.TrimSpace(t[len("- run:"):]))
		}
	}
	return runs, nil
}

// coreJobStepNames reads ci.yml as TEXT and returns the `- name:` step names inside the
// `core:` job, in order, stopping at the next 2-space-indented job header.
func coreJobStepNames(ymlPath string) ([]string, error) {
	b, err := os.ReadFile(ymlPath)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(b), "\n")
	start := -1
	for i, ln := range lines {
		if strings.TrimRight(ln, " ") == "  core:" {
			start = i
			break
		}
	}
	if start < 0 {
		return nil, fmt.Errorf("no `  core:` job found in %s", ymlPath)
	}
	var names []string
	for _, ln := range lines[start+1:] {
		if len(ln) > 2 && ln[0] == ' ' && ln[1] == ' ' && isJobNameStart(ln[2]) {
			break // a sibling job header at 2-space indent — end of the core block
		}
		t := strings.TrimSpace(ln)
		if strings.HasPrefix(t, "- name:") {
			names = append(names, strings.TrimSpace(t[len("- name:"):]))
		}
	}
	return names, nil
}

func isJobNameStart(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}
