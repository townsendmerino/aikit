package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/tools/gate"
)

// a realistic backend go.mod: a require block with both aikit series, an indirect block, and
// the two replace directives that must NOT be mistaken for requires.
const sampleGomod = `module github.com/townsendmerino/aikit/gpu/annmetal

go 1.26.6

require (
	github.com/townsendmerino/aikit v1.17.0
	github.com/townsendmerino/aikit/gpu v0.28.0
)

require (
	github.com/ebitengine/purego v0.10.1 // indirect
	golang.org/x/sys v0.47.0 // indirect
)

replace github.com/townsendmerino/aikit => ../../

replace github.com/townsendmerino/aikit/gpu => ../
`

func TestRequires_readsBothSeriesNotReplaces(t *testing.T) {
	root, gpu := requires(sampleGomod)
	if root != "v1.17.0" {
		t.Errorf("root require = %q, want v1.17.0 (a replace line must not be read as the require)", root)
	}
	if gpu != "v0.28.0" {
		t.Errorf("gpu require = %q, want v0.28.0", gpu)
	}
}

func TestPinField_classifies(t *testing.T) {
	cases := []struct {
		got, want, state string
	}{
		{"v1.17.1", "v1.17.1", "ok"},
		{"v1.17.0", "v1.17.1", "DRIFT"},
		{"", "v1.17.1", "MISSING"},
	}
	for _, c := range cases {
		if f := pinField("root", c.got, c.want); f.State != c.state {
			t.Errorf("pinField(got=%q,want=%q).State = %q, want %q", c.got, c.want, f.State, c.state)
		}
	}
}

func TestRewrite_bumpsRootOnlyAndReports(t *testing.T) {
	// gpu already current, root stale — only the root line should change, and the edit
	// list should name exactly that transition.
	out, edits := rewrite(sampleGomod, "v1.17.1", "v0.28.0")
	gotRoot, gotGpu := requires(out)
	if gotRoot != "v1.17.1" {
		t.Errorf("after rewrite root = %q, want v1.17.1", gotRoot)
	}
	if gotGpu != "v0.28.0" {
		t.Errorf("after rewrite gpu = %q, want v0.28.0 (unchanged)", gotGpu)
	}
	if len(edits) != 1 || edits[0] != "root v1.17.0→v1.17.1" {
		t.Errorf("edits = %v, want exactly [root v1.17.0→v1.17.1]", edits)
	}
	// the replace directives must survive untouched.
	if want := "replace github.com/townsendmerino/aikit => ../../"; !contains(out, want) {
		t.Errorf("rewrite dropped a replace directive; output:\n%s", out)
	}
}

func TestRewrite_noopWhenCurrent(t *testing.T) {
	out, edits := rewrite(sampleGomod, "v1.17.0", "v0.28.0")
	if len(edits) != 0 {
		t.Errorf("edits = %v, want none when already current", edits)
	}
	if out != sampleGomod {
		t.Errorf("rewrite changed bytes on a no-op:\n%s", out)
	}
}

func TestRewrite_bumpsBoth(t *testing.T) {
	out, edits := rewrite(sampleGomod, "v1.18.0", "v0.29.0")
	gotRoot, gotGpu := requires(out)
	if gotRoot != "v1.18.0" || gotGpu != "v0.29.0" {
		t.Errorf("after rewrite root=%q gpu=%q, want v1.18.0 / v0.29.0", gotRoot, gotGpu)
	}
	if len(edits) != 2 {
		t.Errorf("edits = %v, want two (root and gpu)", edits)
	}
}

func contains(hay, needle string) bool {
	for i := 0; i+len(needle) <= len(hay); i++ {
		if hay[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}

func TestRewrite_emptyRootTargetLeavesTheRootRequire(t *testing.T) {
	// AX5: the backends no longer track the newest root tag, so --fix passes no root target.
	out, edits := rewrite(sampleGomod, "", "v0.35.1")
	gotRoot, gotGpu := requires(out)
	if gotRoot != "v1.17.0" || gotGpu != "v0.35.1" {
		t.Errorf("after rewrite root=%q gpu=%q, want v1.17.0 (untouched) and v0.35.1", gotRoot, gotGpu)
	}
	if len(edits) != 1 {
		t.Errorf("edits = %v, want exactly the gpu bump", edits)
	}
}

func TestBackends_areTheFourThatStay(t *testing.T) {
	want := []string{"gpu/anncuda", "gpu/annmetal", "gpu/enccuda", "gpu/encmetal"}
	if len(backends) != len(want) {
		t.Fatalf("backends = %v, want %v", backends, want)
	}
	for i := range want {
		if backends[i] != want[i] {
			t.Fatalf("backends = %v, want %v", backends, want)
		}
	}
	if !reGpuTag.MatchString("gpu/v0.35.0") || reGpuTag.MatchString("gpu/v0.36.0") || reGpuTag.MatchString("gpu/anncuda/v0.35.0") {
		t.Error("reGpuTag must match exactly the gpu/v0.35.x series")
	}
}

// writeBackend writes a backend go.mod requiring rootV and gpuV under dir/mod.
func writeBackend(t *testing.T, dir, mod, rootV, gpuV string) {
	t.Helper()
	gm := "module github.com/townsendmerino/aikit/" + mod + "\n\ngo 1.27.0\n\nrequire (\n" +
		"\tgithub.com/townsendmerino/aikit " + rootV + "\n\tgithub.com/townsendmerino/aikit/gpu " + gpuV + "\n)\n\n" +
		"replace github.com/townsendmerino/aikit => ../../\n\nreplace github.com/townsendmerino/aikit/gpu => ../\n"
	if err := os.MkdirAll(filepath.Join(dir, mod), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, mod, "go.mod"), []byte(gm), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestCheckModule_theFreezeHolds is SD2's check: a backend that bumps its root require off the freeze, or moves its gpu
// require off the v0.35 series, FAILs, and the frozen pair passes.
func TestCheckModule_theFreezeHolds(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		name, root, gpu string
		ok              bool
	}{
		{"frozen", frozenRoot, "v0.35.0", true},
		{"root bumped", "v1.76.0", "v0.35.0", false},
		{"root pseudo-version", "v1.76.1-0.20261027120000-abcdefabcdef", "v0.35.0", false},
		{"gpu off the series", frozenRoot, "v0.36.0", false},
	}
	for _, c := range cases {
		writeBackend(t, dir, "gpu/anncuda", c.root, c.gpu)
		cell := checkModule(dir, "gpu/anncuda", frozenRoot, "v0.35.0")
		if got := cell.Outcome == gate.OK; got != c.ok {
			t.Errorf("%s (root %s, gpu %s): outcome %v, want ok=%v", c.name, c.root, c.gpu, cell.Outcome, c.ok)
		}
		if moved := rootMoved([]gate.Cell{cell}); moved != (c.root != frozenRoot) {
			t.Errorf("%s: rootMoved = %v", c.name, moved)
		}
	}
}

// TestFix_neverMovesTheFrozenRoot: --fix passes no root target, so a stale gpu require is rewritten and the root
// require, frozen, is left as it is.
func TestFix_neverMovesTheFrozenRoot(t *testing.T) {
	gm := "require (\n\tgithub.com/townsendmerino/aikit " + frozenRoot + "\n\tgithub.com/townsendmerino/aikit/gpu v0.35.0\n)\n"
	out, edits := rewrite(gm, "", "v0.35.1")
	if r, g := requires(out); r != frozenRoot || g != "v0.35.1" || len(edits) != 1 {
		t.Errorf("after --fix's rewrite root=%q gpu=%q edits=%v, want %s untouched and gpu v0.35.1", r, g, edits, frozenRoot)
	}
}
