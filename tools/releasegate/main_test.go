package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/tools/gate"
)

func TestFirstLine(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"single line", "hello", "hello"},
		{"leading blank lines skipped", "\n\n  \nhello\nworld", "hello"},
		{"trims whitespace", "  hello  \nworld", "hello"},
		{"all blank", "\n \n\t\n", ""},
		{"empty", "", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := firstLine(c.in); got != c.want {
				t.Errorf("firstLine(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

func TestOkFailInconCell(t *testing.T) {
	if c := okCell("x"); c.Outcome != gate.OK || c.Name != "x" {
		t.Errorf("okCell = %+v, want Name=x Outcome=OK", c)
	}
	if c := failMsg("x", "boom"); c.Outcome != gate.Fail || c.Field("msg") != "boom" {
		t.Errorf("failMsg = %+v, want Outcome=Fail msg=boom", c)
	}
	if c := inconMsg("x", "boom"); c.Outcome != gate.Inconclusive || c.Field("msg") != "boom" {
		t.Errorf("inconMsg = %+v, want Outcome=Inconclusive msg=boom", c)
	}
}

// TestFilterExperimental_dropsListedSymbol confirms a top-level Experimental symbol's
// apidiff line is dropped, using experimentalSyms's real "ann" entry — a real Hard-tier
// package this gate protects, not a synthetic stand-in.
func TestFilterExperimental_dropsListedSymbol(t *testing.T) {
	in := strings.Join([]string{
		"- FlatBinary: removed",
		"- (*FlatBinary).Query: changed from func([]float32, int) []Hit to func([]float32, int, int) []Hit",
		"- Flat: removed", // NOT experimental for ann — must survive
	}, "\n")
	got := filterExperimental(in, "ann")
	if strings.Contains(got, "FlatBinary") {
		t.Errorf("filterExperimental kept an experimental symbol's line:\n%s", got)
	}
	if !strings.Contains(got, "- Flat: removed") {
		t.Errorf("filterExperimental dropped a non-experimental (Hard-tier) line it must keep:\n%s", got)
	}
}

// TestFilterExperimental_dropsListedMember confirms an Experimental MEMBER on an otherwise
// Hard-tier type is dropped without exempting the type's other members — the exact
// "listing one member never exempts the rest of its type" property the doc comment states,
// using experimentalMembers's real "embed" entry.
func TestFilterExperimental_dropsListedMember(t *testing.T) {
	in := strings.Join([]string{
		"- (*SafetensorsFile).ReleaseTensors: removed",
		"- (*SafetensorsFile).Close: removed", // a DIFFERENT method on the same Hard-tier type — must survive
	}, "\n")
	got := filterExperimental(in, "embed")
	if strings.Contains(got, "ReleaseTensors") {
		t.Errorf("filterExperimental kept an experimental member's line:\n%s", got)
	}
	if !strings.Contains(got, "(*SafetensorsFile).Close") {
		t.Errorf("filterExperimental dropped a sibling Hard-tier member it must keep — "+
			"one experimental member must not exempt the whole type:\n%s", got)
	}
}

// TestFilterExperimental_noExemptionsForPkg confirms a package with no entry in either map
// (e.g. "topk") passes every line through unchanged — the filter must not accidentally
// exempt a package nobody declared experimental symbols for.
func TestFilterExperimental_noExemptionsForPkg(t *testing.T) {
	in := "- Selector: removed\n- New: changed"
	got := filterExperimental(in, "topk")
	if got != in {
		t.Errorf("filterExperimental(unlisted pkg) = %q, want unchanged %q", got, in)
	}
}

// TestFilterExperimental_symbolMatchIsAnchored guards the regex's own precision: a line
// whose apidiff symbol merely CONTAINS an experimental name as a substring (e.g.
// "FlatBinaryHelper", not "FlatBinary" itself) must NOT be dropped — the anchor (^- \(?\*?
// (sym)[).: ]) requires the symbol name end at one of ), ., :, or a space.
func TestFilterExperimental_symbolMatchIsAnchored(t *testing.T) {
	in := "- FlatBinaryHelper: removed"
	got := filterExperimental(in, "ann")
	if got != in {
		t.Errorf("filterExperimental dropped a line whose symbol only shares a PREFIX with an experimental name: %q", got)
	}
}

func writeChangelog(t *testing.T, root, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(root, "CHANGELOG.md"), []byte(body), 0o644); err != nil {
		t.Fatalf("write CHANGELOG.md: %v", err)
	}
}

func TestCheckChangelog_ok(t *testing.T) {
	root := t.TempDir()
	writeChangelog(t, root, "# Changelog\n\n## [1.4.0]\n\nsome notes\n\n[1.4.0]: https://example.com/compare/v1.3.0...v1.4.0\n")
	c := checkChangelog(root, "1.4.0")
	if c.Outcome != gate.OK {
		t.Errorf("Outcome = %v, want OK; msg=%q", c.Outcome, c.Field("msg"))
	}
}

func TestCheckChangelog_missingSection(t *testing.T) {
	root := t.TempDir()
	writeChangelog(t, root, "# Changelog\n\n[1.4.0]: https://example.com/compare/v1.3.0...v1.4.0\n")
	c := checkChangelog(root, "1.4.0")
	if c.Outcome != gate.Fail {
		t.Fatalf("Outcome = %v, want Fail", c.Outcome)
	}
	if !strings.Contains(c.Field("msg"), "no '## [1.4.0]' section") {
		t.Errorf("msg = %q, want it to name the missing section", c.Field("msg"))
	}
}

func TestCheckChangelog_missingCompareLink(t *testing.T) {
	root := t.TempDir()
	writeChangelog(t, root, "# Changelog\n\n## [1.4.0]\n\nsome notes\n")
	c := checkChangelog(root, "1.4.0")
	if c.Outcome != gate.Fail {
		t.Fatalf("Outcome = %v, want Fail", c.Outcome)
	}
	if !strings.Contains(c.Field("msg"), "no '[1.4.0]:' compare link") {
		t.Errorf("msg = %q, want it to name the missing compare link", c.Field("msg"))
	}
}

func TestCheckChangelog_missingFile(t *testing.T) {
	root := t.TempDir() // no CHANGELOG.md at all
	c := checkChangelog(root, "1.4.0")
	if c.Outcome != gate.Fail {
		t.Errorf("Outcome = %v, want Fail", c.Outcome)
	}
	if !strings.Contains(c.Field("msg"), "cannot read CHANGELOG.md") {
		t.Errorf("msg = %q, want it to name the read failure", c.Field("msg"))
	}
}

// TestCheckPerfEvidence_proseSayingVerdictFails is the regression pin for the
// gate-that-cannot-fail bug found 2026-09-11 (see checkPerfEvidence's own
// comment): a section containing the WORDS "perfgate" and "verdict"/
// "exception" only in ordinary prose — explicitly saying no measurement has
// been taken yet — must still FAIL, because there is no real
// "`perfgate` VERDICT: ..." line or "PERFGATE EXCEPTION" blockquote at the
// structural position RELEASING.md defines. The old strings.Contains check
// passed this exact shape of text.
func TestCheckPerfEvidence_proseSayingVerdictFails(t *testing.T) {
	root := t.TempDir()
	writeChangelog(t, root, "# Changelog\n\n## [1.4.0]\n\n"+
		"Measurement pending nvidia-rtx2070s: no perfgate verdict is recorded here on purpose, "+
		"and there is no explicit exception either — nobara was unreachable this session.\n\n"+
		"[1.4.0]: https://example.com/compare/v1.3.0...v1.4.0\n")
	c := checkPerfEvidence(root, "1.4.0")
	if c.Outcome != gate.Fail {
		t.Fatalf("Outcome = %v, want Fail — prose mentioning \"perfgate\"/\"verdict\" is not a recorded result", c.Outcome)
	}
	if !strings.Contains(c.Field("msg"), "no perfgate VERDICT line") {
		t.Errorf("msg = %q, want it to name what is missing", c.Field("msg"))
	}
}

// TestCheckPerfEvidence_realVerdictLinePasses is the positive twin: a section
// carrying the actual convention every shipped release has used — a line
// starting with the literal "`perfgate` VERDICT: " token — passes, even
// though the surrounding paragraph ALSO uses the word "perfgate" and
// "verdict" in prose (proving the fix is about POSITION, not merely
// requiring the token to appear more than once).
func TestCheckPerfEvidence_realVerdictLinePasses(t *testing.T) {
	root := t.TempDir()
	writeChangelog(t, root, "# Changelog\n\n## [1.4.0]\n\n"+
		"Ran the perf gate on a real box; see below for the verdict it reached.\n\n"+
		"`perfgate` VERDICT: PASS — no regression vs v1.3.0 above each shape's floor — 10/10 shapes resolve "+
		"the 5.0% class.\n\n"+
		"[1.4.0]: https://example.com/compare/v1.3.0...v1.4.0\n")
	c := checkPerfEvidence(root, "1.4.0")
	if c.Outcome != gate.OK {
		t.Fatalf("Outcome = %v, want OK; msg=%q", c.Outcome, c.Field("msg"))
	}
}

// TestCheckChangelog_versionIsRegexEscaped confirms a version string containing regex
// metacharacters (a plausible pre-release tag, e.g. "1.4.0-rc.1") is matched LITERALLY, not
// interpreted as a pattern — regexp.QuoteMeta's job, pinned so a future edit that drops it
// fails loudly instead of silently over-matching.
func TestCheckChangelog_versionIsRegexEscaped(t *testing.T) {
	root := t.TempDir()
	// A literal ver containing a dot; without QuoteMeta, "1x4x0" would also match "1.4.0"
	// since '.' is "any character" in a pattern — assert it does NOT.
	writeChangelog(t, root, "# Changelog\n\n## [1x4x0]\n\n[1x4x0]: https://example.com\n")
	c := checkChangelog(root, "1.4.0")
	if c.Outcome != gate.Fail {
		t.Errorf("Outcome = %v, want Fail — '.' in the version must not match any character", c.Outcome)
	}
}
