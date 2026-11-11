package main

import (
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/tools/gate"
)

func TestJudgeVulnStatement(t *testing.T) {
	for _, tc := range []struct {
		name, sec string
		want      gate.Outcome
		msg       string
	}{
		{"the tool's clean line", "\nSTATEMENT: no reachable vulnerabilities in 11/11 modules at d6d1dec (2026-11-11T00:38:49Z)\n", gate.OK, ""},
		{"with a note after it, as past entries have", "\nSTATEMENT: no reachable vulnerabilities in 16/16 modules at 6d95b81 (…). nobara (linux/amd64), Go 1.27.3.\n", gate.OK, ""},
		{"no line", "\n### Changed\n- a thing\n", gate.Fail, "no clean vulncheck STATEMENT"},
		{"an incomplete scan is not a statement", "\nSTATEMENT: INCOMPLETE — 9 clean, 0 vulnerable, 2 unscanned, of 11 at d6d1dec\n", gate.Fail, "no clean vulncheck STATEMENT"},
		{"a refusal is not a statement", "\nSTATEMENT: INCONCLUSIVE — vulncheck must run on linux\n", gate.Fail, "no clean vulncheck STATEMENT"},
		{"not every module", "\nSTATEMENT: no reachable vulnerabilities in 9/11 modules at d6d1dec\n", gate.Fail, "covers 9 of 11"},
		{"prose that mentions it", "\nThe STATEMENT: no reachable vulnerabilities in 11/11 modules line is pending.\n", gate.Fail, "no clean vulncheck STATEMENT"},
		{"a recorded exception", "\n> **VULNCHECK EXCEPTION: the vuln DB was unreachable on release day; scanned clean the day before at abc1234.**\n", gate.OK, ""},
	} {
		c := judgeVulnStatement(tc.sec, "1.2.3")
		if c.Outcome != tc.want || !strings.Contains(c.Field("msg"), tc.msg) {
			t.Errorf("%s: outcome %s, msg %q; want %s, msg containing %q", tc.name, c.Outcome, c.Field("msg"), tc.want, tc.msg)
		}
	}
}

// perfgate's line as printed passes; a line measured with --allow-dirty does not.
func TestPerfEvidence_pasteAndDirty(t *testing.T) {
	root := t.TempDir()
	write := func(sec string) {
		t.Helper()
		writeChangelog(t, root, "# Changelog\n\n## [1.2.3] — today\n\n"+sec+"\n\n## [1.2.2] — before\n")
	}
	write("`perfgate` VERDICT: PASS — no regression vs v1.2.2 above each shape's floor — 4/49 shapes resolve the 5.0% class")
	if c := checkPerfEvidence(root, "1.2.3"); c.Outcome != gate.OK {
		t.Errorf("perfgate's line as printed: %+v, want OK", c)
	}
	write("`perfgate` VERDICT: PASS — no regression vs v1.2.2 — DIRTY TREE, NOT RELEASE EVIDENCE")
	if c := checkPerfEvidence(root, "1.2.3"); c.Outcome != gate.Fail || !strings.Contains(c.Field("msg"), "--allow-dirty") {
		t.Errorf("a line measured with --allow-dirty: %+v, want a failure naming --allow-dirty", c)
	}
}
