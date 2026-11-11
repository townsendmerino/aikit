package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"

	"github.com/townsendmerino/aikit/tools/gate"
)

// (8) VULNCHECK STATEMENT — the release's CHANGELOG section must carry the line
// tools/vulncheck prints for a complete, clean scan, or say in the open why it does not.
//
// RELEASING.md has called that line "a deliverable, not hygiene" since it was introduced,
// and it was the one piece of release evidence nothing checked: the perfgate line and the
// retrieval mark stop a release that forgot them, and this did not.
//
// vulnStatementLine matches the clean form only, with every module scanned ("N/N"; the two
// counts are compared below). "STATEMENT: INCOMPLETE — …" and "STATEMENT: INCONCLUSIVE — …"
// are what vulncheck prints when it could not make the statement, and a release cannot ship
// on those without an exception. Anchored to the start of a line, as the other two are.
var vulnStatementLine = regexp.MustCompile(`(?m)^STATEMENT: no reachable vulnerabilities in (\d+)/(\d+) modules\b`)

// vulnExceptionLine is the recorded exception, the same form as PERFGATE EXCEPTION: a
// blockquote that says why this release ships without a clean statement.
var vulnExceptionLine = regexp.MustCompile(`(?m)^>\s*\*\*VULNCHECK EXCEPTION:`)

func checkVulnStatement(root, ver string) gate.Cell {
	const name = "vuln-statement"
	data, err := os.ReadFile(filepath.Join(root, "CHANGELOG.md"))
	if err != nil {
		return failMsg(name, "cannot read CHANGELOG.md: "+err.Error())
	}
	sec, ok := changelogSection(string(data), ver)
	if !ok {
		// checkChangelog already fails on this; do not double-report.
		return gate.Cell{Name: name, Outcome: gate.OK,
			Fields: []gate.Field{{Key: "section", State: "absent (reported by changelog check)"}}}
	}
	return judgeVulnStatement(sec, ver)
}

func judgeVulnStatement(sec, ver string) gate.Cell {
	const name = "vuln-statement"
	if vulnExceptionLine.MatchString(sec) {
		fmt.Println("release-gate: VULNCHECK EXCEPTION recorded for v" + ver)
		return gate.Cell{Name: name, Outcome: gate.OK, Fields: []gate.Field{{Key: "vulncheck", State: "exception recorded"}}}
	}
	m := vulnStatementLine.FindStringSubmatch(sec)
	switch {
	case m == nil:
		return failMsg(name, fmt.Sprintf("CHANGELOG [%s] has no clean vulncheck STATEMENT line. Run `go run -C tools ./vulncheck` "+
			"and paste its STATEMENT line (\"STATEMENT: no reachable vulnerabilities in N/N modules at …\") on a line of its own, "+
			"or record a \"> **VULNCHECK EXCEPTION: …**\" blockquote saying why this release ships without one", ver))
	case m[1] != m[2]:
		return failMsg(name, fmt.Sprintf("CHANGELOG [%s]'s STATEMENT covers %s of %s modules; a statement is about all of them", ver, m[1], m[2]))
	}
	fmt.Printf("release-gate: vulncheck STATEMENT recorded for v%s (%s/%s modules)\n", ver, m[1], m[2])
	return gate.Cell{Name: name, Outcome: gate.OK, Fields: []gate.Field{{Key: "vulncheck", State: "recorded"}}}
}
