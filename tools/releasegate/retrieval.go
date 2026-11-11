package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"

	"github.com/townsendmerino/aikit/tools/gate"
	"github.com/townsendmerino/aikit/tools/retrievalzone"
)

// (7) RETRIEVAL MARK — the release's CHANGELOG section must say whether the release
// could change retrieval results or the retrieval API, and may not say "no" when the
// files changed since the previous tag say "yes".
//
// aikit serves two consumers on one version line. ken uses the retrieval packages and
// re-runs its parity and recall checks on every aikit bump; goinfer drives most
// releases through linalg and embed's model-file readers. The mark is what lets ken
// skip those checks on a release that cannot reach it. tools/retrievalzone computes it
// from the changed files (see its package doc for the method and where it fails safe).
//
// The line is matched at the START of a line, in the form tools/retrievalmark prints,
// for the reason perfgateVerdictLine gives: prose that mentions the word must not
// satisfy the gate.
//
// A recorded "yes" is accepted when the tool says "no": the author may know something
// the files do not show, and a spurious yes costs ken one run of its checks. A
// recorded "no" over a computed "yes" is the failure this check exists for.
var retrievalMarkLine = regexp.MustCompile(`(?m)^RETRIEVAL: (yes|no)\b`)

func checkRetrievalMark(root, ver string) gate.Cell {
	const name = "retrieval-mark"
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
	prev := prevTag(root, ver)
	if prev == "" {
		return inconMsg(name, "no previous root tag to compare against")
	}
	m, err := retrievalzone.Between(root, prev, "")
	if err != nil {
		return inconMsg(name, "could not compare against "+prev+": "+err.Error())
	}
	return judgeRetrievalMark(sec, ver, prev, m)
}

// judgeRetrievalMark compares the recorded mark with the computed one.
func judgeRetrievalMark(sec, ver, prev string, m retrievalzone.Mark) gate.Cell {
	const name = "retrieval-mark"
	want := m.Line(prev)
	fail := func(msg string) gate.Cell { return failMsg(name, msg) } // run() prints every failure's reason
	found := retrievalMarkLine.FindAllStringSubmatch(sec, -1)
	switch {
	case len(found) == 0:
		return fail(fmt.Sprintf("CHANGELOG [%s] has no RETRIEVAL line. Run `go run -C tools ./retrievalmark` "+
			"and paste its last line on a line of its own: %q", ver, want))
	case len(found) > 1:
		return fail(fmt.Sprintf("CHANGELOG [%s] has %d RETRIEVAL lines; it must have one", ver, len(found)))
	}
	recorded := found[0][1]
	if recorded == "no" && m.Yes {
		for _, f := range m.Files {
			if f.Zone {
				fmt.Printf("::error::release-gate: retrieval zone: %s — %s\n", f.Path, f.Why)
			}
		}
		return fail(fmt.Sprintf("CHANGELOG [%s] says RETRIEVAL: no, but %d file(s) changed since %s are in the "+
			"retrieval zone. The line should be: %q", ver, len(m.InZone()), prev, want))
	}
	computed := "no"
	if m.Yes {
		computed = "yes"
	}
	fmt.Printf("release-gate: retrieval mark recorded for v%s: %s (computed vs %s: %s)\n", ver, recorded, prev, computed)
	return gate.Cell{Name: name, Outcome: gate.OK,
		Fields: []gate.Field{{Key: "recorded", State: recorded}, {Key: "computed", State: computed}}}
}
