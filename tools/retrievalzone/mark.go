package retrievalzone

import (
	"bytes"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// File is the verdict on one changed path.
type File struct {
	Path string
	Zone bool
	Why  string
}

// Mark is the answer for one change: Yes if any changed path could change retrieval
// results or the retrieval API.
type Mark struct {
	Yes   bool
	Files []File // every changed path, sorted
}

// InZone lists the changed paths that are in the zone.
func (m Mark) InZone() []string {
	var out []string
	for _, f := range m.Files {
		if f.Zone {
			out = append(out, f.Path)
		}
	}
	return out
}

// Line is the one-line form recorded in a CHANGELOG entry and checked by the release
// gate, e.g. "RETRIEVAL: no — vs v1.76.0, none of 12 changed files is in the retrieval zone".
func (m Mark) Line(from string) string {
	in := m.InZone()
	if !m.Yes {
		return fmt.Sprintf("RETRIEVAL: no — vs %s, none of %d changed files is in the retrieval zone", from, len(m.Files))
	}
	show := in
	more := ""
	if len(show) > 4 {
		more = fmt.Sprintf(", and %d more", len(show)-4)
		show = show[:4]
	}
	return fmt.Sprintf("RETRIEVAL: yes — vs %s, %d of %d changed files are in the retrieval zone (%s%s)",
		from, len(in), len(m.Files), strings.Join(show, ", "), more)
}

// Compare classifies every path that differs between two trees. A path is judged in
// each tree it exists in, and is in the zone if either says so. If a tree cannot be
// analyzed, every path is in the zone and the reason says why.
func Compare(old, new Source, changed []string) Mark {
	sort.Strings(changed)
	oldA, oldErr := Analyze(old)
	newA, newErr := Analyze(new)
	var m Mark
	for _, p := range changed {
		f := File{Path: p}
		switch {
		case oldErr != nil:
			f.Zone, f.Why = true, "unclassified: the old tree did not analyze: "+oldErr.Error()
		case newErr != nil:
			f.Zone, f.Why = true, "unclassified: the new tree did not analyze: "+newErr.Error()
		case p == "go.mod":
			f.Zone, f.Why = goModChange(old, new)
		default:
			inOld, inNew := oldA.files[p], newA.files[p]
			var whyOld, whyNew string
			var zOld, zNew bool
			if inOld {
				zOld, whyOld = oldA.Classify(p)
			}
			if inNew {
				zNew, whyNew = newA.Classify(p)
			}
			switch {
			case zNew:
				f.Zone, f.Why = true, whyNew
			case zOld:
				f.Zone, f.Why = true, "before the change: "+whyOld
			case inNew:
				f.Why = whyNew
			case inOld:
				f.Why = whyOld
			default:
				f.Zone, f.Why = true, "unclassified: in neither tree"
			}
		}
		m.Yes = m.Yes || f.Zone
		m.Files = append(m.Files, f)
	}
	return m
}

// goModNoise is what a consumer's build does not read in a dependency's go.mod: the
// toolchain line, and blank lines.
var goModNoise = regexp.MustCompile(`(?m)^(toolchain\s.*)?\r?\n`)

// goModChange: the root go.mod's `toolchain` line binds only builds of this module, so
// a change to it alone reaches no consumer. Anything else in go.mod (the go version,
// a dependency's version) can.
func goModChange(old, new Source) (bool, string) {
	a, errA := old.Read("go.mod")
	b, errB := new.Read("go.mod")
	if errA != nil || errB != nil {
		return true, "unclassified: go.mod is missing from one tree"
	}
	if bytes.Equal(goModNoise.ReplaceAll(a, nil), goModNoise.ReplaceAll(b, nil)) {
		return false, "only the toolchain line changed, which a consumer's build ignores"
	}
	return true, "go.mod changed beyond the toolchain line (go version or a dependency)"
}
