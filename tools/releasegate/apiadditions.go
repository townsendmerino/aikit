package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/townsendmerino/aikit/tools/canary"
	"github.com/townsendmerino/aikit/tools/gate"
	"github.com/townsendmerino/aikit/tools/gpumod"
)

// (6) API ADDITIONS ARE IN THE CHANGELOG. The apidiff check above only looks for breakage in the Hard tier; an exported symbol
// ADDED anywhere in the module (an Experimental package, a method on an existing type) needs no gate to ship, so whether the release
// notes say so has been up to whoever writes them. This compares every public package against the previous tag and requires each added
// symbol to be named in the release's CHANGELOG section.
//
// The rule is a substring match on purpose (a whole word, anywhere in the section): it catches the omission, which is what
// happens, and not a careless sentence, which no tool catches. A symbol path is `Name`, `Type.Member` or `(*Type).Member`:
//   - a plain symbol is covered when its name appears;
//   - a member is covered when its own name appears, or when its type is itself new in this release and the type's name appears
//     (a new type's methods are described with the type);
//   - a package that did not exist at the previous tag is covered when its path's last element appears.
//
// A symbol whose name ends in ForTest is a cross-package test seam and is exempt. Any other symbol that is deliberately not release-note material goes in tools/releasegate/api-waivers.txt, one `<package> <symbol> <reason>`
// per line, the reason mandatory: a waiver is a committed sentence, printed by the gate every time it is used.

// apiChange is one line of apidiff's report.
type apiChange struct {
	Section string // "Incompatible changes" or "Compatible changes"
	Symbol  string // the path before the colon: Name, Type.Member or (*Type).Member
	What    string // added, removed, changed type, ...
}

var apiChangeLine = regexp.MustCompile(`^- (.+?): (.+)$`)

// parseAPIDiff reads apidiff's default output: a `Section:` heading, then `- symbol: what` lines. Anything else is ignored.
func parseAPIDiff(out string) []apiChange {
	var cs []apiChange
	section := ""
	for _, ln := range strings.Split(out, "\n") {
		ln = strings.TrimRight(ln, " \t\r")
		switch {
		case strings.HasSuffix(ln, " changes:") && !strings.HasPrefix(ln, "-"):
			section = strings.TrimSuffix(ln, ":")
		case section != "":
			if m := apiChangeLine.FindStringSubmatch(ln); m != nil {
				cs = append(cs, apiChange{Section: section, Symbol: m[1], What: m[2]})
			}
		}
	}
	return cs
}

// splitSymbol splits `(*Type).Member`, `Type.Member` and `Name` into the type (empty for a plain symbol) and the member or name.
func splitSymbol(sym string) (typ, member string) {
	s := strings.TrimPrefix(sym, "(")
	s = strings.Replace(s, ")", "", 1)
	s = strings.TrimPrefix(s, "*")
	if i := strings.Index(s, "."); i >= 0 {
		return s[:i], s[i+1:]
	}
	return "", s
}

// mentions reports whether word appears in text as a whole word.
func mentions(text, word string) bool {
	if word == "" {
		return false
	}
	return regexp.MustCompile(`(^|[^A-Za-z0-9_])` + regexp.QuoteMeta(word) + `($|[^A-Za-z0-9_])`).MatchString(text)
}

// uncoveredAdditions returns the added symbols of one package that sec does not name, as `pkg symbol` strings.
func uncoveredAdditions(pkg string, changes []apiChange, sec string, waived func(pkg, sym string) bool) []string {
	newTypes := map[string]bool{}
	for _, c := range changes {
		if c.Section == "Compatible changes" && c.What == "added" {
			if typ, name := splitSymbol(c.Symbol); typ == "" {
				newTypes[name] = true
			}
		}
	}
	var missing []string
	for _, c := range changes {
		if c.Section != "Compatible changes" || c.What != "added" {
			continue
		}
		typ, member := splitSymbol(c.Symbol)
		if strings.HasSuffix(member, "ForTest") { // an exported test seam (a cross-package test hook), by this repo's naming: not release-note material
			continue
		}
		covered := mentions(sec, member) || (typ != "" && newTypes[typ] && mentions(sec, typ))
		if !covered && !waived(pkg, c.Symbol) {
			missing = append(missing, pkg+" "+c.Symbol)
		}
	}
	sort.Strings(missing)
	return missing
}

// loadAPIWaivers reads api-waivers.txt (absent is none). A line without a reason is an error, not a waiver.
func loadAPIWaivers(path string) (map[string]string, error) {
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	w := map[string]string{}
	for i, ln := range strings.Split(string(b), "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" || strings.HasPrefix(ln, "#") {
			continue
		}
		f := strings.Fields(ln)
		if len(f) < 3 {
			return nil, fmt.Errorf("%s:%d: a waiver is `<package> <symbol> <reason>`; the reason is mandatory", filepath.Base(path), i+1)
		}
		w[f[0]+" "+f[1]] = strings.Join(f[2:], " ")
	}
	return w, nil
}

// publicPackages lists the root module's packages whose API is part of what a release ships: not internal, tools or examples.
func publicPackages(root string) ([]string, error) {
	out, rc := gpumod.Exec(root, "", nil, "go", "list", "./...")
	if rc != 0 {
		return nil, fmt.Errorf("go list ./... failed: %s", firstLine(out))
	}
	var pkgs []string
	for _, p := range strings.Fields(out) {
		rel := strings.TrimPrefix(strings.TrimPrefix(p, rootMod), "/")
		if rel == "" || strings.HasPrefix(rel, "internal") || strings.Contains(rel, "/internal") || strings.HasPrefix(rel, "tools") ||
			strings.HasPrefix(rel, "examples") || strings.HasPrefix(rel, "bench/cmd") {
			continue
		}
		pkgs = append(pkgs, rel)
	}
	return pkgs, nil
}

func checkAPIAdditions(root, ver string) gate.Cell {
	name := "changelog-api"
	data, err := os.ReadFile(filepath.Join(root, "CHANGELOG.md"))
	if err != nil {
		return failMsg(name, "cannot read CHANGELOG.md: "+err.Error())
	}
	sec, ok := changelogSection(string(data), ver)
	if !ok {
		return gate.Cell{Name: name, Outcome: gate.OK, Fields: []gate.Field{{Key: "section", State: "absent (reported by changelog check)"}}}
	}
	// CANARY: the parser must find the known break in the apidiff fixture pair before "no additions are missing" means anything.
	if parseCanary := parseAPIDiff(apidiffCanaryOutAll(root)); len(parseCanary) == 0 {
		return inconMsg(name, "CANNOT-EVALUATE — the apidiff output parser found nothing in the canary fixture's known change")
	}
	prev := prevTag(root, ver)
	if prev == "" {
		fmt.Println("release-gate: no previous tag — skipping the API additions check")
		return okCell(name)
	}
	waivers, err := loadAPIWaivers(filepath.Join(root, "tools", "releasegate", "api-waivers.txt"))
	if err != nil {
		return failMsg(name, err.Error())
	}
	pkgs, err := publicPackages(root)
	if err != nil {
		return inconMsg(name, err.Error())
	}
	wt, err := os.MkdirTemp("", "releasegate-api-wt")
	if err != nil {
		return inconMsg(name, "mktemp: "+err.Error())
	}
	if _, rc := gpumod.Exec(root, "", nil, "git", "worktree", "add", "-q", "-f", wt, prev); rc != 0 {
		return inconMsg(name, "git worktree add "+prev+" failed")
	}
	defer func() {
		gpumod.Exec(root, "", nil, "git", "worktree", "remove", "--force", wt)
		gpumod.Exec(root, "", nil, "git", "worktree", "prune")
	}()
	used := map[string]bool{}
	waived := func(pkg, sym string) bool {
		if _, ok := waivers[pkg+" "+sym]; ok {
			used[pkg+" "+sym] = true
			return true
		}
		return false
	}
	var missing []string
	fmt.Printf("release-gate: API additions against %s (%d public packages)\n", prev, len(pkgs))
	for _, p := range pkgs {
		if _, err := os.Stat(filepath.Join(wt, p)); err != nil { // the package is new in this release
			if !mentions(sec, filepath.Base(p)) && !waived(p, "package") {
				missing = append(missing, p+" (new package)")
			}
			continue
		}
		base := filepath.Join(os.TempDir(), "gate-add-"+strings.ReplaceAll(p, "/", "-")+".api")
		if o, rc := gpumod.Exec(wt, "", nil, "go", "run", apidiffPkg, "-w", base, rootMod+"/"+p); rc != 0 {
			return inconMsg(name, "apidiff could not record the "+prev+" baseline for "+p+": "+firstLine(o))
		}
		out, rc := gpumod.Exec(root, "", nil, "go", "run", apidiffPkg, base, rootMod+"/"+p)
		if rc != 0 {
			return inconMsg(name, "apidiff could not compare "+p+" against "+prev+": "+firstLine(out))
		}
		missing = append(missing, uncoveredAdditions(p, parseAPIDiff(out), sec, waived)...)
	}
	for _, k := range sortedKeys(used) {
		fmt.Printf("release-gate: waived %s: %s\n", k, waivers[k])
	}
	if len(missing) > 0 {
		fmt.Printf("::error::release-gate: added API with no CHANGELOG [%s] line:\n  %s\n", ver, strings.Join(missing, "\n  "))
		return failMsg(name, fmt.Sprintf("%d added symbol(s) not named in the CHANGELOG: %s", len(missing), strings.Join(missing, "; ")))
	}
	fmt.Println("release-gate: every added exported symbol is named in the CHANGELOG [" + ver + "]")
	return okCell(name)
}

func sortedKeys(m map[string]bool) []string {
	var ks []string
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}

// apidiffCanaryOutAll is apidiffCanaryOut without -incompatible: the full report for the fixture pair, in the shape the additions check reads.
func apidiffCanaryOutAll(root string) string {
	toolsDir := filepath.Join(root, "tools")
	base := filepath.Join(os.TempDir(), "canary-apidiff-all-old.api")
	if _, rc := gpumod.Exec(toolsDir, "", nil, "go", "run", apidiffPkg, "-w", base, canary.ApidiffFixtureOld); rc != 0 {
		return ""
	}
	out, _ := gpumod.Exec(toolsDir, "", nil, "go", "run", apidiffPkg, base, canary.ApidiffFixtureNew)
	return out
}
