package main

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"
)

// columns are the consumers, in table order. Each but the first pair is a repository; "linalg" is the target package.
var columns = []string{"linalg", "linalg tests", "aikit", "aikit tests", "goinfer", "goinfer tests", "ken", "ken tests", "tack", "tack tests"}

// column places a file: its repository, or the target package within aikit, and whether it is a test file.
func (r *result) column(file string) string {
	best := (*repo)(nil)
	for _, rp := range r.repos {
		if strings.HasPrefix(file, rp.root+string(filepath.Separator)) && (best == nil || len(rp.root) > len(best.root)) {
			best = rp
		}
	}
	if best == nil {
		return ""
	}
	col := best.name
	if best.name == "aikit" {
		if rel, err := filepath.Rel(r.aikit, filepath.Dir(file)); err == nil && filepath.ToSlash(rel) == r.target {
			col = "linalg"
		}
	}
	if strings.HasSuffix(file, "_test.go") {
		col += " tests"
	}
	return col
}

// reach is, for one name, the files in each column by how they reach it.
type reach map[string]map[string]map[string]bool // column → how → file

func (r *result) reaches() map[string]reach {
	out := map[string]reach{}
	for u := range r.c.uses {
		if r.c.names[u.key] == nil || !r.tracked[u.file] {
			continue // a test-only declaration, a method of an unexported type, or a file git does not track
		}
		col := r.column(u.file)
		if col == "" {
			continue
		}
		rc := out[u.key]
		if rc == nil {
			rc = reach{}
			out[u.key] = rc
		}
		if rc[col] == nil {
			rc[col] = map[string]map[string]bool{}
		}
		if rc[col][u.how] == nil {
			rc[col][u.how] = map[string]bool{}
		}
		rc[col][u.how][u.file] = true
	}
	return out
}

// has reports whether the column reaches the name named or by value (an iface reach is a possibility, not a reach).
func (rc reach) has(col string) bool {
	return len(rc[col][named]) > 0 || len(rc[col][value]) > 0
}

func (rc reach) cell(col string) string {
	var parts []string
	n := len(rc[col][named])
	if n > 0 {
		parts = append(parts, fmt.Sprint(n))
	}
	if v := countNotIn(rc[col][value], rc[col][named]); v > 0 {
		parts = append(parts, fmt.Sprintf("%dv", v))
	}
	if i := countNotIn(rc[col][iface], rc[col][named], rc[col][value]); i > 0 {
		parts = append(parts, fmt.Sprintf("%di", i))
	}
	return strings.Join(parts, "+")
}

func countNotIn(s map[string]bool, others ...map[string]bool) int {
	n := 0
	for f := range s {
		in := false
		for _, o := range others {
			if o[f] {
				in = true
			}
		}
		if !in {
			n++
		}
	}
	return n
}

// Category is what the census says to do with a name, before anyone's judgment.
const (
	catProd      = "production outside linalg"
	catTests     = "only tests outside linalg"
	catInside    = "only linalg's own code"
	catInsideTst = "only linalg's own tests"
	catNothing   = "nothing"
)

var categories = []string{catProd, catTests, catInside, catInsideTst, catNothing}

func (rc reach) category() string {
	switch {
	case rc.has("aikit") || rc.has("goinfer") || rc.has("ken") || rc.has("tack"):
		return catProd
	case rc.has("aikit tests") || rc.has("goinfer tests") || rc.has("ken tests") || rc.has("tack tests"):
		return catTests
	case rc.has("linalg"):
		return catInside
	case rc.has("linalg tests"):
		return catInsideTst
	}
	return catNothing
}

// platforms lists the GOOS/GOARCH pairs of the configurations a name is declared in, or "" when it is declared in all.
func (r *result) platforms(n *Name) string {
	all, got := map[string]bool{}, map[string]bool{}
	for _, c := range r.configs {
		all[c.goos+"/"+c.goarch] = true
		if n.Configs[c.String()] {
			got[c.goos+"/"+c.goarch] = true
		}
	}
	if len(got) == len(all) {
		return ""
	}
	return strings.Join(sortedKeys(got), " ")
}

func sortedKeys[V any](m map[string]V) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	slices.Sort(ks)
	return ks
}

func (r *result) byCategory() map[string][]string {
	rs := r.reaches()
	out := map[string][]string{}
	for _, k := range sortedKeys(r.c.names) {
		rc := rs[k]
		if rc == nil {
			rc = reach{}
		}
		out[rc.category()] = append(out[rc.category()], k)
	}
	return out
}

func (r *result) kindCounts(keys []string) string {
	n := map[string]int{}
	for _, k := range keys {
		n[r.c.names[k].Kind]++
	}
	var parts []string
	for _, kind := range []string{"func", "type", "const", "var", "method", "field"} {
		if n[kind] > 0 {
			parts = append(parts, fmt.Sprintf("%d %s", n[kind], kind))
		}
	}
	return strings.Join(parts, ", ")
}

func (r *result) summary() string {
	var b strings.Builder
	bc := r.byCategory()
	fmt.Fprintf(&b, "exported names: %d (%s)\n", len(r.c.names), r.kindCounts(sortedKeys(r.c.names)))
	for _, c := range categories {
		fmt.Fprintf(&b, "  %-28s %4d (%s)\n", c+":", len(bc[c]), r.kindCounts(bc[c]))
	}
	fmt.Fprintf(&b, "package checks with type errors (the package does not build in that configuration): %d\n", len(r.c.errs))
	for _, f := range r.strays {
		fmt.Fprintf(&b, "not tracked by git, so not counted: %s\n", r.rel(f))
	}
	for _, f := range r.unclean {
		fmt.Fprintf(&b, "never checked cleanly, and imports the target: %s\n", r.rel(f))
	}
	return b.String()
}

func (r *result) markdown(repos []*repo) string {
	var b strings.Builder
	b.WriteString("# linalg census\n\nGenerated by `go run -C tools ./linalgcensus`; do not edit by hand.\n\n")
	for _, rp := range repos {
		fmt.Fprintf(&b, "- %s at `%s`\n", rp.name, rp.commit)
	}
	b.WriteString("\nConfigurations (chosen so every buildable Go file of every checkout is checked once):\n\n")
	for _, c := range r.configs {
		fmt.Fprintf(&b, "- %s\n", c)
	}
	if len(r.never) > 0 {
		fmt.Fprintf(&b, "\n%d files build in no candidate configuration and are not checked:\n\n", len(r.never))
		for _, f := range r.never {
			fmt.Fprintf(&b, "- `%s`\n", r.rel(f))
		}
	}
	bc := r.byCategory()
	fmt.Fprintf(&b, "\n## Summary\n\n%d exported names (%s).\n\n| reached by | names | kinds |\n|---|---|---|\n", len(r.c.names), r.kindCounts(sortedKeys(r.c.names)))
	for _, c := range categories {
		fmt.Fprintf(&b, "| %s | %d | %s |\n", c, len(bc[c]), r.kindCounts(bc[c]))
	}
	rs := r.reaches()
	for _, c := range categories[1:] {
		fmt.Fprintf(&b, "\n### %s (%d)\n\n", c, len(bc[c]))
		var names []string
		for _, k := range bc[c] {
			s := "`" + k + "`"
			if rc := rs[k]; rc != nil {
				var via []string
				for _, col := range columns {
					if len(rc[col][iface]) > 0 && !rc.has(col) {
						via = append(via, col)
					}
				}
				if len(via) > 0 {
					s += " (an interface in " + strings.Join(via, ", ") + " may dispatch to it)"
				}
			}
			names = append(names, s)
		}
		b.WriteString(strings.Join(names, ", ") + "\n")
	}
	b.WriteString("\n## Every name\n\nA cell counts files: `N` name it, `Nv` hold a value of the type without naming it, `Ni` declare an interface it satisfies (a possible dispatch). " +
		"Zone: the declaring file is in the retrieval zone (`tools/retrievalzone`). Platforms: where it is declared, when not everywhere.\n\n")
	b.WriteString("| name | kind | category | zone | platforms | " + strings.Join(columns, " | ") + " |\n|---|---|---|---|---|" + strings.Repeat("---|", len(columns)) + "\n")
	for _, k := range sortedKeys(r.c.names) {
		n := r.c.names[k]
		rc := rs[k]
		if rc == nil {
			rc = reach{}
		}
		zone := ""
		if r.zone != nil && n.Decl != "" && r.zone(n.Decl) {
			zone = "yes"
		}
		cells := make([]string, len(columns))
		for i, col := range columns {
			cells[i] = rc.cell(col)
		}
		fmt.Fprintf(&b, "| `%s` | %s | %s | %s | %s | %s |\n", k, n.Kind, rc.category(), zone, r.platforms(n), strings.Join(cells, " | "))
	}
	if len(r.c.errs) > 0 {
		b.WriteString("\n## Packages that do not build in a configuration\n\nEach was checked cleanly in another; its first error here:\n\n")
		for _, k := range sortedKeys(r.c.errs) {
			e := r.c.errs[k][0]
			for _, rp := range r.repos {
				e = strings.ReplaceAll(e, rp.root+"/", rp.name+"/")
			}
			fmt.Fprintf(&b, "- %s: %s\n", k, e)
		}
	}
	return b.String()
}

func (r *result) rel(f string) string {
	for _, rp := range r.repos {
		if rel, err := filepath.Rel(rp.root, f); err == nil && !strings.HasPrefix(rel, "..") {
			return rp.name + "/" + filepath.ToSlash(rel)
		}
	}
	return f
}

type jsonUse struct {
	Name    string   `json:"name"`
	File    string   `json:"file"`
	Column  string   `json:"column"`
	How     string   `json:"how"`
	Configs []string `json:"configs"`
}

func (r *result) jsonUses() []jsonUse {
	var out []jsonUse
	for u, cfgs := range r.c.uses {
		if r.c.names[u.key] == nil || !r.tracked[u.file] {
			continue
		}
		out = append(out, jsonUse{u.key, r.rel(u.file), r.column(u.file), u.how, sortedKeys(cfgs)})
	}
	slices.SortFunc(out, func(a, b jsonUse) int {
		return strings.Compare(a.Name+"\x00"+a.File+"\x00"+a.How, b.Name+"\x00"+b.File+"\x00"+b.How)
	})
	return out
}
