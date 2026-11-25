package main

import (
	"bufio"
	"go/build"
	"go/build/constraint"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// config is one build configuration: a GOOS/GOARCH pair, a set of build tags and a set of GOEXPERIMENTs.
type config struct {
	goos, goarch string
	tags         []string
	experiments  []string
}

func (c config) String() string {
	s := c.goos + "/" + c.goarch
	if len(c.tags) > 0 {
		s += " tags=" + strings.Join(c.tags, ",")
	}
	if len(c.experiments) > 0 {
		s += " GOEXPERIMENT=" + strings.Join(c.experiments, ",")
	}
	return s
}

func (c config) env() []string {
	env := []string{"GOOS=" + c.goos, "GOARCH=" + c.goarch, "CGO_ENABLED=1", "GOFLAGS=-tags=" + strings.Join(c.tags, ",")}
	if len(c.experiments) > 0 {
		env = append(env, "GOEXPERIMENT="+strings.Join(c.experiments, ","))
	}
	return env
}

// notATag are the build-constraint terms a configuration sets by other means, or never.
var notATag = map[string]bool{"cgo": true, "gc": true, "gccgo": true, "unix": true, "ignore": true}

// knownOS and knownArch are the GOOS and GOARCH names a constraint can name.
var knownOS = map[string]bool{}
var knownArch = map[string]bool{}

func init() {
	for _, s := range strings.Fields("aix android darwin dragonfly freebsd hurd illumos ios js linux nacl netbsd openbsd plan9 solaris wasip1 windows zos") {
		knownOS[s] = true
	}
	for _, s := range strings.Fields("386 amd64 amd64p32 arm arm64 arm64be armbe loong64 mips mipsle mips64 mips64le mips64p32 mips64p32le ppc ppc64 ppc64le riscv riscv64 s390 s390x sparc sparc64 wasm") {
		knownArch[s] = true
	}
}

// constraintTerms returns the tags a file's //go:build line names, minus GOOS, GOARCH, release and toolchain terms.
func constraintTerms(path string) (tags, experiments []string) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		ln := strings.TrimSpace(sc.Text())
		if strings.HasPrefix(ln, "package ") {
			break
		}
		if !constraint.IsGoBuild(ln) {
			continue
		}
		x, err := constraint.Parse(ln)
		if err != nil {
			continue
		}
		x.Eval(func(tag string) bool {
			switch {
			case strings.HasPrefix(tag, "goexperiment."):
				experiments = append(experiments, strings.TrimPrefix(tag, "goexperiment."))
			case notATag[tag], knownOS[tag], knownArch[tag], strings.HasPrefix(tag, "go1."):
			default:
				tags = append(tags, tag)
			}
			return true
		})
	}
	return tags, experiments
}

// candidates are the configurations the census may choose from: each platform with no tags, with every tag, and with
// each tag alone; and each of those again with every GOEXPERIMENT the files name.
func candidates(platforms [][2]string, tags, experiments []string) []config {
	var sets [][]string
	sets = append(sets, nil)
	if len(tags) > 1 {
		sets = append(sets, tags)
	}
	for _, t := range tags {
		sets = append(sets, []string{t})
	}
	var out []config
	for _, p := range platforms {
		for _, s := range sets {
			out = append(out, config{goos: p[0], goarch: p[1], tags: s})
			if len(experiments) > 0 {
				out = append(out, config{goos: p[0], goarch: p[1], tags: s, experiments: experiments})
			}
		}
	}
	return out
}

// fileRule is what decides whether a configuration builds a file: its //go:build expression and its name's GOOS/GOARCH
// suffix. Each file is read once; go/build's MatchFile would read it once per candidate.
type fileRule struct {
	expr     constraint.Expr // nil: no //go:build line
	nameTags []string        // the _GOOS, _GOARCH or _GOOS_GOARCH suffix, all of which must hold
}

func readRule(path string) fileRule {
	var r fileRule
	if f, err := os.Open(path); err == nil {
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			ln := strings.TrimSpace(sc.Text())
			if strings.HasPrefix(ln, "package ") {
				break
			}
			if constraint.IsGoBuild(ln) {
				if x, err := constraint.Parse(ln); err == nil {
					r.expr = x
				}
			}
		}
		f.Close()
	}
	// go/build's goodOSArchFile: only the part after the first underscore counts, and a _test suffix is dropped.
	name := strings.TrimSuffix(filepath.Base(path), ".go")
	if i := strings.Index(name, "_"); i >= 0 {
		l := strings.Split(name[i:], "_")
		if n := len(l); n > 0 && l[n-1] == "test" {
			l = l[:n-1]
		}
		switch n := len(l); {
		case n >= 2 && knownOS[l[n-2]] && knownArch[l[n-1]]:
			r.nameTags = []string{l[n-2], l[n-1]}
		case n >= 1 && (knownOS[l[n-1]] || knownArch[l[n-1]]):
			r.nameTags = []string{l[n-1]}
		}
	}
	return r
}

var unixOS = map[string]bool{"aix": true, "android": true, "darwin": true, "dragonfly": true, "freebsd": true, "hurd": true,
	"illumos": true, "ios": true, "linux": true, "netbsd": true, "openbsd": true, "solaris": true}

// tagFunc is go/build's matchTag for c: GOOS (with android ⊃ linux, ios ⊃ darwin, illumos ⊃ solaris), GOARCH, unix,
// cgo, gc, every go1.N release tag, the configuration's tags and its experiments.
func (c config) tagFunc() func(string) bool {
	on := map[string]bool{c.goos: true, c.goarch: true, "cgo": true, "gc": true}
	switch c.goos {
	case "android":
		on["linux"] = true
	case "ios":
		on["darwin"] = true
	case "illumos":
		on["solaris"] = true
	}
	if unixOS[c.goos] {
		on["unix"] = true
	}
	for _, t := range build.Default.ReleaseTags {
		on[t] = true
	}
	for _, t := range c.tags {
		on[t] = true
	}
	for _, e := range c.experiments {
		on["goexperiment."+e] = true
	}
	return func(t string) bool { return on[t] }
}

func (r fileRule) builds(tag func(string) bool) bool {
	for _, t := range r.nameTags {
		if !tag(t) {
			return false
		}
	}
	return r.expr == nil || r.expr.Eval(tag)
}

// cover picks each platform's untagged configuration, then more greedily until every file some candidate builds is
// built by one picked. It returns the picks and the files no candidate builds.
func cover(files []string, cands []config) (picked []config, never []string) {
	rules := make([]fileRule, len(files))
	for i, f := range files {
		rules[i] = readRule(f)
	}
	builds := make([]map[string]bool, len(cands))
	all := map[string]bool{}
	for i, c := range cands {
		tag := c.tagFunc()
		builds[i] = map[string]bool{}
		for j, f := range files {
			if rules[j].builds(tag) {
				builds[i][f] = true
				all[f] = true
			}
		}
	}
	for _, f := range files {
		if !all[f] {
			never = append(never, f)
		}
	}
	left := maps2(all)
	used := make([]bool, len(cands))
	// Every platform is checked untagged, so a name declared on one platform only is seen as such; the greedy picks
	// then add the tags and experiments the files need.
	seen := map[string]bool{}
	for i, c := range cands {
		if p := c.goos + "/" + c.goarch; len(c.tags) == 0 && len(c.experiments) == 0 && !seen[p] {
			seen[p], used[i] = true, true
			picked = append(picked, c)
			for f := range builds[i] {
				delete(left, f)
			}
		}
	}
	for len(left) > 0 {
		best, bestN := -1, 0
		for i := range cands {
			if used[i] {
				continue
			}
			n := 0
			for f := range left {
				if builds[i][f] {
					n++
				}
			}
			if n > bestN {
				best, bestN = i, n
			}
		}
		if best < 0 {
			break
		}
		used[best] = true
		picked = append(picked, cands[best])
		for f := range builds[best] {
			delete(left, f)
		}
	}
	return picked, never
}

func maps2(m map[string]bool) map[string]bool {
	c := make(map[string]bool, len(m))
	for k, v := range m {
		c[k] = v
	}
	return c
}

// sortedUnique sorts and dedupes.
func sortedUnique(s []string) []string {
	slices.Sort(s)
	return slices.Compact(s)
}
