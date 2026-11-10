package retrievalzone

import (
	"fmt"
	"sort"
	"strings"
	"testing"
)

// mem is a Source held in memory.
type mem map[string]string

func (m mem) Files() []string {
	var out []string
	for f := range m {
		out = append(out, f)
	}
	sort.Strings(out)
	return out
}

func (m mem) Read(p string) ([]byte, error) {
	s, ok := m[p]
	if !ok {
		return nil, fmt.Errorf("%s: not in this tree", p)
	}
	return []byte(s), nil
}

func (m mem) with(kv ...string) mem {
	out := mem{}
	for k, v := range m {
		out[k] = v
	}
	for i := 0; i < len(kv); i += 2 {
		if kv[i+1] == "" {
			delete(out, kv[i])
		} else {
			out[kv[i]] = kv[i+1]
		}
	}
	return out
}

// base is a small module shaped like the real one: ann is a zone package, linalg is
// reached only in part, vision is outside.
var base = mem{
	"go.mod": "module example.com/m\n\ngo 1.27.0\n\nrequire golang.org/x/text v0.40.0\n",
	"go.sum": "x\n",

	"ann/flat.go": `package ann
import "example.com/m/linalg"
func Query(a, b []float32) float32 { return linalg.Dot(a, b) }
`,
	"linalg/dot.go": `package linalg
func Dot(a, b []float32) float32 { return dotAsm(a, b) + float32(table[0]) }
`,
	"linalg/dot_amd64.go": `package linalg
func dotAsm(a, b []float32) float32
`,
	"linalg/dot_amd64.s":    "TEXT ·dotAsm(SB), 4, $0\n\tMOVQ ·asmConst(SB), AX\n\tRET\n",
	"linalg/asmconst.go":    "package linalg\nvar asmConst uint64 = 7\n",
	"linalg/table.go":       "package linalg\nvar table = [1]int{1}\n",
	"linalg/vision_only.go": "package linalg\nfunc Attend() int { return helper() }\n",
	"linalg/helper.go":      "package linalg\nfunc helper() int { return 1 }\n",
	"linalg/other.s":        "TEXT ·unrelated(SB), 4, $0\n\tRET\n",
	"linalg/weight.go":      "package linalg\ntype Weight struct{}\n",
	"linalg/dispatch.go":    "package linalg\nvar useFast = detect()\nfunc detect() bool { return true }\n",
	"linalg/dot_test.go":    "package linalg\n",

	"vision/tower.go": `package vision
import "example.com/m/linalg"
func Forward() int { return linalg.Attend() }
`,
	"gpu/go.mod":     "module example.com/m/gpu\n",
	"gpu/metal.go":   "package gpu\n",
	"docs/notes.md":  "notes\n",
	"README.md":      "readme\n",
	"testdata/a.bin": "x",
}

func classify(t *testing.T, src mem, p string) (bool, string) {
	t.Helper()
	a, err := Analyze(src)
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	return a.Classify(p)
}

func TestClassify(t *testing.T) {
	for _, tc := range []struct {
		path string
		zone bool
		why  string // a substring of the reason
	}{
		{"ann/flat.go", true, "in zone package ann"},
		{"linalg/dot.go", true, "ann/flat.go: Query"},
		{"linalg/dot_amd64.go", true, "linalg/dot.go: Dot"},
		{"linalg/dot_amd64.s", true, "implements"},
		{"linalg/asmconst.go", true, "linalg/dot_amd64.go: dotAsm"}, // named only by the assembly
		{"linalg/table.go", true, "linalg/dot.go: Dot"},
		{"linalg/dispatch.go", true, "runs at import"}, // a call in a var initializer
		{"linalg/vision_only.go", false, "nothing in the zone reaches it"},
		{"linalg/helper.go", false, "nothing in the zone reaches it"},
		{"linalg/other.s", false, "nothing in the zone reaches it"},
		{"linalg/weight.go", false, "nothing in the zone reaches it"},
		{"linalg/dot_test.go", false, "test file"},
		{"vision/tower.go", false, "nothing in the zone reaches it"},
		{"gpu/metal.go", false, "nested module"},
		{"docs/notes.md", false, "prose"},
		{"README.md", false, "prose"},
		{"testdata/a.bin", false, "test fixture"},
		{"go.sum", false, "own go.sum"},
		{".github/workflows/ci.yml", false, "ignores"},
		// Fail safe: paths no rule knows.
		{"linalg/table.bin", true, "unclassified file in linalg"},
		{"Makefile", true, "unclassified"},
		{"newpkg/thing.txt", true, "unclassified"},
	} {
		zone, why := classify(t, base, tc.path)
		if zone != tc.zone || !strings.Contains(why, tc.why) {
			t.Errorf("Classify(%q) = %v, %q; want %v, a reason containing %q", tc.path, zone, why, tc.zone, tc.why)
		}
	}
}

// A reached type brings every method declared on it, wherever the method lives.
func TestMethodsFollowTheirType(t *testing.T) {
	src := base.with(
		"ann/flat.go", "package ann\nimport \"example.com/m/linalg\"\nvar W linalg.Weight\n",
		"linalg/weight_methods.go", "package linalg\nfunc (w *Weight) Rows() int { return 0 }\n",
	)
	if zone, why := classify(t, src, "linalg/weight_methods.go"); !zone {
		t.Errorf("a method on a reached type is not in the zone: %s", why)
	}
	if zone, _ := classify(t, base.with("linalg/weight_methods.go", "package linalg\nfunc (w *Weight) Rows() int { return 0 }\n"), "linalg/weight_methods.go"); zone {
		t.Error("a method on an unreached type is in the zone")
	}
}

// Everything a reached file declares is reached, so the reached set compiles alone.
func TestReachedFilesAreClosed(t *testing.T) {
	src := base.with("linalg/table.go", "package linalg\nvar table = [1]int{1}\nfunc Unused() int { return helper() }\n")
	if zone, why := classify(t, src, "linalg/helper.go"); !zone {
		t.Errorf("helper.go is named by a reached file but is not in the zone: %s", why)
	}
}

// A new init function runs at import whichever file holds it.
func TestInitIsARoot(t *testing.T) {
	src := base.with("linalg/weight.go", "package linalg\ntype Weight struct{}\nfunc init() { table[0] = 2 }\n")
	if zone, why := classify(t, src, "linalg/weight.go"); !zone || !strings.Contains(why, "runs at import") {
		t.Errorf("a file with an init function in a linked package: zone=%v, %q", zone, why)
	}
}

// A function with no body and no assembly file that defines it takes every assembly
// file in its package.
func TestBodylessWithoutDefinerTakesAllAssembly(t *testing.T) {
	src := base.with("linalg/dot_amd64.s", "// renamed\nTEXT ·dotAsm2(SB), 4, $0\n\tRET\n")
	for _, f := range []string{"linalg/dot_amd64.s", "linalg/other.s"} {
		if zone, why := classify(t, src, f); !zone {
			t.Errorf("%s is not in the zone: %s", f, why)
		}
	}
}

func mark(t *testing.T, old, new mem) Mark {
	t.Helper()
	seen := map[string]bool{}
	var changed []string
	for _, m := range []mem{old, new} {
		for f := range m {
			if old[f] != new[f] && !seen[f] {
				seen[f] = true
				changed = append(changed, f)
			}
		}
	}
	return Compare(old, new, changed)
}

func TestCompare(t *testing.T) {
	edit := func(f string) mem { return base.with(f, base[f]+"\n// edited\n") }
	for _, tc := range []struct {
		name string
		new  mem
		yes  bool
	}{
		{"nothing outside the zone", edit("linalg/vision_only.go").with("vision/tower.go", base["vision/tower.go"]+"//x\n", "README.md", "new\n"), false},
		{"a zone package", edit("ann/flat.go"), true},
		{"a reached kernel", edit("linalg/dot_amd64.s"), true},
		{"an unreached kernel", edit("linalg/other.s"), false},
		{"a test file", edit("linalg/dot_test.go"), false},
		{"a nested module", edit("gpu/metal.go"), false},
		// A new file needs no entry anywhere: it is walked like the rest.
		{"a new file nothing reaches", base.with("linalg/new_tower.go", "package linalg\nfunc Tower() {}\n"), false},
		{"a new file the zone reaches", base.with("linalg/new_kernel.go", "package linalg\nfunc init() {}\n"), true},
		{"a new unknown file beside the sources", base.with("linalg/blob.dat", "x"), true},
		// The zone starts using an unreached file: the zone file changed, so yes; and
		// from then on that file is reached.
		{"the zone reaches further", base.with("ann/flat.go", "package ann\nimport \"example.com/m/linalg\"\nvar _ = linalg.Attend\n"), true},
		// A file is judged in the tree it existed in.
		{"a reached file is deleted", base.with("linalg/table.go", "", "linalg/dot.go", base["linalg/dot.go"]), true},
		{"only the toolchain line", base.with("go.mod", base["go.mod"]+"toolchain go1.27.4\n"), false},
		{"the toolchain line is added", base.with("go.mod", strings.Replace(base["go.mod"], "require", "toolchain go1.27.9\n\nrequire", 1)), false},
		{"a dependency bump", base.with("go.mod", strings.Replace(base["go.mod"], "v0.40.0", "v0.41.0", 1)), true},
		{"the go version", base.with("go.mod", strings.Replace(base["go.mod"], "go 1.27.0", "go 1.28.0", 1)), true},
		{"go.sum alone", base.with("go.sum", "y\n"), false},
	} {
		m := mark(t, base, tc.new)
		if m.Yes != tc.yes {
			t.Errorf("%s: Yes = %v, want %v\n%+v", tc.name, m.Yes, tc.yes, m.Files)
		}
		if len(m.Files) == 0 {
			t.Errorf("%s: no changed files; the case tests nothing", tc.name)
		}
	}
}

// What the tool cannot follow puts every changed path in the zone.
func TestUnanalyzableTreeIsYes(t *testing.T) {
	for name, src := range map[string]mem{
		"a parse error": base.with("vision/tower.go", "package vision\nfunc {"),
		"go:linkname":   base.with("vision/tower.go", "package vision\n//go:linkname x y\nvar x int\n"),
		"a dot import":  base.with("vision/tower.go", "package vision\nimport . \"example.com/m/linalg\"\n"),
	} {
		m := mark(t, base, src.with("README.md", "changed\n"))
		for _, f := range m.Files {
			if !f.Zone || !strings.Contains(f.Why, "unclassified") {
				t.Errorf("%s: %s is %v (%s); want unclassified, in the zone", name, f.Path, f.Zone, f.Why)
			}
		}
		if !m.Yes {
			t.Errorf("%s: Yes = false", name)
		}
	}
}

func TestLine(t *testing.T) {
	no := Mark{Files: []File{{Path: "README.md"}}}
	if got, want := no.Line("v1.0.0"), "RETRIEVAL: no — vs v1.0.0, none of 1 changed files is in the retrieval zone"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
	yes := Mark{Yes: true, Files: []File{{Path: "a", Zone: true}, {Path: "b"}}}
	if got, want := yes.Line("v1.0.0"), "RETRIEVAL: yes — vs v1.0.0, 1 of 2 changed files are in the retrieval zone (a)"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
