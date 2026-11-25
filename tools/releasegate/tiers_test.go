package main

import (
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// readmeTiers is what README § "Stability tiers" names, split by its two subsections.
type readmeTiers struct {
	hardPkgs  map[string]bool // root packages any Hard bullet names
	hardNames map[string]bool // "pkg.Name" and "pkg.Type.Member" tokens in the Hard bullets
	expNames  map[string]bool // the same tokens in the Experimental bullets
}

// backtick holds one `code` span; tierToken is a span that names a root package or a symbol in one.
var (
	backtick  = regexp.MustCompile("`([^`]+)`")
	tierToken = regexp.MustCompile(`^([a-z][a-z0-9]*)((?:\.[A-Z][A-Za-z0-9_]*)*)(?:\[[A-Za-z, ]+\])?$`)
)

// rootPackages lists the module's top-level directories that hold a non-test Go file.
func rootPackages(t *testing.T, root string) map[string]bool {
	t.Helper()
	ents, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	pkgs := map[string]bool{}
	for _, e := range ents {
		if !e.IsDir() {
			continue
		}
		gos, _ := filepath.Glob(filepath.Join(root, e.Name(), "*.go"))
		for _, g := range gos {
			if !strings.HasSuffix(g, "_test.go") {
				pkgs[e.Name()] = true
				break
			}
		}
	}
	return pkgs
}

// parseTiers reads the two tier subsections of the README. A code span counts when its first element is a root package
// (so `regex` the chunker name and `vision_serve.go` the file do not).
func parseTiers(t *testing.T, readme string, pkgs map[string]bool) readmeTiers {
	t.Helper()
	_, rest, ok := strings.Cut(readme, "\n### Hard")
	if !ok {
		t.Fatal(`README has no "### Hard" subsection`)
	}
	hard, rest, ok := strings.Cut(rest, "\n### Experimental")
	if !ok {
		t.Fatal(`README has no "### Experimental" subsection after "### Hard"`)
	}
	exp, _, _ := strings.Cut(rest, "\n---")
	tr := readmeTiers{hardPkgs: map[string]bool{}, hardNames: map[string]bool{}, expNames: map[string]bool{}}
	scan := func(sec string, names map[string]bool, pkgsOut map[string]bool) {
		for ln := range strings.SplitSeq(sec, "\n") {
			for _, m := range backtick.FindAllStringSubmatch(ln, -1) {
				// `ann.HNSW.MarshalBinary` / `Flat`/`HNSW` spans and `pkg.Name{,Into}` shorthand: take the name before any brace.
				span, _, _ := strings.Cut(m[1], "{")
				tm := tierToken.FindStringSubmatch(span)
				if tm == nil || !pkgs[tm[1]] {
					continue
				}
				if pkgsOut != nil {
					pkgsOut[tm[1]] = true
				}
				if tm[2] != "" {
					names[tm[1]+tm[2]] = true
				}
			}
		}
	}
	// Only bullets carry tier entries; the subsection's intro prose names packages in passing.
	bullets := func(sec string) string {
		var b strings.Builder
		in := false
		for ln := range strings.SplitSeq(sec, "\n") {
			switch {
			case strings.HasPrefix(ln, "- "):
				in = true
			case strings.TrimSpace(ln) == "" || !strings.HasPrefix(ln, " "):
				in = false
			}
			if in {
				b.WriteString(ln + "\n")
			}
		}
		return b.String()
	}
	scan(bullets(hard), tr.hardNames, tr.hardPkgs)
	scan(bullets(exp), tr.expNames, nil)
	return tr
}

// memberName turns an experimentalMembers entry, `(*Type).Member` or `Type.Member`, into README's `Type.Member`.
func memberName(m string) string {
	m = strings.TrimPrefix(m, "(*")
	m = strings.TrimPrefix(m, "(")
	return strings.Replace(m, ").", ".", 1)
}

// TestReadmeTiers_matchGate holds README § "Stability tiers" and the gate's lists equal: the packages the Hard bullets
// name are exactly hardPkgs; no name the gate exempts is listed as Hard; and every exempted name is listed as Experimental,
// so an exemption is never undocumented.
//
// One direction is deliberately not checked: an Experimental name inside a Hard package that is not exempted. The gate
// then fails on a break the README allows, which is the safe way to be wrong.
func TestReadmeTiers_matchGate(t *testing.T) {
	root := filepath.Join("..", "..")
	b, err := os.ReadFile(filepath.Join(root, "README.md"))
	if err != nil {
		t.Fatal(err)
	}
	tr := parseTiers(t, string(b), rootPackages(t, root))

	if got, want := slices.Sorted(maps.Keys(tr.hardPkgs)), slices.Sorted(slices.Values(hardPkgs)); !slices.Equal(got, want) {
		t.Errorf("README's Hard bullets name packages %v; the gate's hardPkgs is %v", got, want)
	}
	for _, p := range hardPkgs {
		if !tr.hardPkgs[p] {
			continue // reported above
		}
		var exempt []string
		for _, s := range experimentalSyms[p] {
			exempt = append(exempt, p+"."+s)
		}
		for _, m := range experimentalMembers[p] {
			exempt = append(exempt, p+"."+memberName(m))
		}
		for _, n := range exempt {
			if tr.hardNames[n] {
				t.Errorf("%s is exempt from the gate (experimentalSyms/experimentalMembers) but README lists it as Hard", n)
			}
			if !tr.expNames[n] {
				t.Errorf("%s is exempt from the gate but README's Experimental bullets do not name it", n)
			}
		}
	}
	for p := range experimentalSyms {
		if !slices.Contains(hardPkgs, p) {
			t.Errorf("experimentalSyms has an entry for %s, which is not in hardPkgs", p)
		}
	}
	for p := range experimentalMembers {
		if !slices.Contains(hardPkgs, p) {
			t.Errorf("experimentalMembers has an entry for %s, which is not in hardPkgs", p)
		}
	}
}

// TestParseTiers_planted checks the parser on a planted README, so the real test cannot pass by reading nothing.
func TestParseTiers_planted(t *testing.T) {
	readme := "intro `ann.Flat`\n### Hard\n\nprose `bm25` here\n\n- `topk.Selector[T]`, `topk.New`\n- `mmap` — the whole package\n" +
		"- Concrete chunker names registered under `regex`\n  and `vision_serve.go`, `gpu/x.go`\n\n### Experimental\n\n" +
		"- `ann.HNSW.MarshalBinary` / `ann.Load`\n- `linalg.MatmulBTQ8Fused{,Into}`\n- `encoder.Backend` and\n  `embed.Tensor.SubF32`\n\n---\nlater `fuse.RRF`\n"
	pkgs := map[string]bool{"ann": true, "bm25": true, "topk": true, "mmap": true, "linalg": true, "encoder": true, "embed": true, "fuse": true, "vision": true}
	tr := parseTiers(t, readme, pkgs)
	if got := slices.Sorted(maps.Keys(tr.hardPkgs)); !slices.Equal(got, []string{"mmap", "topk"}) {
		t.Errorf("hard packages = %v, want [mmap topk]", got)
	}
	if got := slices.Sorted(maps.Keys(tr.hardNames)); !slices.Equal(got, []string{"topk.New", "topk.Selector"}) {
		t.Errorf("hard names = %v", got)
	}
	want := []string{"ann.HNSW.MarshalBinary", "ann.Load", "embed.Tensor.SubF32", "encoder.Backend", "linalg.MatmulBTQ8Fused"}
	if got := slices.Sorted(maps.Keys(tr.expNames)); !slices.Equal(got, want) {
		t.Errorf("experimental names = %v, want %v", got, want)
	}
	if got := memberName("(*SafetensorsFile).ReleaseTensors"); got != "SafetensorsFile.ReleaseTensors" {
		t.Errorf("memberName = %q", got)
	}
}
