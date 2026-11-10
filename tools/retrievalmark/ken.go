package main

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"

	"github.com/townsendmerino/aikit/tools/retrievalzone"
)

const aikitModule = "github.com/townsendmerino/aikit"

var importRe = regexp.MustCompile(`"` + regexp.QuoteMeta(aikitModule) + `/([a-z0-9/]+)"`)

// kenUses reads every Go file under dir and returns the aikit packages they import and,
// for the partly-in packages, the names they use (`embed.X`).
func kenUses(dir string) (pkgs map[string]bool, syms map[string]map[string]bool, err error) {
	pkgs, syms = map[string]bool{}, map[string]map[string]bool{}
	symRe := map[string]*regexp.Regexp{}
	for p := range retrievalzone.Symbols {
		symRe[p] = regexp.MustCompile(`\b` + filepath.Base(p) + `\.([A-Z][A-Za-z0-9_]*)`)
		syms[p] = map[string]bool{}
	}
	err = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if n := d.Name(); path != dir && (strings.HasPrefix(n, ".") || strings.HasPrefix(n, "_") || n == "vendor") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		for _, m := range importRe.FindAllSubmatch(data, -1) {
			pkgs[string(m[1])] = true
			if re := symRe[string(m[1])]; re != nil {
				for _, s := range re.FindAllSubmatch(data, -1) {
					syms[string(m[1])][string(s[1])] = true
				}
			}
		}
		return nil
	})
	return pkgs, syms, err
}

// checkKen fails if ken imports an aikit package the zone does not name, or uses a
// symbol of a partly-in package that retrievalzone.Symbols does not list. Either means
// the mark could say "no" about a change ken can see.
func checkKen(root, dir string) int {
	pkgs, syms, err := kenUses(dir)
	if err != nil || len(pkgs) == 0 {
		fmt.Fprintf(os.Stderr, "retrievalmark: no aikit imports found under %s (%v)\n", dir, err)
		return 2
	}
	src, err := retrievalzone.WorkTree(root)
	if err != nil {
		fmt.Fprintln(os.Stderr, "retrievalmark:", err)
		return 2
	}
	a, err := retrievalzone.Analyze(src)
	if err != nil {
		fmt.Fprintln(os.Stderr, "retrievalmark:", err)
		return 2
	}
	var bad []string
	names := make([]string, 0, len(pkgs))
	for p := range pkgs {
		names = append(names, p)
	}
	sort.Strings(names)
	for _, p := range names {
		_, partly := retrievalzone.Symbols[p]
		switch {
		case slices.Contains(retrievalzone.Packages, p):
			fmt.Printf("ok    %-16s zone package\n", p)
		case partly:
			var missing []string
			for s := range syms[p] {
				// A name aikit does not declare is another package's (the stdlib has an embed too).
				if a.HasSymbol(p, s) && !slices.Contains(retrievalzone.Symbols[p], s) {
					missing = append(missing, s)
				}
			}
			sort.Strings(missing)
			if len(missing) > 0 {
				bad = append(bad, fmt.Sprintf("ken uses %s.%s, not in retrievalzone.Symbols", p, strings.Join(missing, ", "+p+".")))
				fmt.Printf("STALE %-16s ken uses %s\n", p, strings.Join(missing, ", "))
			} else {
				fmt.Printf("ok    %-16s partly in: every symbol ken uses is listed\n", p)
			}
		case !a.HasPackage(p):
			fmt.Printf("ok    %-16s a nested module with its own tags, outside the root mark\n", p)
		default:
			bad = append(bad, "ken imports "+p+", which is not in the zone")
			fmt.Printf("STALE %-16s not in retrievalzone.Packages or Symbols\n", p)
		}
	}
	if len(bad) > 0 {
		fmt.Println("\nVERDICT: FAIL — the zone does not cover ken: " + strings.Join(bad, "; "))
		return 1
	}
	fmt.Println("\nVERDICT: PASS — the zone covers every aikit package and symbol ken uses")
	return 0
}
