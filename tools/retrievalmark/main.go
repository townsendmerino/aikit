// Command retrievalmark answers, for a range of the root module's history: could this
// change retrieval results or the retrieval API? It prints every changed path with its
// verdict and reason, then one line:
//
//	RETRIEVAL: yes — vs v1.76.0, 3 of 14 changed files are in the retrieval zone (ann/flat.go, ...)
//	RETRIEVAL: no — vs v1.76.0, none of 14 changed files is in the retrieval zone
//
// That line goes in the release's CHANGELOG section, and tools/releasegate refuses a
// release without it. A consumer that uses only the retrieval packages (ken) can skip
// its parity and recall checks on a "no" release. The zone, the method and its limits
// are in tools/retrievalzone and docs/architecture.md § Retrieval zone.
//
// Usage:
//
//	go run -C tools ./retrievalmark                 # previous root tag → working tree
//	go run -C tools ./retrievalmark --from v1.75.0 --to v1.76.0
//	go run -C tools ./retrievalmark --sweep 30      # each of the last 30 root tags vs its predecessor
//	go run -C tools ./retrievalmark --files         # which linalg/embed/... files the zone reaches
//	go run -C tools ./retrievalmark --ken ../ken    # check the zone still covers what ken imports
//
// Exit status: 0 when it answered (yes or no), 2 when it could not.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/townsendmerino/aikit/tools/gpumod"
	"github.com/townsendmerino/aikit/tools/retrievalzone"
)

func main() { os.Exit(run(os.Args[1:])) }

func run(args []string) int {
	fs := flag.NewFlagSet("retrievalmark", flag.ContinueOnError)
	from := fs.String("from", "", "ref to compare from (default: the highest root tag)")
	to := fs.String("to", "", "ref to compare to (default: the working tree)")
	sweep := fs.Int("sweep", 0, "mark each of the last N root tags against its predecessor")
	files := fs.Bool("files", false, "list which files of the partly-in packages the zone reaches")
	ken := fs.String("ken", "", "path to a ken checkout: check the zone covers what ken imports")
	quiet := fs.Bool("q", false, "print only the RETRIEVAL line")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Fprintln(os.Stderr, "retrievalmark: cannot locate the repo root:", err)
		return 2
	}
	fail := func(err error) int {
		fmt.Fprintln(os.Stderr, "retrievalmark:", err)
		return 2
	}

	switch {
	case *ken != "":
		return checkKen(root, *ken)
	case *files:
		return listFiles(root, *to)
	case *sweep > 0:
		tags, err := retrievalzone.RootTags(root)
		if err != nil {
			return fail(err)
		}
		if len(tags) < *sweep+1 {
			return fail(fmt.Errorf("--sweep %d needs %d root tags, found %d", *sweep, *sweep+1, len(tags)))
		}
		no := 0
		for i := 0; i < *sweep; i++ {
			m, err := retrievalzone.Between(root, tags[i+1], tags[i])
			if err != nil {
				return fail(err)
			}
			if !m.Yes {
				no++
			}
			fmt.Printf("%-9s %s\n", tags[i], m.Line(tags[i+1]))
		}
		fmt.Printf("\n%d of the last %d root releases are RETRIEVAL: no\n", no, *sweep)
		return 0
	}

	if *from == "" {
		tags, err := retrievalzone.RootTags(root)
		if err != nil || len(tags) == 0 {
			return fail(fmt.Errorf("no root tag to compare from (pass --from): %v", err))
		}
		*from = tags[0]
	}
	m, err := retrievalzone.Between(root, *from, *to)
	if err != nil {
		return fail(err)
	}
	if !*quiet {
		for _, f := range m.Files {
			v := "no "
			if f.Zone {
				v = "YES"
			}
			fmt.Printf("%s  %-44s %s\n", v, f.Path, f.Why)
		}
		fmt.Println()
	}
	fmt.Println(m.Line(*from))
	return 0
}

func listFiles(root, ref string) int {
	var src retrievalzone.Source
	var err error
	if ref == "" {
		src, err = retrievalzone.WorkTree(root)
	} else {
		src, err = retrievalzone.AtRef(root, ref)
	}
	if err == nil {
		var a *retrievalzone.Analysis
		if a, err = retrievalzone.Analyze(src); err == nil {
			for _, dir := range a.Linked() {
				in, out := a.Reached(dir)
				fmt.Printf("%s: %d of %d files reached\n", dir, len(in), len(in)+len(out))
				for _, f := range in {
					_, why := a.Classify(f)
					fmt.Printf("  in   %-52s %s\n", f, strings.TrimPrefix(why, "reached: "))
				}
				for _, f := range out {
					fmt.Printf("  out  %s\n", f)
				}
			}
			return 0
		}
	}
	fmt.Fprintln(os.Stderr, "retrievalmark:", err)
	return 2
}
