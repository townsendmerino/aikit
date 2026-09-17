package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const rootPath = "github.com/townsendmerino/aikit"

// published modules are consumables the proxy must serve; internal ones are consumers of
// this repo (a benchmark harness, an example binary, this tools module itself), never
// imported by anyone. THE CLASSIFICATION LIVES HERE AND ONLY HERE: a go.mod whose module
// path is in neither map is unclassified and fails the gate, which is the mechanism that
// stops a newly-added module from skipping this check unnoticed.
var published = map[string]bool{
	rootPath:                       true,
	rootPath + "/gpu":              true,
	rootPath + "/gpu/anncuda":      true,
	rootPath + "/gpu/annmetal":     true,
	rootPath + "/gpu/enccuda":      true,
	rootPath + "/gpu/encmetal":     true,
	rootPath + "/gpu/qwencuda":     true,
	rootPath + "/gpu/qwenmetal":    true,
	rootPath + "/gpu/visioncuda":   true,
	rootPath + "/gpu/visionmetal":  true,
	rootPath + "/gpu/webgpu":       true,
	rootPath + "/chunk/treesitter": true,
}

var internal = map[string]bool{
	rootPath + "/benchmarks":               true,
	rootPath + "/examples/embedded-corpus": true,
	rootPath + "/examples/gpu-ann":         true,
	rootPath + "/tools":                    true,
	rootPath + "/tools/canary/vulnfixture": true,
}

// modulePath reads the `module <path>` line of a go.mod.
func modulePath(goMod string) (string, error) {
	f, err := os.Open(goMod)
	if err != nil {
		return "", err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) >= 2 && fields[0] == "module" {
			return fields[1], nil
		}
	}
	return "", fmt.Errorf("no module line in %s", goMod)
}

// enumerate walks the tree for every go.mod and returns their module paths, skipping .git.
func enumerate(root string) ([]string, error) {
	var paths []string
	err := filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && d.Name() == ".git" {
			return filepath.SkipDir
		}
		if !d.IsDir() && d.Name() == "go.mod" {
			mp, err := modulePath(p)
			if err != nil {
				return err
			}
			paths = append(paths, mp)
		}
		return nil
	})
	return paths, err
}

// classifyTree enumerates and reconciles every module against the classification. It
// returns the published module paths (sorted, the derived set) and an error that is
// non-nil — a HARD FAILURE — if any module is unclassified or none were found. "Never an
// empty green": zero modules is a failure, not a vacuous pass.
func classifyTree(root string) (pub []string, total int, err error) {
	mods, err := enumerate(root)
	if err != nil {
		return nil, 0, fmt.Errorf("enumerating go.mod files: %w", err)
	}
	total = len(mods)
	if total == 0 {
		return nil, 0, fmt.Errorf("enumerated ZERO go.mod files; refusing an empty green")
	}
	var unclassified []string
	for _, mp := range mods {
		switch {
		case published[mp]:
			pub = append(pub, mp)
		case internal[mp]:
		default:
			unclassified = append(unclassified, mp)
		}
	}
	if len(unclassified) > 0 {
		return nil, total, fmt.Errorf("unclassified module(s): %s\n"+
			"  Every go.mod must be listed published or internal in tools/consumergate/classify.go so it cannot skip the gate",
			strings.Join(unclassified, " "))
	}
	sort.Strings(pub)
	return pub, total, nil
}

var reSubTag = regexp.MustCompile(`^(.+)/v[0-9].*$`)
var reRootTag = regexp.MustCompile(`^v[0-9].*$`)

// mapTagToPair turns a pushed release tag into the one module@version it names. A tag is
// <subpath>/vX.Y.Z, or vX.Y.Z at the root; the module path is rootPath[/<subpath>] and the
// version is the vX.Y.Z. It errors if the ref is not a recognized release tag OR maps to a
// module that is not published — a release-shaped tag the gate does not know how to cover
// is a real gap, reported, not silently skipped.
func mapTagToPair(ref string) (Pair, error) {
	ref = strings.TrimPrefix(ref, "refs/tags/")
	var mod, ver string
	switch {
	case reSubTag.MatchString(ref):
		i := strings.LastIndex(ref, "/v")
		mod = rootPath + "/" + ref[:i]
		ver = ref[i+1:]
	case reRootTag.MatchString(ref):
		mod, ver = rootPath, ref
	default:
		return Pair{}, fmt.Errorf("tag %q is not a recognized release tag "+
			"(vX.Y.Z, gpu/vX.Y.Z, gpu/<name>/vX.Y.Z, chunk/treesitter/vX.Y.Z)", ref)
	}
	if !published[mod] {
		return Pair{}, fmt.Errorf("tag %q maps to %q, which is not a published module the gate covers", ref, mod)
	}
	return Pair{Module: mod, Version: ver}, nil
}
