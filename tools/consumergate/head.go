package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/townsendmerino/aikit/tools/gate"
)

// --head: THE TREE'S COPY of every published nested module, checked against the versions
// its go.mod requires.
//
// The other modes ask "does the TAG resolve and build for a consumer". This one asks the
// question a tag cannot: "is the code on main still true to its own go.mod". Most nested
// modules commit `replace` directives into the tree, so every in-tree build, lint and
// test runs them against the tree's aikit, whatever version their go.mod names. A
// consumer's build ignores a dependency's replaces and gets the named version. A module
// whose code has moved past its require therefore builds everywhere in this repo and
// nowhere outside it, and the first sign is a tag that does not compile.
//
// So each published module is copied out of the checkout, its replace directives are
// dropped, and it is vetted with GOWORK=off against the proxy: the named versions and
// nothing else. Modules outside gpu/ are tested too; the gpu modules' tests need a
// device and stay hand-run (RELEASING.md), so for them this leg is a type-check of the
// code and its tests.
//
// A module with in-repo requires and NO replace (chunk/treesitter) has the opposite
// gap: it is only ever built against its minimum. It gets a second leg against the tree.
//
// Retired modules (go.mod carries a `// Deprecated:` comment) get no further tags, so
// there is no future tag to protect: reported n/a.

// modFile is the part of `go mod edit -json` this mode reads.
type modFile struct {
	Module  struct{ Path string }
	Require []struct{ Path, Version string }
	Replace []struct{ Old struct{ Path string } }
}

// moduleDirs maps every module path in the tree to its directory relative to root.
func moduleDirs(root string) (map[string]string, error) {
	dirs := map[string]string{}
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
			rel, err := filepath.Rel(root, filepath.Dir(p))
			if err != nil {
				return err
			}
			dirs[mp] = filepath.ToSlash(rel)
		}
		return nil
	})
	return dirs, err
}

// copyModule copies the tracked files of the module at dir into dst, leaving out any
// nested module below it (gpu/ holds eight).
func copyModule(root, dir, dst string, dirs map[string]string) error {
	cmd := exec.Command("git", "ls-files", "-z", "--", dir)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("git ls-files %s: %w", dir, err)
	}
	n := 0
	for _, f := range strings.Split(string(out), "\x00") {
		if f == "" || inDeeperModule(f, dir, dirs) {
			continue
		}
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(f)))
		if err != nil {
			if os.IsNotExist(err) {
				continue // deleted in the working tree
			}
			return err
		}
		to := filepath.Join(dst, filepath.FromSlash(strings.TrimPrefix(f, dir+"/")))
		if err := os.MkdirAll(filepath.Dir(to), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(to, data, 0o644); err != nil {
			return err
		}
		n++
	}
	if n == 0 {
		return fmt.Errorf("no tracked files under %s", dir)
	}
	return nil
}

func inDeeperModule(file, dir string, dirs map[string]string) bool {
	for _, d := range dirs {
		if d != dir && strings.HasPrefix(d, dir+"/") && strings.HasPrefix(file, d+"/") {
			return true
		}
	}
	return false
}

func readMod(cfg config, dir string) (modFile, error) {
	var mf modFile
	out, err := goRun(cfg, dir, "go", "mod", "edit", "-json")
	if err != nil {
		return mf, fmt.Errorf("go mod edit -json: %s", firstLine(out))
	}
	return mf, json.Unmarshal([]byte(out), &mf)
}

// inRepo lists a go.mod's requires on other modules of this repository, as "path@version".
func inRepo(mf modFile) (paths []string, label string) {
	var parts []string
	for _, r := range mf.Require {
		if r.Path == rootPath || strings.HasPrefix(r.Path, rootPath+"/") {
			paths = append(paths, r.Path)
			parts = append(parts, strings.TrimPrefix(shortLabel(r.Path), "aikit/")+"@"+r.Version)
		}
	}
	return paths, strings.Join(parts, ",")
}

// headTestable reports whether a module's tests run without a device.
func headTestable(dir string) bool { return dir != "gpu" && !strings.HasPrefix(dir, "gpu/") }

// vetAndTest runs the leg in dir and returns one Field per step plus the leg's outcome.
func vetAndTest(cfg config, dir, prefix string, test bool) ([]gate.Field, gate.Outcome) {
	env := []string{"GOFLAGS=-mod=mod", "CGO_ENABLED=0"}
	judge := func(out string) gate.Outcome {
		if reUnreachable.MatchString(out) {
			return gate.Inconclusive
		}
		return gate.Fail
	}
	if out, err := goRunEnv(cfg, dir, env, "go", "vet", "./..."); err != nil {
		if reNoPackages.MatchString(out) {
			return []gate.Field{{Key: prefix + "vet", State: "n/a", Detail: "no packages on " + cfg.goos}}, gate.NA
		}
		o := judge(out)
		return []gate.Field{{Key: prefix + "vet", State: string(o), Detail: buildError(out)}}, o
	}
	fields := []gate.Field{{Key: prefix + "vet", State: "ok"}}
	if !test {
		return append(fields, gate.Field{Key: prefix + "test", State: "n/a", Detail: "device tests are hand-run"}), gate.OK
	}
	if out, err := goRunEnv(cfg, dir, env, "go", "test", "./..."); err != nil {
		o := judge(out)
		return append(fields, gate.Field{Key: prefix + "test", State: string(o), Detail: buildError(out)}), o
	}
	return append(fields, gate.Field{Key: prefix + "test", State: "ok"}), gate.OK
}

// checkHead is one module's cell.
func checkHead(cfg config, root, mod string, dirs map[string]string) gate.Cell {
	cell := gate.Cell{Name: shortLabel(mod)}
	incon := func(msg string) gate.Cell {
		cell.Outcome = gate.Inconclusive
		cell.Fields = []gate.Field{{Key: "setup", State: "INCONCLUSIVE", Detail: msg}}
		return cell
	}
	dir := dirs[mod]
	gomod, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(dir), "go.mod"))
	if err != nil {
		return incon(err.Error())
	}
	if strings.Contains(string(gomod), "// Deprecated:") {
		cell.Outcome = gate.NA
		cell.Fields = []gate.Field{{Key: "retired", State: "n/a", Detail: "no further tags"}}
		return cell
	}
	tmp, err := os.MkdirTemp("", "consumergate-head")
	if err != nil {
		return incon(err.Error())
	}
	defer os.RemoveAll(tmp) //nolint:errcheck // a scratch copy under the OS temp dir: nothing to report
	if err := copyModule(root, dir, tmp, dirs); err != nil {
		return incon(err.Error())
	}
	mf, err := readMod(cfg, tmp)
	if err != nil {
		return incon(err.Error())
	}
	reqs, label := inRepo(mf)
	if label == "" {
		label = "none in-repo"
	}
	cell.Fields = []gate.Field{{Key: "requires", State: label}}
	for _, r := range mf.Replace {
		if out, err := goRun(cfg, tmp, "go", "mod", "edit", "-dropreplace="+r.Old.Path); err != nil {
			return incon("go mod edit -dropreplace: " + firstLine(out))
		}
	}
	fields, outcome := vetAndTest(cfg, tmp, "", headTestable(dir))
	cell.Fields = append(cell.Fields, fields...)
	cell.Outcome = outcome
	if outcome != gate.OK || len(reqs) == 0 || len(mf.Replace) > 0 {
		return cell
	}
	// No replace of its own: nothing else builds it against the tree, so do it here.
	sort.Strings(reqs)
	for _, r := range reqs {
		target := filepath.Join(root, filepath.FromSlash(dirs[r]))
		if r == rootPath {
			target = root
		}
		if out, err := goRun(cfg, tmp, "go", "mod", "edit", "-replace="+r+"="+target); err != nil {
			return incon("go mod edit -replace: " + firstLine(out))
		}
	}
	fields, outcome = vetAndTest(cfg, tmp, "tree-", headTestable(dir))
	cell.Fields = append(cell.Fields, fields...)
	cell.Outcome = outcome
	return cell
}

// runHead is the --head mode.
func runHead(cfg config, root string, pub []string, prov, box, date string) int {
	dirs, err := moduleDirs(root)
	if err != nil {
		return incon("enumerating modules: " + err.Error())
	}
	var checks []gate.Check
	for _, m := range pub {
		if m == rootPath {
			continue // the root has no in-repo requires; its own CI jobs are this check
		}
		checks = append(checks, gate.Check{Name: shortLabel(m), Run: func() gate.Cell { return checkHead(cfg, root, m, dirs) }})
	}
	cells := gate.RunAll(checks)
	for _, c := range cells {
		var fs []string
		for _, f := range c.Fields {
			fs = append(fs, f.String())
		}
		verdict := "ok"
		if c.Outcome != gate.OK {
			verdict = string(c.Outcome)
		}
		fmt.Printf("  %-22s %-12s %s\n", c.Name, verdict, strings.Join(fs, "  "))
	}
	fmt.Println()
	rep := gate.Reconcile(cells)
	applic := rep.Total - rep.NA
	where := fmt.Sprintf("at %s (%s, GOOS=%s, %s)", prov, box, cfg.goos, date)
	switch rep.Outcome {
	case gate.Inconclusive:
		fmt.Println(gate.Verdict(gate.Inconclusive, fmt.Sprintf("%d/%d module(s) could not be judged (proxy unreachable) %s", rep.Incon, applic, where)))
	case gate.Fail:
		fmt.Println(gate.Verdict(gate.Fail, fmt.Sprintf("%d/%d published nested module(s) on this tree do not hold against the versions their go.mod requires %s", rep.Fail, applic, where)))
	default:
		if applic == 0 {
			fmt.Println(gate.Verdict(gate.Inconclusive, "no module was applicable here; an empty run is not a pass "+where))
			return 2
		}
		fmt.Println(gate.Verdict(gate.OK, fmt.Sprintf("%d/%d applicable published nested module(s) hold against the versions their go.mod requires, no replace, GOWORK=off %s", rep.Pass, applic, where)))
		fmt.Printf("         %d n/a here (retired, or needs the other OS); the other runner covers the platform-gated ones.\n", rep.NA)
	}
	return rep.Exit
}
