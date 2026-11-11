// Command scriptsguard backs two campaign claims about scripts/ with derivable rules rather than
// hand-kept lists, so neither can silently regress:
//
//	PYTHON — every .py under scripts/ lives in exactly one declared category; a stray .py directly
//	under scripts/ (the residue the zero-Python campaign removed) fails.
//	  scripts/oracle/   — reference generators that reach external ecosystems Go has no equivalent
//	                      for: PyTorch/HF goldens, and llama.cpp's ggml grid tables. Irreducible.
//	  scripts/fixtures/ — kept on cost grounds, reducible in principle: prep_beir.py fetches an HF
//	                      parquet dataset for a manual benchmark; porting it would add a parquet
//	                      dependency for no gate. Reviewable.
//
//	SHELL — NO .sh anywhere under scripts/. Every deciding-shell gate was migrated to a Go command
//	under tools/ (the shell-to-zero campaign); scripts/ holds declared Python and nothing else. A
//	.sh reappearing under scripts/ is a deciding gate that skipped tools/ — it fails. (The gpu/
//	CUDA build toolchain — build_ptx.sh + nvrtc_compile.py, NVRTC via a C library Go has no
//	build-time binding for, shared verbatim with goinfer — is separate build infrastructure, out
//	of scope by the scripts/ boundary.)
//
// Both rules are derived by path, so nothing is hand-enumerated: adding a pin_ script to
// oracle/ needs no edit here. (This cites the same "derived by path, not hand-enumerated"
// argument as B7 once did — that tag no longer resolves to anything in the archived perf-
// campaign docs, so it's dropped rather than left pointing nowhere; see docs/architecture.md
// § Numbered-citation index.) The DENOMINATOR is printed so a pass and a scan-of-nothing differ.
// Usage: go run -C tools ./scriptsguard
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/townsendmerino/aikit/tools/gpumod"
)

func main() { os.Exit(run()) }

func run() int {
	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Println("scriptsguard: CANNOT-EVALUATE — cannot locate repo root: " + err.Error())
		return 2
	}
	scripts := filepath.Join(root, "scripts")
	var oracle, fixtures, strayPy, straySh []string
	//nolint:errcheck // the callback never returns an error, so WalkDir cannot; unreadable entries are skipped below
	_ = filepath.WalkDir(scripts, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(scripts, p) //nolint:errcheck // p is under scripts, so Rel cannot fail; e.g. "oracle/pin_bge.py"
		switch {
		case strings.HasSuffix(d.Name(), ".sh"):
			straySh = append(straySh, "scripts/"+rel)
		case strings.HasSuffix(d.Name(), ".py"):
			switch {
			case strings.HasPrefix(rel, "oracle"+string(filepath.Separator)):
				oracle = append(oracle, rel)
			case strings.HasPrefix(rel, "fixtures"+string(filepath.Separator)):
				fixtures = append(fixtures, rel)
			default:
				strayPy = append(strayPy, "scripts/"+rel)
			}
		}
		return nil
	})
	nPy := len(oracle) + len(fixtures) + len(strayPy)

	// Denominator first — a scan that found nothing must not read as a pass.
	fmt.Printf("scriptsguard: %d .py under scripts/ — %d oracle, %d fixtures, %d stray; %d .sh (want 0)\n",
		nPy, len(oracle), len(fixtures), len(strayPy), len(straySh))
	if nPy == 0 {
		fmt.Println("scriptsguard: CANNOT-EVALUATE — found no .py under scripts/ at all (did the tree move, or is scripts/ absent?)")
		return 2
	}

	stray := append(append([]string{}, strayPy...), straySh...)
	if len(stray) > 0 {
		sort.Strings(stray)
		for _, s := range stray {
			fmt.Println("  STRAY: " + s)
		}
		if len(straySh) > 0 {
			fmt.Printf("scriptsguard: FAIL — %d .sh under scripts/. Deciding-shell gates live in tools/ as Go "+
				"commands (shell-to-zero); a .sh here is a gate that skipped tools/.\n", len(straySh))
		}
		if len(strayPy) > 0 {
			fmt.Printf("scriptsguard: FAIL — %d .py under scripts/ outside scripts/{oracle,fixtures}/. Every "+
				"scripts/ Python must be an external-reference generator (oracle/) or a declared fixture "+
				"(fixtures/); if it is neither, it is residue.\n", len(strayPy))
		}
		return 1
	}
	fmt.Println("scriptsguard: OK — scripts/ holds no deciding-shell (all in tools/) and no Python except " +
		"(a) reference generators reaching external ecosystems Go has no equivalent for (PyTorch/HF goldens, " +
		"llama.cpp ggml tables) and (b) one fixture fetcher kept because porting it would add a parquet " +
		"dependency for a manual benchmark.")
	return 0
}
