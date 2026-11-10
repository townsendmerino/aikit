// Command consumergate verifies that every PUBLISHED aikit module resolves and builds from
// a CONSUMER's position, through the Go module proxy, with none of this repo's replace
// directives or dev go.work in scope. Paste its VERDICT line into a tag message.
//
// WHY THIS EXISTS, AND WHAT IT CLOSES. On 2026-08-12 the eight gpu backend modules were
// found not to resolve for consumers at all: each carries `replace` directives into the
// tree, so every in-tree build passed while an outside importer got a 404 fetching
// `require .../aikit/gpu v0.0.0` (RELEASING.md, "Backend submodules"). The defect was
// STRUCTURALLY invisible from inside the checkout — the replaces and the dev go.work mask
// it by construction — so the only honest check runs from OUTSIDE the tree: a scratch
// module, no replaces, GOWORK=off, and the public proxy as the only source.
//
// TWO TIERS, because one machine cannot compile all nine gpu modules:
//   - resolve  (platform-INDEPENDENT): `go get module@version` + `go mod download` against
//     the proxy. This tier alone would have caught the v0.0.0 404 on ANY OS, because it
//     fetches the whole require graph. It runs and is judged even when compile is n/a.
//   - compile  (platform-DEPENDENT): `go build module/...`. A platform-gated module on the
//     wrong OS (the four *metal on Linux, the four *cuda on darwin) has no packages there;
//     that is n/a — neither ok nor FAIL — as gpu_device.sh classifies it. The union of a
//     Linux run and a macOS run covers all nine.
//
// WHAT THIS GATE CANNOT DO. A Go module tag is IMMUTABLE once the proxy fetches it, so a
// red tag-triggered run cannot unpublish anything. It shortens detection of the "does not
// resolve for consumers" class from "a consumer eventually complains" to "minutes after
// push"; the fix is a follow-up tag. Prevention stays in RELEASING.md, "Fixing it, in this
// order". This command reports; it does not prevent.
//
// Usage:
//
//	go run -C tools ./consumergate                     # every published module @latest (watch)
//	go run -C tools ./consumergate MOD@VER [MOD@VER…]  # explicit pairs
//	go run -C tools ./consumergate --tag <ref>         # map one pushed tag → its module@version
//	go run -C tools ./consumergate --head              # the TREE's copy of each published nested module,
//	                                                   # replaces dropped, against the versions it requires (head.go)
//
// Env:
//
//	CONSUMER_GATE_PROXY     GOPROXY value (default https://proxy.golang.org, NO ,direct).
//	                        Set to `off` to simulate an unreachable proxy.
//	CONSUMER_GATE_ATTEMPTS  resolve attempts before a not-found goes red (default 4).
//	CONSUMER_GATE_BACKOFF   base backoff seconds between attempts (default 5; 5,10,20,…).
package main

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/townsendmerino/aikit/tools/gate"
	"github.com/townsendmerino/aikit/tools/gpumod"
)

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(args []string) int {
	goos := goEnv("GOOS")
	cfg := config{
		proxy:    envOr("CONSUMER_GATE_PROXY", "https://proxy.golang.org"),
		attempts: envInt("CONSUMER_GATE_ATTEMPTS", 4),
		backoff:  time.Duration(envInt("CONSUMER_GATE_BACKOFF", 5)) * time.Second,
		goos:     goos,
	}
	// root, prov, box, date computed before the repoRoot error check below, same as before:
	// gpumod.Provenance("") still resolves commit/dirty (git auto-discovers the repo from
	// cwd; cmd.Dir="" is the same "inherit cwd" behavior this tool always had), so the error
	// path can still report full provenance rather than degrading to bare strings.
	root, rootErr := gpumod.RepoRoot()
	p := gpumod.Provenance(root)
	prov := p.Commit + p.Dirty
	// Host, not gpumod's uname-based OSName/Arch: those report the box gpumod's OWN gpu
	// gates build ON, but this gate's whole point is testing a PLATFORM UNDER TEST that
	// varies per run (goEnv("GOOS")/("GOARCH") below), which uname can't answer.
	box := p.Host + " " + goEnv("GOOS") + "/" + goEnv("GOARCH")
	date := p.Date

	fmt.Printf("aikit consumer gate — %s — %s — proxy=%s — %s\n\n", prov, box, cfg.proxy, date)

	// Classification runs ALWAYS, even for --tag and explicit pairs: it is the integrity
	// check ("no module skips the gate") and it is what validates a tag against the
	// published set. A guard failure is a hard FAIL, distinct from a module defect.
	if rootErr != nil {
		return incon(fmt.Sprintf("cannot locate repo root: %v (at %s, %s, %s)", rootErr, prov, box, date))
	}
	pub, total, err := classifyTree(root)
	if err != nil {
		fmt.Println(gate.Verdict(gate.Fail, err.Error()))
		fmt.Printf("  (%d go.mod files enumerated at %s, %s, %s)\n", total, prov, box, date)
		return 1
	}

	if len(args) == 1 && args[0] == "--head" {
		return runHead(cfg, root, pub, prov, box, date)
	}

	pairs, code, msg := decidePairs(args, pub)
	if code >= 0 {
		return incon(fmt.Sprintf("%s (at %s, %s, %s)", msg, prov, box, date))
	}

	// The matrix is DATA: one Check per pair, run by the shared gate.RunAll so a failing
	// cell never aborts the rest and the count is never lost.
	checks := make([]gate.Check, 0, len(pairs))
	for _, p := range pairs {
		checks = append(checks, gate.Check{Name: p.String(), Run: func() gate.Cell { return checkPair(cfg, p) }})
	}
	cells := gate.RunAll(checks)

	// Per-cell scope lines, then the "which OS covered which compile cells" summary.
	var compiled, naHere []string
	for _, c := range cells {
		var rfield, cfield string
		for _, f := range c.Fields {
			switch f.Key {
			case "resolve":
				rfield = f.String()
			case "compile":
				cfield = f.String()
			}
		}
		verdict := "ok"
		if c.Outcome != gate.OK {
			verdict = string(c.Outcome)
		}
		fmt.Printf("  %-52s %-30s %-30s %s\n", c.Name, rfield, cfield, verdict)
		if c.Outcome == gate.OK {
			mod, _, _ := strings.Cut(c.Name, "@")
			for _, f := range c.Fields {
				if f.Key == "compile" && f.State == "ok" {
					compiled = append(compiled, shortLabel(mod))
				}
				if f.Key == "compile" && f.State == "n/a" {
					naHere = append(naHere, shortLabel(mod))
				}
			}
		}
	}
	fmt.Printf("\n  compiled here (GOOS=%s):%s\n", goos, listOrNone(compiled))
	fmt.Printf("  n/a here (needs the other OS):%s\n\n", listOrNone(naHere))

	rep := gate.Reconcile(cells)
	switch rep.Outcome {
	case gate.Inconclusive:
		fmt.Println(gate.Verdict(gate.Inconclusive, fmt.Sprintf(
			"%d/%d module(s) could not be judged (proxy unreachable) at %s (%s, GOOS=%s, %s)",
			rep.Incon, rep.Total, prov, box, goos, date)))
	case gate.Fail:
		fmt.Println(gate.Verdict(gate.Fail, fmt.Sprintf(
			"%d/%d published module(s) do not resolve/compile for consumers at %s (%s, GOOS=%s, %s)",
			rep.Fail, rep.Total, prov, box, goos, date)))
	default:
		fmt.Println(gate.Verdict(gate.OK, fmt.Sprintf(
			"%d/%d published module(s) resolve for consumers at %s (%s, GOOS=%s, %s)",
			rep.Pass, rep.Total, prov, box, goos, date)))
		fmt.Printf("         compile verified on GOOS=%s; wrong-OS modules reported n/a — the other runner covers those.\n", goos)
		fmt.Println("Paste that line into the tag message. It is one OS's half; a full picture needs the other runner's line too.")
	}
	return rep.Exit
}

// decidePairs turns the CLI args into the pairs to check. It returns (pairs, -1, "") on
// success, or (nil, exitCode, message) for an INCONCLUSIVE precondition — an unmappable
// tag, a malformed argument — that must be reported, never silently skipped.
func decidePairs(args []string, pub []string) ([]Pair, int, string) {
	switch {
	case len(args) == 0:
		var pairs []Pair
		for _, m := range pub {
			pairs = append(pairs, Pair{Module: m, Version: "latest"})
		}
		return pairs, -1, ""
	case args[0] == "--tag":
		if len(args) != 2 {
			return nil, 2, "--tag needs exactly one ref"
		}
		p, err := mapTagToPair(args[1])
		if err != nil {
			return nil, 2, "tag maps to no known published module; the gate cannot cover it: " + err.Error()
		}
		return []Pair{p}, -1, ""
	default:
		var pairs []Pair
		for _, a := range args {
			if !strings.Contains(a, "@") {
				return nil, 2, fmt.Sprintf("argument %q is not MODULE@VERSION", a)
			}
			i := strings.LastIndex(a, "@")
			pairs = append(pairs, Pair{Module: a[:i], Version: a[i+1:]})
		}
		return pairs, -1, ""
	}
}

func incon(msg string) int {
	fmt.Println(gate.Verdict(gate.Inconclusive, msg))
	return 2
}

func listOrNone(xs []string) string {
	if len(xs) == 0 {
		return " none"
	}
	return " " + strings.Join(xs, " ")
}

// --- small environment helpers -----------------------------------------------------

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func goEnv(k string) string {
	out, err := exec.Command("go", "env", k).Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}
