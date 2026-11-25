// Command perfgate is the performance gate aikit did not have when v1.17.0 shipped a W8A8
// kernel that was numerically identical, passed every correctness test, and cost a downstream
// consumer ~3% of decode throughput at a streamed shape. Two things were missing: a benchmark
// that sampled the STREAMED regime (the one that existed sampled only cache-resident B, where
// the regression looked like a +30% win), and something that RAN the benchmarks and compared
// them. This is the second half; BenchmarkW8A8SpanShapes is the first.
//
// WHY NOT A STORED BASELINE. The obvious design — record ns/op once, compare a later run
// against it — measures the machine, not the change. Session-level drift on the reference box
// was ~5% between two runs of the same binary: larger than the 3% regression this gate exists
// to catch. A number compared across days or machines is dominated by that drift in whichever
// direction it happened to fall.
//
// THE METHOD (goinfer docs/measurements/aikit-v1.17.1-*-ab.md, applied here to an aikit
// instrument instead of a goinfer one):
//   - INTERLEAVED A/B in ONE session on ONE box: the working tree and the previous tag are
//     built into two test binaries and run pre/post/pre/post, so drift hits both arms equally.
//   - The FLOOR is DERIVED FIRST, from a characterization pass (the current binary measured
//     against itself, same interleaving), and FIXED before the first comparison sample — so a
//     marginal result cannot bend the floor toward the answer. It is per-shape, k·σ of the
//     instrument's own run-to-run noise on THIS machine: a quiet box gets a tight floor and
//     catches the 3% class; a noisy CI runner gets a loose floor and honestly catches only
//     gross regressions, rather than pretending to a precision it does not have.
//   - WARM-UP DISCARD: the first visit of each arm is dropped (cold caches / first-touch).
//   - STATISTIC: the median of the retained visits per arm, per shape.
//   - THREE FIXED BRANCHES per shape: within ±floor → flat (ok); slower by >floor → REGRESSION
//     (FAIL); faster by >floor → a win, reported and scoped, never a pass masking a sibling
//     regression. Shapes present only in the working tree (a benchmark added since the tag)
//     have no baseline and are reported as such, not judged.
//
// It is the release-ritual half of the perf gate — run on a real box before a tag, like
// gpudevice — because a shared CI runner cannot measure 3%. CI runs the benchmarks in a
// smoke mode (execute once, catch a panic or OOM) via the perf-smoke job; the A/B verdict
// belongs to a box whose noise floor is tight enough to mean something.
//
// Usage:
//
//	go run -C tools ./perfgate [previous-tag]      # default: the latest vX.Y.Z tag
//	go run -C tools ./perfgate --visits 8 --benchtime 300ms v1.17.1
//	go run -C tools ./perfgate --allow-dirty       # measure uncommitted kernel work; verdict is marked, not release evidence
//	go run -C tools ./perfgate --bench BenchmarkQ56KvsInt8 --pkg ./linalg   # one family, one package
//
// The families judged and the packages searched for them are defaultFamilies and
// defaultPackages. A family is matched across the two arms by its benchmark name, whichever
// package holds it; a family listed for this arch that no package of the working tree defines
// fails the gate before anything is measured (exit 6).
//
// Its "`perfgate` VERDICT: …" line goes into the release's CHANGELOG section as printed.
// A tree dirty only in CHANGELOG.md is measured as release evidence; see dirtOutsideChangelog.
package main

import (
	"fmt"
	"maps"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"github.com/townsendmerino/aikit/tools/gate"
	"github.com/townsendmerino/aikit/tools/gpumod"
)

type config struct {
	prevTag     string
	pkgs        []string // the packages built in each arm (defaultPackages)
	families    []family // the benchmarks judged (defaultFamilies)
	benchtime   string
	visits      int
	floorK      float64
	minFloor    float64
	targetClass float64 // the regression magnitude this gate exists to catch (percent)

	allowDirty   bool // --allow-dirty: measure a tree with uncommitted changes outside CHANGELOG.md
	dirtyOutside bool // such changes are present; the verdict line is marked notEvidence
}

func main() { os.Exit(run(os.Args[1:])) }

// verdictPrefix opens perfgate's result line. tools/releasegate looks for exactly this
// at the start of a line in the release's CHANGELOG section, so the line this tool
// prints can be pasted there as it is.
const verdictPrefix = "`perfgate` "

// notEvidence marks a verdict measured on a tree with uncommitted changes other than
// CHANGELOG.md (--allow-dirty). releasegate refuses a line that carries it.
const notEvidence = " — DIRTY TREE, NOT RELEASE EVIDENCE"

func verdict(cfg config, o gate.Outcome, msg string) string {
	line := verdictPrefix + gate.Verdict(o, msg)
	if cfg.dirtyOutside {
		line += notEvidence
	}
	return line
}

// dirtOutsideChangelog lists what `git status --porcelain` reports, leaving out
// CHANGELOG.md and paths the go tool never builds (an element starting with "." or "_").
//
// THE RULE FOR A DIRTY TREE. RELEASING.md has the release's CHANGELOG section written
// before perfgate runs, so the tree is dirty by design at that step, and CHANGELOG.md
// cannot change a benchmark: that case is measured and is release evidence. Any other
// uncommitted change means the verdict describes a tree no commit names, so perfgate
// stops before measuring. --allow-dirty measures it anyway (the A/B use while developing
// a kernel) and marks the verdict line so it cannot pass for release evidence.
func dirtOutsideChangelog(porcelain string) []string {
	var out []string
	for ln := range strings.SplitSeq(porcelain, "\n") {
		if len(ln) < 4 {
			continue
		}
		path := strings.TrimSpace(ln[3:])
		if i := strings.Index(path, " -> "); i >= 0 {
			path = path[i+4:]
		}
		path = strings.Trim(path, `"`)
		if path == "CHANGELOG.md" || ignoredByGo(path) {
			continue
		}
		out = append(out, path)
	}
	return out
}

func ignoredByGo(path string) bool {
	for el := range strings.SplitSeq(path, "/") {
		if strings.HasPrefix(el, ".") || strings.HasPrefix(el, "_") {
			return true
		}
	}
	return false
}

func run(args []string) int {
	cfg := config{pkgs: defaultPackages, families: defaultFamilies, benchtime: "200ms", visits: 6, floorK: 3.0, minFloor: 2.0, targetClass: 5.0}
	var pos []string
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--visits":
			i++
			n, err := strconv.Atoi(args[i])
			if err != nil || n < 1 {
				fmt.Printf("perfgate: INCONCLUSIVE — --visits wants a positive integer, got %q\n", args[i])
				return 2
			}
			cfg.visits = n
		case "--allow-dirty":
			cfg.allowDirty = true
		case "--benchtime":
			i++
			cfg.benchtime = args[i]
		case "--bench":
			i++
			fams, err := parseFamilies(args[i])
			if err != nil {
				fmt.Printf("perfgate: INCONCLUSIVE — --bench wants benchmark names separated by commas: %v\n", err)
				return 2
			}
			cfg.families = fams
		case "--pkg":
			i++
			cfg.pkgs = nil
			for p := range strings.SplitSeq(args[i], ",") {
				if p = strings.TrimSpace(p); p != "" {
					cfg.pkgs = append(cfg.pkgs, p)
				}
			}
			if len(cfg.pkgs) == 0 {
				fmt.Printf("perfgate: INCONCLUSIVE — --pkg wants package paths separated by commas, got %q\n", args[i])
				return 2
			}
		case "--target":
			i++
			f, err := strconv.ParseFloat(args[i], 64)
			if err != nil || f <= 0 {
				fmt.Printf("perfgate: INCONCLUSIVE — --target wants a positive percentage, got %q\n", args[i])
				return 2
			}
			cfg.targetClass = f
		default:
			pos = append(pos, args[i])
		}
	}

	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Println("perfgate: INCONCLUSIVE — cannot locate repo root: " + err.Error())
		return 2
	}
	cfg.prevTag = defaultTag(root, pos)
	if cfg.prevTag == "" {
		fmt.Println("perfgate: INCONCLUSIVE — no previous vX.Y.Z tag to compare against")
		return 2
	}

	p := gpumod.Provenance(root)
	porcelain, err := gitIn(root, "status", "--porcelain")
	if err != nil {
		fmt.Println(verdict(cfg, gate.Inconclusive, "cannot read git status: "+err.Error()))
		return 2
	}
	if dirt := dirtOutsideChangelog(porcelain); len(dirt) > 0 {
		show := dirt
		if len(show) > 5 {
			show = append(append([]string{}, show[:5]...), fmt.Sprintf("and %d more", len(dirt)-5))
		}
		if !cfg.allowDirty {
			fmt.Println(verdict(cfg, gate.Inconclusive, "the tree has uncommitted changes outside CHANGELOG.md ("+strings.Join(show, ", ")+
				"), so a verdict would describe no commit. Commit or stash them, or pass --allow-dirty to measure anyway (not release evidence)"))
			return 2
		}
		cfg.dirtyOutside = true
		fmt.Println("perfgate: --allow-dirty: measuring a tree with uncommitted changes outside CHANGELOG.md (" + strings.Join(show, ", ") + "); the verdict is marked and is not release evidence")
	} else if p.Dirty != "" {
		fmt.Println("perfgate: the tree is dirty only in CHANGELOG.md or in paths the go tool ignores (the release section is written before this step): measured as release evidence")
	}
	fmt.Printf("aikit perf gate — %s%s vs %s — %s/%s — %s\n", p.Commit, p.Dirty, cfg.prevTag, p.OSName, p.Arch, p.Date)
	wanted := familyNames(cfg.families, runtime.GOARCH)
	fmt.Printf("instrument: %s, %d families   visits: %d   benchtime: %s   floor: max(%.1f%%, %.1f·σ)   targets: %.1f%% regressions\n\n",
		strings.Join(cfg.pkgs, ","), len(wanted), cfg.visits, cfg.benchtime, cfg.minFloor, cfg.floorK, cfg.targetClass)

	// Build both arms: the working tree (uncommitted changes included) and the previous tag in a
	// detached worktree. Each arm finds every family by name among its own packages.
	cur, cleanCur, err := buildArm(root, cfg.pkgs, "cur", wanted, false)
	defer cleanCur()
	if err != nil {
		fmt.Println(verdict(cfg, gate.Inconclusive, "the working tree: "+err.Error()))
		return 2
	}
	if line, rc := missingVerdict(cfg, cur.res.missing, runtime.GOARCH); rc != 0 {
		fmt.Println(line)
		return rc
	}

	wt, cleanup, err := worktreeAt(root, cfg.prevTag)
	if err != nil {
		fmt.Println("perfgate: INCONCLUSIVE — could not check out " + cfg.prevTag + ": " + err.Error())
		return 2
	}
	defer cleanup()
	prev, cleanPrev, err := buildArm(wt, cfg.pkgs, "prev", wanted, true)
	defer cleanPrev()
	if err != nil {
		fmt.Println(verdict(cfg, gate.Inconclusive, cfg.prevTag+": "+err.Error()))
		return 2
	}
	if len(prev.res.missing) > 0 {
		fmt.Printf("not in %s (judged from the next release on): %s\n\n", cfg.prevTag, strings.Join(prev.res.missing, ", "))
	}

	// CHARACTERIZATION — the current binary against itself, interleaved exactly as the
	// comparison will be, to derive the per-shape floor BEFORE any comparison sample exists.
	fmt.Println("characterizing instrument noise (current vs current)…")
	charA := make([]map[string]float64, 0, cfg.visits)
	charB := make([]map[string]float64, 0, cfg.visits)
	for v := 0; v < cfg.visits; v++ {
		a, err := runVisit(cur, cfg)
		if err != nil {
			fmt.Println(verdict(cfg, gate.Inconclusive, "characterization: "+err.Error()))
			return 4
		}
		b, err := runVisit(cur, cfg)
		if err != nil {
			fmt.Println(verdict(cfg, gate.Inconclusive, "characterization: "+err.Error()))
			return 4
		}
		charA, charB = append(charA, a), append(charB, b)
	}
	floors := deriveFloors(charA, charB, cfg)

	// COMPARISON — working tree vs previous tag, interleaved.
	fmt.Println("measuring working tree vs " + cfg.prevTag + " (interleaved)…")
	fmt.Println()
	curV := make([]map[string]float64, 0, cfg.visits)
	prevV := make([]map[string]float64, 0, cfg.visits)
	// ABBA, not ABAB (audit G-05). With a fixed A-then-B order every visit, a
	// monotone drift over the run — the machine warming, a background task
	// ramping — lands entirely on one arm and reads as a real difference. That
	// is the single bias the interleave exists to remove. Alternating the order
	// each visit puts half of any drift on each arm.
	for v := 0; v < cfg.visits; v++ {
		first, second := cur, prev
		if v%2 == 1 {
			first, second = prev, cur
		}
		r1, err := runVisit(first, cfg)
		if err != nil {
			fmt.Println(verdict(cfg, gate.Inconclusive, "comparison: "+err.Error()))
			return 4
		}
		r2, err := runVisit(second, cfg)
		if err != nil {
			fmt.Println(verdict(cfg, gate.Inconclusive, "comparison: "+err.Error()))
			return 4
		}
		if v%2 == 1 {
			r1, r2 = r2, r1
		}
		curV, prevV = append(curV, r1), append(prevV, r2)
	}

	shapes := sharedShapes(curV, prevV)
	onlyCur := onlyIn(curV, prevV)

	checks := make([]gate.Check, 0, len(shapes))
	for _, sh := range shapes {
		checks = append(checks, gate.Check{Name: sh, Run: func() gate.Cell { return judge(sh, curV, prevV, floors[sh], cfg) }})
	}
	cells := gate.RunAll(checks)

	for _, c := range cells {
		fmt.Printf("  %-40s %s\n", c.Name, verdictLine(c))
	}
	for _, sh := range onlyCur {
		fmt.Printf("  %-40s new — no baseline in %s (not judged)\n", sh, cfg.prevTag)
	}
	for _, name := range silentFamilies(wanted, curV) {
		fmt.Printf("  %-40s produced no rows on this host (the benchmark skipped; not judged)\n", name)
	}
	rep := gate.ReconcileWith(cells, gate.FailWins)

	// SENSITIVITY. A green here is "no regression above each shape's floor", not "no
	// regression". State how many shapes actually resolve the target class, and name the ones
	// that do not — on those, a green is only evidence against a LARGER regression.
	below, blind := 0, []string{}
	for _, sh := range shapes {
		if floors[sh] <= cfg.targetClass {
			below++
		} else {
			blind = append(blind, fmt.Sprintf("%s(±%.1f%%)", shortShape(sh), floors[sh]))
		}
	}
	fmt.Println()
	fmt.Printf("sensitivity: %d/%d shapes have a floor ≤ %.1f%% (the class this gate targets)\n", below, len(shapes), cfg.targetClass)
	if len(blind) > 0 {
		fmt.Printf("  BLIND to the %.1f%% class on %d shape(s): %s — a green there is only evidence against a larger regression.\n",
			cfg.targetClass, len(blind), strings.Join(blind, " "))
	}

	fmt.Println()
	switch rep.Outcome {
	case gate.Fail:
		// CONFIRMATION ROUND (audit G-03). The floor is derived per run from this
		// machine's own noise, so a single FAIL is not reproducible — the recorded
		// v1.37.0 floors at one shape span 4.39% -> 18.96% -> 11.34% across runs,
		// and "re-run until green" was therefore structurally available, which is
		// what that release did.
		//
		// Re-measuring and requiring the SAME shape to fail twice takes that away
		// symmetrically: it costs nothing on a green run, and a regression that
		// only appears once is reported as INCONCLUSIVE rather than silently
		// dropped by a human re-run. It does not make the floor reproducible —
		// persisting per-(box, shape) floors would — but it removes the asymmetry
		// that let one noisy sample decide a release either way.
		failed := map[string]bool{}
		for _, c := range cells {
			if c.Outcome == gate.Fail {
				failed[c.Name] = true
			}
		}
		fmt.Printf("confirming %d regression(s) with a second measurement…\n", len(failed))
		curV2 := make([]map[string]float64, 0, cfg.visits)
		prevV2 := make([]map[string]float64, 0, cfg.visits)
		for v := 0; v < cfg.visits; v++ {
			first, second := cur, prev
			if v%2 == 1 {
				first, second = prev, cur
			}
			r1, err := runVisit(first, cfg)
			if err != nil {
				fmt.Println(verdict(cfg, gate.Inconclusive, "confirmation: "+err.Error()))
				return 4
			}
			r2, err := runVisit(second, cfg)
			if err != nil {
				fmt.Println(verdict(cfg, gate.Inconclusive, "confirmation: "+err.Error()))
				return 4
			}
			if v%2 == 1 {
				r1, r2 = r2, r1
			}
			curV2, prevV2 = append(curV2, r1), append(prevV2, r2)
		}
		confirmed, transient := []string{}, []string{}
		for sh := range failed {
			if judge(sh, curV2, prevV2, floors[sh], cfg).Outcome == gate.Fail {
				confirmed = append(confirmed, shortShape(sh))
			} else {
				transient = append(transient, shortShape(sh))
			}
		}
		sort.Strings(confirmed)
		sort.Strings(transient)
		if len(transient) > 0 {
			fmt.Printf("  did NOT reproduce: %s\n", strings.Join(transient, " "))
		}
		if len(confirmed) == 0 {
			fmt.Println(verdict(cfg, gate.Inconclusive, fmt.Sprintf(
				"%d regression(s) did not reproduce on a second measurement — not green, not a fail; re-run or investigate",
				len(transient))))
			return 5
		}
		fmt.Printf("  reproduced: %s\n", strings.Join(confirmed, " "))
		fmt.Println(verdict(cfg, gate.Fail, fmt.Sprintf("%d regression(s) vs %s across %d shapes, confirmed on a second measurement",
			len(confirmed), cfg.prevTag, rep.Total)))
		return 1
	case gate.Inconclusive:
		fmt.Println(verdict(cfg, gate.Inconclusive, "a shape could not be measured"))
		return 2
	}
	// A gate that measured nothing, or that resolved the class it targets on
	// NOTHING, must not exit 0 (audit G-02, G-06). Both used to print a green.
	// Distinct codes so a caller can tell "this found no regression" from "this
	// could not have found one".
	if rep.Total == 0 {
		fmt.Println(verdict(cfg, gate.Inconclusive, "no shapes were measured — the gate did not run"))
		return 4
	}
	if below == 0 {
		fmt.Println(verdict(cfg, gate.Inconclusive, fmt.Sprintf(
			"BLIND — 0/%d shapes resolve the %.1f%% class, so this run is not evidence against a %.1f%% regression",
			rep.Total, cfg.targetClass, cfg.targetClass)))
		return 3
	}
	fmt.Println(verdict(cfg, gate.OK, fmt.Sprintf("no regression vs %s above each shape's floor — %d/%d shapes resolve the %.1f%% class",
		cfg.prevTag, below, rep.Total, cfg.targetClass)))
	return 0
}

// missingVerdict is the gate's result when listed families resolve to no benchmark in the
// working tree: a FAIL (exit 6), since a release would otherwise be green on kernels it never
// compared. rc is 0 when nothing is missing.
func missingVerdict(cfg config, missing []string, goarch string) (line string, rc int) {
	if len(missing) == 0 {
		return "", 0
	}
	return verdict(cfg, gate.Fail, fmt.Sprintf(
		"%d listed famil(ies) resolve to no benchmark in the working tree on %s (%s), so the gate would not judge them: restore the benchmark, or fix the family table",
		len(missing), goarch, strings.Join(missing, ", "))), 6
}

// shortShape trims the family prefix for the blind-list, keeping the K/N that identifies it.
func shortShape(s string) string {
	if i := strings.LastIndex(s, "/"); i >= 0 {
		return s[i+1:]
	}
	return s
}

// judge applies the three fixed branches to one shape.
func judge(shape string, curV, prevV []map[string]float64, floor float64, cfg config) gate.Cell {
	cur := retained(seriesFor(curV, shape))
	prev := retained(seriesFor(prevV, shape))
	if len(cur) == 0 || len(prev) == 0 {
		return gate.Cell{Name: shape, Outcome: gate.Inconclusive, Fields: []gate.Field{{Key: "measure", State: "n/a"}}}
	}
	mc, mp := median(cur), median(prev)
	delta := (mc - mp) / mp * 100 // + = current slower
	// covers: whether this shape's floor is tight enough to catch the target regression class.
	// A green on a shape whose floor is ABOVE the target only means "no regression bigger than
	// this floor" — a much weaker statement than "no regression", and it must say so.
	covers := "5%✓"
	if floor > cfg.targetClass {
		covers = fmt.Sprintf(">%.0f%% BLIND", cfg.targetClass)
	}
	f := []gate.Field{
		{Key: "cur", State: fmt.Sprintf("%.3gms", mc/1e6)},
		{Key: "prev", State: fmt.Sprintf("%.3gms", mp/1e6)},
		{Key: "Δ", State: fmt.Sprintf("%+.2f%%", delta)},
		{Key: "floor", State: fmt.Sprintf("±%.2f%%", floor)},
		{Key: "covers", State: covers},
	}
	switch {
	case delta > floor:
		return gate.Cell{Name: shape, Outcome: gate.Fail, Fields: append(f, gate.Field{Key: "branch", State: "REGRESSION"})}
	case delta < -floor:
		return gate.Cell{Name: shape, Outcome: gate.OK, Fields: append(f, gate.Field{Key: "branch", State: "faster (scoped, not published)"})}
	default:
		return gate.Cell{Name: shape, Outcome: gate.OK, Fields: append(f, gate.Field{Key: "branch", State: "flat"})}
	}
}

func verdictLine(c gate.Cell) string {
	parts := make([]string, 0, len(c.Fields))
	for _, f := range c.Fields {
		parts = append(parts, f.String())
	}
	return strings.Join(parts, "  ")
}

// deriveFloors computes the per-shape floor: k·σ of the instrument's own run-to-run relative
// noise (both characterization arms pooled, warm-up discarded), never below minFloor.
func deriveFloors(a, b []map[string]float64, cfg config) map[string]float64 {
	floors := map[string]float64{}
	for _, sh := range sharedShapes(a, b) {
		pooled := append(retained(seriesFor(a, sh)), retained(seriesFor(b, sh))...)
		if len(pooled) < 2 {
			floors[sh] = cfg.minFloor
			continue
		}
		m := mean(pooled)
		relSD := 0.0
		if m > 0 {
			relSD = stddev(pooled) / m * 100
		}
		floors[sh] = maxF(cfg.minFloor, cfg.floorK*relSD)
	}
	return floors
}

// runVisit runs each of an arm's benchmark binaries once and merges their rows.
//
// It returns an error when a binary FAILS or when the arm produced no parseable benchmark lines
// (audit G-06): a gate whose instrument did not run must not report on the thing it was meant to
// measure.
func runVisit(a arm, cfg config) (map[string]float64, error) {
	rows := map[string]float64{}
	for _, r := range a.runs {
		cmd := exec.Command(r.bin, "-test.run=^$", "-test.bench="+r.bench, "-test.benchtime="+cfg.benchtime, "-test.count=1")
		outB, err := cmd.CombinedOutput()
		if err != nil {
			return rows, fmt.Errorf("benchmark binary %s failed: %w\n%s", filepath.Base(r.bin), err, tail(string(outB), 20))
		}
		maps.Copy(rows, parseBench(string(outB)))
	}
	if len(rows) == 0 {
		return rows, fmt.Errorf("the %s arm produced no parseable benchmark rows", a.label)
	}
	return rows, nil
}

// tail returns the last n lines, for putting a failing binary's own output in
// the error rather than making the reader go and re-run it.
func tail(s string, n int) string {
	ln := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(ln) > n {
		ln = ln[len(ln)-n:]
	}
	return "    " + strings.Join(ln, "\n    ")
}

var reBench = regexp.MustCompile(`^(Benchmark\S+?)-\d+\s+\d+\s+([\d.]+)\s+ns/op`)
var reKN = regexp.MustCompile(`K(\d+)_N(\d+)`)

// parseBench maps each benchmark line to a stable shape key. The key is the family plus the
// K/N digits, so a sub-benchmark renamed between tags (v1.17.1's `K768_N8192` vs the working
// tree's `K768_N8192_resident`) still matches on the shape it measures.
func parseBench(s string) map[string]float64 {
	m := map[string]float64{}
	for ln := range strings.SplitSeq(s, "\n") {
		g := reBench.FindStringSubmatch(strings.TrimSpace(ln))
		if g == nil {
			continue
		}
		name, nsStr := g[1], g[2]
		family := name
		if before, _, ok := strings.Cut(name, "/"); ok {
			family = before
		}
		key := family
		if kn := reKN.FindStringSubmatch(name); kn != nil {
			key = family + "/K" + kn[1] + "_N" + kn[2]
		} else if found := strings.Contains(name, "/"); found {
			key = name // no K/N — key on the full sub-benchmark name
		}
		ns, err := strconv.ParseFloat(nsStr, 64)
		if err == nil {
			m[key] = ns
		}
	}
	return m
}

func seriesFor(visits []map[string]float64, shape string) []float64 {
	var out []float64
	for _, v := range visits {
		if ns, ok := v[shape]; ok {
			out = append(out, ns)
		}
	}
	return out
}

// retained drops the first sample (warm-up discard) when there is more than one.
func retained(s []float64) []float64 {
	if len(s) <= 1 {
		return s
	}
	return s[1:]
}

func sharedShapes(a, b []map[string]float64) []string {
	inA, inB := keySet(a), keySet(b)
	var out []string
	for k := range inA {
		if inB[k] {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

func onlyIn(a, b []map[string]float64) []string {
	inA, inB := keySet(a), keySet(b)
	var out []string
	for k := range inA {
		if !inB[k] {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

func keySet(visits []map[string]float64) map[string]bool {
	set := map[string]bool{}
	for _, v := range visits {
		for k := range v {
			set[k] = true
		}
	}
	return set
}

// defaultTag returns the positional tag if given, else the highest vX.Y.Z tag.
func defaultTag(root string, pos []string) string {
	if len(pos) > 0 && strings.TrimSpace(pos[0]) != "" {
		return pos[0]
	}
	out, _ := gitIn(root, "tag", "--list", "v*", "--sort=-v:refname") //nolint:errcheck // a failed listing yields no tag, which the caller reports as INCONCLUSIVE
	re := regexp.MustCompile(`^v\d+\.\d+\.\d+$`)
	for ln := range strings.SplitSeq(out, "\n") {
		t := strings.TrimSpace(ln)
		if re.MatchString(t) {
			return t
		}
	}
	return ""
}

// arm is one side of the comparison: a test binary per package that defines a family, each run
// with a pattern naming exactly its families.
type arm struct {
	label string
	runs  []benchRun
	res   resolution
}

type benchRun struct {
	bin, bench string
}

// buildArm compiles each package's test binary in dir, lists its benchmarks and resolves the
// wanted families among them. skipAbsent skips a package directory dir does not have (a package
// added since the previous tag). The returned cleanup removes the binaries, on error too.
func buildArm(dir string, pkgs []string, label string, wanted []string, skipAbsent bool) (arm, func(), error) {
	a := arm{label: label}
	var bins []string
	cleanup := func() {
		for _, b := range bins {
			os.Remove(b) //nolint:errcheck // a temp benchmark binary: nothing to report
		}
	}
	binOf := map[string]string{}
	listed := map[string][]string{}
	for i, pkg := range pkgs {
		if skipAbsent {
			if fi, err := os.Stat(filepath.Join(dir, pkg)); err != nil || !fi.IsDir() {
				continue
			}
		}
		bin, err := buildBench(dir, pkg, fmt.Sprintf("%s-%d", label, i))
		if err != nil {
			return a, cleanup, fmt.Errorf("could not build %s's benchmarks: %w", pkg, err)
		}
		if bin == "" {
			continue // a package with no test files
		}
		bins = append(bins, bin)
		out, err := exec.Command(bin, "-test.list=^Benchmark").Output()
		if err != nil {
			return a, cleanup, fmt.Errorf("could not list %s's benchmarks: %w", pkg, err)
		}
		binOf[pkg], listed[pkg] = bin, parseTestList(string(out))
	}
	res, err := resolveFamilies(wanted, listed)
	if err != nil {
		return a, cleanup, err
	}
	a.res = res
	for _, pkg := range pkgs {
		if names := res.byPkg[pkg]; len(names) > 0 {
			a.runs = append(a.runs, benchRun{bin: binOf[pkg], bench: benchRegex(names)})
		}
	}
	return a, cleanup, nil
}

// buildBench compiles the test binary of pkg in dir into a temp file named for the arm, or
// returns "" when pkg has no test files. Compiling once per arm keeps the compiler out of the
// measured loop. GOWORK=off so the arm resolves its own module graph, not the dev workspace.
func buildBench(dir, pkg, label string) (string, error) {
	out := filepath.Join(os.TempDir(), "perfgate-"+label+".test")
	os.Remove(out) //nolint:errcheck // a stale binary from an earlier run, if any
	cmd := exec.Command("go", "test", "-c", "-o", out, pkg)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off")
	if b, err := cmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("%v: %s", err, firstLine(string(b)))
	}
	if _, err := os.Stat(out); err != nil {
		return "", nil
	}
	return out, nil
}

func worktreeAt(root, tag string) (string, func(), error) {
	wt, err := os.MkdirTemp("", "perfgate-wt")
	if err != nil {
		return "", func() {}, err
	}
	if b, err := gitIn(root, "worktree", "add", "-q", "-f", "--detach", wt, tag); err != nil {
		return "", func() {}, fmt.Errorf("worktree add: %s", firstLine(b))
	}
	cleanup := func() {
		gitIn(root, "worktree", "remove", "--force", wt) //nolint:errcheck // cleanup of a temp worktree: nothing to report
		gitIn(root, "worktree", "prune")                 //nolint:errcheck // same
	}
	return wt, cleanup, nil
}

func gitIn(dir string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// --- small stats ---

func median(s []float64) float64 {
	c := append([]float64(nil), s...)
	sort.Float64s(c)
	n := len(c)
	if n == 0 {
		return 0
	}
	if n%2 == 1 {
		return c[n/2]
	}
	return (c[n/2-1] + c[n/2]) / 2
}

func mean(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range s {
		sum += v
	}
	return sum / float64(len(s))
}

func stddev(s []float64) float64 {
	if len(s) < 2 {
		return 0
	}
	m := mean(s)
	var ss float64
	for _, v := range s {
		d := v - m
		ss += d * d
	}
	return sqrt(ss / float64(len(s)-1))
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return math.Sqrt(x)
}

func maxF(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func firstLine(s string) string {
	for ln := range strings.SplitSeq(s, "\n") {
		if strings.TrimSpace(ln) != "" {
			return strings.TrimSpace(ln)
		}
	}
	return ""
}
