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
package main

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/townsendmerino/aikit/tools/gate"
	"github.com/townsendmerino/aikit/tools/gpumod"
)

const (
	// The gate's coverage (audit G-01). It used to be the two W8A8 M=1 families
	// alone, which left the production int4 decode kernel, BOTH M>1 tiles, the
	// acc64 attention kernels, the activation quantiser, the S-06 contract
	// kernels and AttendTileFused with no performance gate at all — every one of
	// them a path a release has changed. Each family below is here because a
	// regression in it would otherwise ship green:
	//
	//	W8A8SpanShapes            W8A8 M=1, the original two
	//	GEMV_W8A8_baseline        W8A8 M=1 through the f32 entry
	//	MatmulBTW8A8Batch_prefill W8A8 M>1 — the batched tile, goinfer's prefill
	//	W4A8_CanonicalVsSplitHalf W4A8 M=1 — the int4 decode kernel
	//	MatmulBTW4A8SplitHalfTile W4A8 M>1 on amd64 split-half-only — the tile
	//	                          this gate exists to protect (audit M-22 follow-up:
	//	                          split-half M>1)
	//	MatmulQKAcc64/AVAcc64     the f64 attention kernels at depth
	//	AttendTileFused_expKind   the fused ViT attention schedule
	//	SoftmaxRowKernels/SiLUKernels  the S-06 contract transcendentals
	//	QuantizeRowInt8           the activation quantiser
	defaultBench = `^(BenchmarkW8A8SpanShapes|BenchmarkGEMV_W8A8_baseline|` +
		`BenchmarkMatmulBTW8A8Batch_prefill|BenchmarkW4A8_CanonicalVsSplitHalf|` +
		`BenchmarkMatmulBTW4A8SplitHalfTile|` +
		`BenchmarkMatmulQKAcc64|BenchmarkMatmulAVAcc64|BenchmarkAttendTileFused_expKind|` +
		`BenchmarkSoftmaxRowKernels|BenchmarkSiLUKernels|BenchmarkQuantizeRowInt8)$`
	defaultPkg = "./linalg"
)

type config struct {
	prevTag     string
	pkg         string
	benchRe     string
	benchtime   string
	visits      int
	floorK      float64
	minFloor    float64
	targetClass float64 // the regression magnitude this gate exists to catch (percent)
}

func main() { os.Exit(run(os.Args[1:])) }

func run(args []string) int {
	cfg := config{pkg: defaultPkg, benchRe: defaultBench, benchtime: "200ms", visits: 6, floorK: 3.0, minFloor: 2.0, targetClass: 5.0}
	var pos []string
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--visits":
			i++
			cfg.visits, _ = strconv.Atoi(args[i])
		case "--benchtime":
			i++
			cfg.benchtime = args[i]
		case "--bench":
			i++
			cfg.benchRe = args[i]
		case "--pkg":
			i++
			cfg.pkg = args[i]
		case "--target":
			i++
			cfg.targetClass, _ = strconv.ParseFloat(args[i], 64)
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
	fmt.Printf("aikit perf gate — %s%s vs %s — %s/%s — %s\n", p.Commit, p.Dirty, cfg.prevTag, p.OSName, p.Arch, p.Date)
	fmt.Printf("instrument: %s %s   visits: %d   benchtime: %s   floor: max(%.1f%%, %.1f·σ)   targets: %.1f%% regressions\n\n",
		cfg.pkg, cfg.benchRe, cfg.visits, cfg.benchtime, cfg.minFloor, cfg.floorK, cfg.targetClass)

	// Build both arms. The working tree (uncommitted changes included — that is how the
	// acceptance reconstruction is measured) and the previous tag in a detached worktree.
	curBin, err := buildBench(root, cfg.pkg, "cur")
	if err != nil {
		fmt.Println("perfgate: INCONCLUSIVE — could not build the working-tree benchmark: " + err.Error())
		return 2
	}
	defer os.Remove(curBin)

	wt, cleanup, err := worktreeAt(root, cfg.prevTag)
	if err != nil {
		fmt.Println("perfgate: INCONCLUSIVE — could not check out " + cfg.prevTag + ": " + err.Error())
		return 2
	}
	defer cleanup()
	prevBin, err := buildBench(wt, cfg.pkg, "prev")
	if err != nil {
		fmt.Println("perfgate: INCONCLUSIVE — could not build the " + cfg.prevTag + " benchmark: " + err.Error())
		return 2
	}
	defer os.Remove(prevBin)

	// CHARACTERIZATION — the current binary against itself, interleaved exactly as the
	// comparison will be, to derive the per-shape floor BEFORE any comparison sample exists.
	fmt.Println("characterizing instrument noise (current vs current)…")
	charA := make([]map[string]float64, 0, cfg.visits)
	charB := make([]map[string]float64, 0, cfg.visits)
	for v := 0; v < cfg.visits; v++ {
		a, err := runVisit(curBin, cfg)
		if err != nil {
			fmt.Println(gate.Verdict(gate.Inconclusive, "characterization: "+err.Error()))
			return 4
		}
		b, err := runVisit(curBin, cfg)
		if err != nil {
			fmt.Println(gate.Verdict(gate.Inconclusive, "characterization: "+err.Error()))
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
		first, second := curBin, prevBin
		if v%2 == 1 {
			first, second = prevBin, curBin
		}
		r1, err := runVisit(first, cfg)
		if err != nil {
			fmt.Println(gate.Verdict(gate.Inconclusive, "comparison: "+err.Error()))
			return 4
		}
		r2, err := runVisit(second, cfg)
		if err != nil {
			fmt.Println(gate.Verdict(gate.Inconclusive, "comparison: "+err.Error()))
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
			first, second := curBin, prevBin
			if v%2 == 1 {
				first, second = prevBin, curBin
			}
			r1, err := runVisit(first, cfg)
			if err != nil {
				fmt.Println(gate.Verdict(gate.Inconclusive, "confirmation: "+err.Error()))
				return 4
			}
			r2, err := runVisit(second, cfg)
			if err != nil {
				fmt.Println(gate.Verdict(gate.Inconclusive, "confirmation: "+err.Error()))
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
			fmt.Println(gate.Verdict(gate.Inconclusive, fmt.Sprintf(
				"%d regression(s) did not reproduce on a second measurement — not green, not a fail; re-run or investigate",
				len(transient))))
			return 5
		}
		fmt.Printf("  reproduced: %s\n", strings.Join(confirmed, " "))
		fmt.Println(gate.Verdict(gate.Fail, fmt.Sprintf("%d regression(s) vs %s across %d shapes, confirmed on a second measurement",
			len(confirmed), cfg.prevTag, rep.Total)))
		return 1
	case gate.Inconclusive:
		fmt.Println(gate.Verdict(gate.Inconclusive, "a shape could not be measured"))
		return 2
	}
	// A gate that measured nothing, or that resolved the class it targets on
	// NOTHING, must not exit 0 (audit G-02, G-06). Both used to print a green.
	// Distinct codes so a caller can tell "this found no regression" from "this
	// could not have found one".
	if rep.Total == 0 {
		fmt.Println(gate.Verdict(gate.Inconclusive, "no shapes were measured — the gate did not run"))
		return 4
	}
	if below == 0 {
		fmt.Println(gate.Verdict(gate.Inconclusive, fmt.Sprintf(
			"BLIND — 0/%d shapes resolve the %.1f%% class, so this run is not evidence against a %.1f%% regression",
			rep.Total, cfg.targetClass, cfg.targetClass)))
		return 3
	}
	fmt.Println(gate.Verdict(gate.OK, fmt.Sprintf("no regression vs %s above each shape's floor — %d/%d shapes resolve the %.1f%% class",
		cfg.prevTag, below, rep.Total, cfg.targetClass)))
	return 0
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

// runVisit runs the compiled benchmark once and parses ns/op per shape key.
// runVisit runs the benchmark binary once and parses its rows.
//
// It returns an error when the binary FAILS or when it produced no parseable
// benchmark lines (audit G-06). Both used to be swallowed — `outB, _ :=` and
// then an empty map — so a panic, a regex typo or a missing fixture in the
// prev-tag worktree produced zero rows, zero shapes, and a green "PASS — 0/0".
// A gate whose instrument did not run must not report on the thing it was
// meant to measure.
func runVisit(bin string, cfg config) (map[string]float64, error) {
	cmd := exec.Command(bin, "-test.run=^$", "-test.bench="+cfg.benchRe, "-test.benchtime="+cfg.benchtime, "-test.count=1")
	outB, err := cmd.CombinedOutput()
	rows := parseBench(string(outB))
	if err != nil {
		return rows, fmt.Errorf("benchmark binary %s failed: %w\n%s", filepath.Base(bin), err, tail(string(outB), 20))
	}
	if len(rows) == 0 {
		return rows, fmt.Errorf("benchmark binary %s produced no parseable rows for %s\n%s",
			filepath.Base(bin), cfg.benchRe, tail(string(outB), 20))
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
	out, _ := gitIn(root, "tag", "--list", "v*", "--sort=-v:refname")
	re := regexp.MustCompile(`^v\d+\.\d+\.\d+$`)
	for ln := range strings.SplitSeq(out, "\n") {
		t := strings.TrimSpace(ln)
		if re.MatchString(t) {
			return t
		}
	}
	return ""
}

// buildBench compiles the benchmark package in `dir` into a temp binary named for the arm. The
// binary is what is run and re-run; compiling once per arm keeps the compiler out of the
// measured loop. GOWORK=off so the arm resolves its own module graph, not the dev workspace.
func buildBench(dir, pkg, label string) (string, error) {
	out := filepath.Join(os.TempDir(), "perfgate-"+label+".test")
	cmd := exec.Command("go", "test", "-c", "-o", out, pkg)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off")
	if b, err := cmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("%v: %s", err, firstLine(string(b)))
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
		gitIn(root, "worktree", "remove", "--force", wt)
		gitIn(root, "worktree", "prune")
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
