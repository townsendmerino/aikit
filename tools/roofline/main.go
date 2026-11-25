// Command roofline is the per-kernel roofline census: each kernel perfgate or goinfer's
// kernelprint names (linalg's BenchmarkRooflineCensus), at one cache-resident and one streamed
// shape, single-threaded, as operand GB/s and GMAC/s and as a fraction of this box's measured
// roof for its class (BenchmarkRooflineRoofs: the f32 FMA peak, the int8 dot peak, a streaming
// read). It writes the table under docs/bench-records/ with the CPU and the commit.
//
// A reading is a timed run: on a settled box, by day, under the exclusive timing lock. The
// registration it is read against is docs/bench-records/roofline-registration.md; the laggard
// lines below are that registration's.
//
// Usage:
//
//	go run -C tools ./roofline                        # three passes at 300ms, writes the record
//	go run -C tools ./roofline --int8-roof 69.0 --int8-roof-source "docs/internal/roofline-2026-08.md §0, Ryzen 7 3700X"
//	go run -C tools ./roofline --passes 1 --benchtime 50ms --out /some/scratch.md   # a quick look, not a reading
//
// It ends with one VERDICT line: PASS when every row was measured and every roof it needs was
// measured and passed its cross-check, INCONCLUSIVE otherwise (the table is still written, with
// the unbound rows marked).
package main

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/townsendmerino/aikit/tools/gate"
	"github.com/townsendmerino/aikit/tools/gpumod"
)

// The registered laggard lines (docs/bench-records/roofline-registration.md).
var registeredLines = laggardLines{streamed: 0.5, resident: 0.3, denseAI: 8, band: 0.05}

type config struct {
	passes     int
	benchtime  string
	int8Roof   float64
	int8Source string
	out        string
}

func main() { os.Exit(run(os.Args[1:], os.Stdout)) }

func run(args []string, w io.Writer) int {
	cfg := config{passes: 3, benchtime: "300ms"}
	for i := 0; i < len(args); i++ {
		next := func() string {
			if i+1 >= len(args) {
				return ""
			}
			i++
			return args[i]
		}
		switch args[i] {
		case "--passes":
			n, err := strconv.Atoi(next())
			if err != nil || n < 1 {
				fmt.Fprintln(w, gate.Verdict(gate.Inconclusive, "--passes wants a positive integer"))
				return 2
			}
			cfg.passes = n
		case "--benchtime":
			cfg.benchtime = next()
		case "--int8-roof":
			f, err := strconv.ParseFloat(next(), 64)
			if err != nil || f <= 0 {
				fmt.Fprintln(w, gate.Verdict(gate.Inconclusive, "--int8-roof wants a positive GMAC/s"))
				return 2
			}
			cfg.int8Roof = f
		case "--int8-roof-source":
			cfg.int8Source = next()
		case "--out":
			cfg.out = next()
		default:
			fmt.Fprintln(w, gate.Verdict(gate.Inconclusive, "unknown argument "+args[i]))
			return 2
		}
	}
	if cfg.int8Roof > 0 && strings.TrimSpace(cfg.int8Source) == "" {
		fmt.Fprintln(w, gate.Verdict(gate.Inconclusive, "--int8-roof needs --int8-roof-source: a cited roof names its record"))
		return 2
	}

	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Fprintln(w, gate.Verdict(gate.Inconclusive, "cannot locate repo root: "+err.Error()))
		return 2
	}
	prov := gpumod.Provenance(root)
	start := time.Now()
	fmt.Fprintf(w, "roofline census — %s%s — %s/%s — %d passes at %s\n", prov.Commit, prov.Dirty, prov.OSName, prov.Arch, cfg.passes, cfg.benchtime)

	bin := filepath.Join(os.TempDir(), "roofline-linalg.test")
	build := exec.Command("go", "test", "-c", "-o", bin, "./linalg")
	build.Dir = root
	build.Env = append(os.Environ(), "GOWORK=off")
	if out, err := build.CombinedOutput(); err != nil {
		fmt.Fprintln(w, gate.Verdict(gate.Inconclusive, "could not build linalg's test binary: "+firstLine(string(out))))
		return 2
	}
	defer os.Remove(bin) //nolint:errcheck // a temp test binary: nothing to report

	fmt.Fprintln(w, "measuring the roofs…")
	roofOut, err := runBench(bin, "^BenchmarkRooflineRoofs$", "1x", nil)
	if err != nil {
		fmt.Fprintln(w, gate.Verdict(gate.Inconclusive, "the roof probes: "+err.Error()))
		return 2
	}
	rp := parseOutput(roofOut)
	rf := roofsFrom(rp, cfg.int8Roof, cfg.int8Source)
	fmt.Fprintf(w, "  fma %.1f GFLOP/s (implies %.2f GHz)   int8 %s   stream %.1f GB/s (resident %.1f, cross-check %s)\n",
		rf.fmaGFLOPS, rf.impliedGHz(), orDash(rf.int8GMACs, "%.1f GMAC/s"), rf.streamGBs, rf.residentGBs, passWord(rf.streamPassed))

	var passes []parsed
	total := 0
	for pass := 1; pass <= cfg.passes; pass++ {
		n := 0
		out, err := runBench(bin, "^BenchmarkRooflineCensus$", cfg.benchtime, func(name string, ns float64) {
			n++
			if total > 0 {
				fmt.Fprintf(w, "  [pass %d/%d, row %d/%d, %s] %s  %.4g ns/op\n", pass, cfg.passes, n, total, since(start), name, ns)
			} else {
				fmt.Fprintf(w, "  [pass %d/%d, row %d, %s] %s  %.4g ns/op\n", pass, cfg.passes, n, since(start), name, ns)
			}
		})
		if err != nil {
			fmt.Fprintln(w, gate.Verdict(gate.Inconclusive, fmt.Sprintf("census pass %d: %v", pass, err)))
			return 2
		}
		p := parseOutput(out)
		if p.cpu == "" {
			p.cpu = rp.cpu
		}
		passes = append(passes, p)
		total = n
	}

	samples := samplesFrom(passes)
	rows := make([]row, 0, len(samples))
	for _, s := range samples {
		rows = append(rows, computeRow(s, rf, registeredLines))
	}
	cpu := rp.cpu
	if cpu == "" && len(passes) > 0 {
		cpu = passes[0].cpu
	}
	rec := record{cpu: cpu, goos: prov.OSName, goarch: prov.Arch, commit: prov.Commit + prov.Dirty, date: prov.Date,
		passes: cfg.passes, benchtime: cfg.benchtime, wall: time.Since(start), roofs: rf, rows: rows}
	path := cfg.out
	if path == "" {
		path = filepath.Join(root, "docs", "bench-records", "roofline-"+slug(cpu)+"-"+prov.Date[:10]+".md")
	}
	if err := os.WriteFile(path, []byte(rec.markdown()), 0o644); err != nil {
		fmt.Fprintln(w, gate.Verdict(gate.Inconclusive, "could not write the record: "+err.Error()))
		return 2
	}
	fmt.Fprintf(w, "wrote %s (%s)\n", path, since(start))
	if why := rec.inconclusive(); why != "" {
		fmt.Fprintln(w, gate.Verdict(gate.Inconclusive, why))
		return 3
	}
	lag, park := rec.laggards()
	fmt.Fprintln(w, gate.Verdict(gate.OK, fmt.Sprintf("%d rows against the roofs: %d laggards, %d parked", len(rows), lag, park)))
	return 0
}

// runBench runs bin's benchmarks matching pattern once, calling onRow for each result line as it
// arrives, and returns the whole output.
func runBench(bin, pattern, benchtime string, onRow func(name string, ns float64)) (string, error) {
	cmd := exec.Command(bin, "-test.run=^$", "-test.bench="+pattern, "-test.benchtime="+benchtime, "-test.count=1")
	pr, pw := io.Pipe()
	cmd.Stdout, cmd.Stderr = pw, pw
	if err := cmd.Start(); err != nil {
		return "", err
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait(); pw.Close() }()
	var all strings.Builder
	sc := bufio.NewScanner(pr)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		ln := sc.Text()
		all.WriteString(ln + "\n")
		if onRow != nil {
			if g := reRow.FindStringSubmatch(strings.TrimSpace(ln)); g != nil {
				if ns, err := strconv.ParseFloat(g[2], 64); err == nil {
					onRow(strings.TrimPrefix(g[1], "BenchmarkRooflineCensus/"), ns)
				}
			}
		}
	}
	if err := <-done; err != nil {
		return all.String(), fmt.Errorf("%w\n%s", err, tail(all.String(), 15))
	}
	return all.String(), nil
}

var reSlug = regexp.MustCompile(`[^a-z0-9]+`)

// slug names a record file after the CPU line, e.g. "amd-ryzen-7-3700x-8-core-processor".
func slug(cpu string) string {
	s := strings.Trim(reSlug.ReplaceAllString(strings.ToLower(cpu), "-"), "-")
	if s == "" {
		return "unknown-cpu"
	}
	return s
}

func since(t time.Time) string { return time.Since(t).Round(time.Second).String() }

func orDash(v float64, format string) string {
	if v <= 0 || math.IsNaN(v) {
		return "—"
	}
	return fmt.Sprintf(format, v)
}

func passWord(ok bool) string {
	if ok {
		return "passed"
	}
	return "FAILED"
}

func firstLine(s string) string {
	for ln := range strings.SplitSeq(s, "\n") {
		if t := strings.TrimSpace(ln); t != "" {
			return t
		}
	}
	return ""
}

func tail(s string, n int) string {
	ln := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(ln) > n {
		ln = ln[len(ln)-n:]
	}
	return "    " + strings.Join(ln, "\n    ")
}
