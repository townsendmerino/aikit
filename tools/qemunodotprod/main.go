// Command qemunodotprod runs a package's test suite on an arm64 CPU WITHOUT DotProd (Cortex-A72, a Raspberry Pi 4) by cross-compiling the test binary and running every test in its own
// qemu-user process with `-cpu cortex-a72`.
//
// It exists for the one class of bug the -tags aikit_nodotprod job cannot reach. That tag makes the dispatchers take the base SMULL/SADALP kernels on a runner that HAS DotProd, so a test or
// kernel that calls an SDOT instruction directly, without asking hasDotProd first, still runs fine there. On a core that really lacks the instruction it dies with SIGILL. This tool is that
// core. One such test was found by hand on 2026-10-04 (TestActGroup_row4KernelMatchesReference called the row4 SDOT kernel directly; fixed in v1.54.0).
//
// Each test gets its own process so a SIGILL names the test (a whole-suite run dies at the first one and hides the rest), and a hang under emulation is a timeout of one test, not of the job.
// Timing tests mean nothing under emulation and are skipped by name (-skip; the default names the FMA-peak probe, which infers the clock from throughput and read 0.35 GHz).
//
// It refuses to pass vacuously: it first runs the package's kernel-report test and fails unless that reports arm64 with dotprod absent, so a qemu without -cpu support, or a flag that
// did not take, is a failure rather than a green run on the wrong CPU. The denominator (tests run, ok, skipped) is printed.
//
// Usage: go run -C tools ./qemunodotprod [-pkg ./linalg] [-cpu cortex-a72] [-j 4] [-timeout 3m] [-skip regexp]
package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/townsendmerino/aikit/tools/gpumod"
)

// outcome is what one test's process did.
type outcome string

const (
	outOK      outcome = "ok"
	outSkip    outcome = "skip"
	outSIGILL  outcome = "SIGILL"
	outTimeout outcome = "timeout"
	outFail    outcome = "fail"
	outNone    outcome = "no-such-test"
)

// classify turns one test process's result into an outcome. exitErr is nil on exit status 0; timedOut means the per-test deadline fired.
func classify(output string, exitErr error, timedOut bool) outcome {
	low := strings.ToLower(output)
	switch {
	case timedOut:
		return outTimeout
	case strings.Contains(low, "sigill") || strings.Contains(low, "illegal instruction"):
		return outSIGILL // the Go runtime prints "SIGILL: illegal instruction" for a guest instruction the CPU lacks; qemu prints "uncaught target signal 4 (Illegal instruction)"
	case strings.Contains(output, "no tests to run"):
		return outNone
	case exitErr != nil:
		return outFail
	case strings.Contains(output, "--- SKIP"):
		return outSkip
	default:
		return outOK
	}
}

var kernelLine = regexp.MustCompile(`arch (\S+) detected \[([^\]]*)\] active \[([^\]]*)\]`)

// checkKernels parses the kernel-report test's -v output and returns an error unless the CPU is arm64 with dotprod neither detected nor active.
func checkKernels(output string) error {
	m := kernelLine.FindStringSubmatch(output)
	if m == nil {
		return fmt.Errorf("the kernel-report test printed no `arch ... detected [...] active [...]` line, so this run cannot say which CPU it ran on")
	}
	if m[1] != "arm64" {
		return fmt.Errorf("the test binary ran as %q, not arm64", m[1])
	}
	for _, s := range []string{m[2], m[3]} {
		for _, k := range strings.Fields(s) {
			if k == "dotprod" {
				return fmt.Errorf("DotProd is still detected or active (detected [%s] active [%s]): -cpu did not remove it, so this would be a green run on the wrong CPU", m[2], m[3])
			}
		}
	}
	return nil
}

type result struct {
	name string
	out  outcome
	tail string
}

func main() { os.Exit(run()) }

func run() int {
	pkg := flag.String("pkg", "./linalg", "package to cross-compile and run")
	cpu := flag.String("cpu", "cortex-a72", "qemu -cpu model (must lack DotProd)")
	jobs := flag.Int("j", 4, "tests to run at once")
	per := flag.Duration("timeout", 3*time.Minute, "per-test deadline")
	skip := flag.String("skip", `^TestFMAPeak`, "tests to skip by name (timing tests that mean nothing under emulation)")
	kernelTest := flag.String("kernel-test", "TestActiveKernels_isConsistent", "the test whose -v output reports the active kernels")
	flag.Parse()

	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Fprintln(os.Stderr, "qemunodotprod:", err)
		return 2
	}
	qemu, err := findQEMU()
	if err != nil {
		fmt.Fprintln(os.Stderr, "qemunodotprod:", err)
		return 2
	}
	skipRE, err := regexp.Compile(*skip)
	if err != nil {
		fmt.Fprintln(os.Stderr, "qemunodotprod: -skip:", err)
		return 2
	}
	tmp, err := os.MkdirTemp("", "qemunodotprod-*")
	if err != nil {
		fmt.Fprintln(os.Stderr, "qemunodotprod:", err)
		return 2
	}
	defer os.RemoveAll(tmp)
	bin := filepath.Join(tmp, "pkg.test")

	build := exec.Command("go", "test", "-c", "-o", bin, *pkg)
	build.Dir = root
	build.Env = append(os.Environ(), "GOARCH=arm64", "CGO_ENABLED=0")
	if out, err := build.CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "qemunodotprod: cross-compiling %s for arm64: %v\n%s", *pkg, err, out)
		return 2
	}

	runTest := func(ctx context.Context, args ...string) (string, error, bool) {
		cctx, cancel := context.WithTimeout(ctx, *per)
		defer cancel()
		cmd := exec.CommandContext(cctx, qemu, append([]string{"-cpu", *cpu, bin}, args...)...)
		cmd.Dir = filepath.Join(root, strings.TrimPrefix(*pkg, "./"))
		var buf bytes.Buffer
		cmd.Stdout, cmd.Stderr = &buf, &buf
		err := cmd.Run()
		return buf.String(), err, cctx.Err() == context.DeadlineExceeded
	}
	ctx := context.Background()

	kout, kerr, kto := runTest(ctx, "-test.run", "^"+regexp.QuoteMeta(*kernelTest)+"$", "-test.v")
	if kerr != nil || kto {
		fmt.Fprintf(os.Stderr, "qemunodotprod: the kernel-report test %s did not pass under -cpu %s: %v\n%s", *kernelTest, *cpu, kerr, kout)
		return 1
	}
	if err := checkKernels(kout); err != nil {
		fmt.Fprintln(os.Stderr, "qemunodotprod:", err)
		return 1
	}
	fmt.Printf("qemunodotprod: %s under %s -cpu %s: %s\n", *pkg, filepath.Base(qemu), *cpu, strings.TrimSpace(kernelLine.FindString(kout)))

	lout, lerr, _ := runTest(ctx, "-test.list", ".")
	if lerr != nil {
		fmt.Fprintf(os.Stderr, "qemunodotprod: listing tests: %v\n%s", lerr, lout)
		return 1
	}
	var names, skipped []string
	for _, l := range strings.Split(lout, "\n") {
		l = strings.TrimSpace(l)
		if !strings.HasPrefix(l, "Test") {
			continue
		}
		if skipRE.MatchString(l) {
			skipped = append(skipped, l)
			continue
		}
		names = append(names, l)
	}
	if len(names) == 0 {
		fmt.Fprintln(os.Stderr, "qemunodotprod: no tests listed; this would pass having run nothing")
		return 1
	}

	work := make(chan string)
	results := make(chan result)
	var wg sync.WaitGroup
	for i := 0; i < *jobs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for n := range work {
				out, err, to := runTest(ctx, "-test.run", "^"+regexp.QuoteMeta(n)+"$", "-test.v")
				o := classify(out, err, to)
				results <- result{n, o, tailOf(out, o)}
			}
		}()
	}
	go func() {
		for _, n := range names {
			work <- n
		}
		close(work)
		wg.Wait()
		close(results)
	}()
	count := map[outcome]int{}
	var bad []result
	for r := range results {
		count[r.out]++
		if r.out != outOK && r.out != outSkip {
			bad = append(bad, r)
		}
	}
	sort.Slice(bad, func(i, j int) bool { return bad[i].name < bad[j].name })
	fmt.Printf("qemunodotprod: %d tests run in their own processes: ok %d, skipped by the test %d, SIGILL %d, timeout %d, fail %d, not found %d; %d skipped by name (%s)\n",
		len(names), count[outOK], count[outSkip], count[outSIGILL], count[outTimeout], count[outFail], count[outNone], len(skipped), *skip)
	for _, r := range bad {
		fmt.Printf("\n--- %s: %s\n%s\n", r.name, r.out, r.tail)
	}
	if len(bad) > 0 {
		return 1
	}
	return 0
}

// tailOf is the part of a failing test's output worth printing: for a SIGILL, from the fault line (the Go runtime's "SIGILL: illegal instruction" and the goroutine that hit it, not the
// register dump after), otherwise the last lines.
func tailOf(out string, o outcome) string {
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if o == outSIGILL {
		for i, l := range lines {
			if strings.Contains(l, "SIGILL") {
				lines = lines[i:]
				if len(lines) > 14 {
					lines = lines[:14]
				}
				return strings.Join(lines, "\n")
			}
		}
	}
	if len(lines) > 12 {
		lines = lines[len(lines)-12:]
	}
	return strings.Join(lines, "\n")
}

func findQEMU() (string, error) {
	for _, n := range []string{"qemu-aarch64-static", "qemu-aarch64"} {
		if p, err := exec.LookPath(n); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("no qemu-aarch64-static or qemu-aarch64 on PATH (apt-get install qemu-user-static)")
}
