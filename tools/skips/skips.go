// Package skips runs `go test` and makes SKIPS VISIBLE. A skipped test prints `--- SKIP`
// under -v but the package summary is still `ok`, so in the automated path — CI's
// `go test -race ./...` and tools/preflight — a skip folds into green with no count. On this
// repo that hides a lot: ~145 tests skip because a model/checkpoint asset is absent (as on
// CI) and ~57 because no GPU device is present. A green that says "412 passed, 202 skipped
// (145 missing asset, 57 no device, …)" is honest; a bare green is not.
//
// This is a DENOMINATOR, not a gate. A skip is not made a failure here (the two hand-run gpu
// gates already do that where it belongs — gpugate maps a skip on its kernel assertions to
// FAIL, gpudevice reports SKIPPED and goes Inconclusive with no backend). Here the run's exit
// code is exactly `go test`'s: pass stays pass, fail stays fail. The only addition is that the
// skip count and its by-reason breakdown are printed next to the verdict.
//
// It works by running `go test -json`, re-emitting the human-readable output verbatim (the
// JSON "output" events carry exactly what `go test` would have printed, so a CI log is
// unchanged), and classifying each skip by the reason text the test passed to t.Skip.
package skips

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"slices"
	"sort"
	"strings"
)

// Result is the tally of one `go test` run.
type Result struct {
	Passed   int            // tests that passed (Action=="pass" with a Test name)
	Failed   int            // tests that failed
	Skipped  int            // tests that skipped
	Packages int            // packages that reported ok/fail (Action pass|fail with no Test)
	ByReason map[string]int // skip count per reason category, e.g. "no GPU device" → 57
	Examples map[string]string
	ExitCode int // `go test`'s own exit code — the run's verdict is unchanged
}

// category classifies a skip's reason text. Ordered, first-match-wins: the earlier rules are
// the specific ones (an "AIKIT_GPU_BENCH" opt-in mentions "gpu" but is not a no-device skip),
// so they are checked before the broad ones. A reason that matches nothing lands in "other" —
// a real bucket that keeps an example, never a silent lump.
var reasonRules = []struct {
	cat  string
	pats []string
}{
	{"short mode", []string{"-short", "short mode"}},
	{"cpu feature", []string{"avx2", "fma", "dotprod", "asimddp", "sdot", "cpu lacks", "cpu has no", "no neon"}},
	{"missing external tool", []string{"nvrtc", "python3", "libnvrtc", "not found in $path", "not on path"}},
	{"env opt-in", []string{"aikit_gpu_bench", "_probe", "goinfer_", "set aikit", "environment variable", "getenv"}},
	{"no GPU device", []string{"no metal device", "metal device", "no cuda", "cuda device", "no gpu", "enableresident", "enablegpu", "newbackend", "no backend", "backend registered", "device?)"}},
	{"missing asset", []string{"not present", "no golden", "testdata", "fetch", "checkpoint", "model at", "no bge", "no bert", "no model", "scripts/pin", "no local", ".safetensors", ".gguf"}},
}

func category(reason string) string {
	r := strings.ToLower(reason)
	for _, rule := range reasonRules {
		for _, p := range rule.pats {
			if strings.Contains(r, p) {
				return rule.cat
			}
		}
	}
	return "other"
}

type testEvent struct {
	Action  string
	Package string
	Test    string
	Output  string
}

// reReason pulls the message out of a `    file_test.go:NN: <message>` line — the shape
// `go test` prints for a t.Skip/t.Skipf/t.Log call.
var reReason = regexp.MustCompile(`^\s+\S+\.go:\d+:\s*(.*)$`)

// Run executes `go test -json <args>` in dir, streams the reconstructed human-readable output
// to out, and returns the tally. extraEnv is appended to the inherited environment per call
// (CI passes nil — the ambient toolchain; preflight passes GOWORK=off for parity with its
// other checks). It never fails the run itself — ExitCode carries go test's verdict. A build
// error (no JSON) still streams and still returns the non-zero code.
func Run(dir string, extraEnv, args []string, out io.Writer) (Result, error) {
	res := Result{ByReason: map[string]int{}, Examples: map[string]string{}}
	full := append([]string{"test", "-json"}, args...)
	cmd := exec.Command("go", full...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), extraEnv...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return res, err
	}
	cmd.Stderr = out // build errors and non-JSON diagnostics pass straight through
	if err := cmd.Start(); err != nil {
		return res, err
	}

	// buffered per-test output, so a skip can be attributed to the reason text that preceded
	// it. Keyed by package+test; cleared on that test's terminal event to bound memory.
	buf := map[string][]string{}
	sc := bufio.NewScanner(stdout)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024) // test output lines can be long
	for sc.Scan() {
		line := sc.Bytes()
		var ev testEvent
		if json.Unmarshal(line, &ev) != nil {
			fmt.Fprintln(out, string(line)) // not JSON — pass it through unchanged
			continue
		}
		key := ev.Package + "\x00" + ev.Test
		switch ev.Action {
		case "output":
			io.WriteString(out, ev.Output) //nolint:errcheck // reproduce the normal go-test log; a write error on the log is not a test result
			if ev.Test != "" {
				buf[key] = append(buf[key], ev.Output)
			}
		case "skip":
			if ev.Test != "" {
				res.Skipped++
				cat := category(reasonFrom(buf[key]))
				res.ByReason[cat]++
				if _, ok := res.Examples[cat]; !ok {
					res.Examples[cat] = reasonFrom(buf[key])
				}
				delete(buf, key)
			}
		case "pass":
			if ev.Test != "" {
				res.Passed++
				delete(buf, key)
			} else {
				res.Packages++
			}
		case "fail":
			if ev.Test != "" {
				res.Failed++
				delete(buf, key)
			} else {
				res.Packages++
			}
		}
	}
	werr := cmd.Wait()
	res.ExitCode = 0
	if werr != nil {
		if ee, ok := werr.(*exec.ExitError); ok {
			res.ExitCode = ee.ExitCode()
		} else {
			res.ExitCode = -1
		}
	}
	return res, sc.Err()
}

// reasonFrom returns the last `file.go:NN:` message in a test's captured output (the t.Skip
// text), or the last non-empty, non-scaffolding line as a fallback.
func reasonFrom(lines []string) string {
	reason := ""
	for _, ln := range lines {
		if m := reReason.FindStringSubmatch(strings.TrimRight(ln, "\n")); m != nil {
			reason = strings.TrimSpace(m[1])
		}
	}
	if reason != "" {
		return reason
	}
	for _, line := range slices.Backward(lines) {
		t := strings.TrimSpace(line)
		if t != "" && !strings.HasPrefix(t, "=== ") && !strings.HasPrefix(t, "--- ") {
			return t
		}
	}
	return ""
}

// Summary formats the one-line-per-category tally printed under the verdict. Categories are
// ordered by count (largest first) so the biggest silent surface reads first; "other" always
// sorts last so a catch-all never leads.
func (r Result) Summary() string {
	if r.Skipped == 0 {
		return fmt.Sprintf("%d passed, 0 skipped", r.Passed)
	}
	type kv struct {
		cat string
		n   int
	}
	rows := make([]kv, 0, len(r.ByReason))
	for c, n := range r.ByReason {
		rows = append(rows, kv{c, n})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].cat == "other" != (rows[j].cat == "other") {
			return rows[j].cat == "other"
		}
		if rows[i].n != rows[j].n {
			return rows[i].n > rows[j].n
		}
		return rows[i].cat < rows[j].cat
	})
	var b strings.Builder
	fmt.Fprintf(&b, "%d passed, %d skipped", r.Passed, r.Skipped)
	if r.Failed > 0 {
		fmt.Fprintf(&b, ", %d FAILED", r.Failed)
	}
	b.WriteString("\n")
	for _, row := range rows {
		ex := r.Examples[row.cat]
		if len(ex) > 64 {
			ex = ex[:61] + "…"
		}
		fmt.Fprintf(&b, "  %4d  %-22s  e.g. %s\n", row.n, row.cat, ex)
	}
	return strings.TrimRight(b.String(), "\n")
}
