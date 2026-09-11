package main

import (
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/tools/gate"
)

func TestDetectBackend(t *testing.T) {
	if got, want := detectBackend("Darwin"), "metal"; got != want {
		t.Errorf("detectBackend(Darwin) = %q, want %q", got, want)
	}
	// Linux without nvidia-smi on PATH (this dev/CI box has no GPU) must fall through to
	// "none" rather than panicking or hanging on a missing binary.
	if got, want := detectBackend("Linux"), "none"; got != want {
		t.Logf("detectBackend(Linux) = %q (want %q unless this box genuinely has nvidia-smi)", got, want)
	}
}

func TestDevCell(t *testing.T) {
	c := devCell("mod", gate.OK, "ok", "(3 passed)")
	if c.Name != "mod" || c.Outcome != gate.OK {
		t.Errorf("devCell Name/Outcome = %q/%v, want mod/OK", c.Name, c.Outcome)
	}
	if got := c.Field("status"); got != "ok" {
		t.Errorf("status = %q, want ok", got)
	}
	if got := c.Field("inline"); got != "(3 passed)" {
		t.Errorf("inline = %q, want (3 passed)", got)
	}
}

func TestFollow(t *testing.T) {
	out := "line one\n\nline two\nline three\nline four\n"
	fs := follow(out, 2)
	if len(fs) != 2 {
		t.Fatalf("len(fs) = %d, want 2", len(fs))
	}
	if fs[0].State != "line one" || fs[1].State != "line two" {
		t.Errorf("fs = %v, want [line one, line two] (blank lines skipped, capped at n)", fs)
	}
}

func TestFollow_fewerLinesThanN(t *testing.T) {
	fs := follow("only one line\n", 5)
	if len(fs) != 1 || fs[0].State != "only one line" {
		t.Errorf("fs = %v, want exactly one field", fs)
	}
}

func TestFailLines_capturesFailAndPanicOnly(t *testing.T) {
	out := strings.Join([]string{
		"=== RUN   TestFoo",
		"--- FAIL: TestFoo (0.00s)",
		"    foo_test.go:10: assertion failed",
		"FAIL",
		"panic: runtime error: index out of range",
		"ordinary log line, not a failure marker",
	}, "\n")
	fs := failLines(out)
	if len(fs) != 3 {
		t.Fatalf("failLines = %v, want 3 matching lines (--- FAIL, FAIL, panic)", fs)
	}
	if !strings.HasPrefix(fs[0].State, "--- FAIL") {
		t.Errorf("fs[0] = %q, want a --- FAIL line", fs[0].State)
	}
	if fs[1].State != "FAIL" {
		t.Errorf("fs[1] = %q, want FAIL", fs[1].State)
	}
	if !strings.HasPrefix(fs[2].State, "panic:") {
		t.Errorf("fs[2] = %q, want a panic line", fs[2].State)
	}
}

func TestFailLines_capsAtFive(t *testing.T) {
	var lines []string
	for range 10 {
		lines = append(lines, "FAIL")
	}
	fs := failLines(strings.Join(lines, "\n"))
	if len(fs) != 5 {
		t.Errorf("len(fs) = %d, want capped at 5", len(fs))
	}
}

func TestCountTopLevel(t *testing.T) {
	out := strings.Join([]string{
		"=== RUN   TestA",
		"--- PASS: TestA (0.00s)",
		"=== RUN   TestB",
		"    --- PASS: TestB/sub (0.00s)", // indented (subtest) — must NOT count as top-level
		"--- PASS: TestB (0.00s)",
		"=== RUN   TestC",
		"--- SKIP: TestC (0.00s)",
	}, "\n")
	ran, skip := countTopLevel(out)
	if ran != 2 {
		t.Errorf("ran = %d, want 2 (indented subtest PASS must not count)", ran)
	}
	if skip != 1 {
		t.Errorf("skip = %d, want 1", skip)
	}
}

func TestCountTopLevel_allSkippedGivesZeroRan(t *testing.T) {
	out := "--- SKIP: TestA (0.00s)\n--- SKIP: TestB (0.00s)\n"
	ran, skip := countTopLevel(out)
	if ran != 0 || skip != 2 {
		t.Errorf("ran=%d skip=%d, want ran=0 skip=2", ran, skip)
	}
}

// TestDeviceVerdict_allSkipIsNotGreen is audit G-07: a module that ran but
// skipped every test must not be counted in the pasted "N/N green" line.
func TestDeviceVerdict_allSkipIsNotGreen(t *testing.T) {
	// No skips: all applicable modules are green.
	if s, ok := deviceVerdict(5, nil, "abc1234", "box", "metal", "today"); !ok ||
		!strings.Contains(s, "5/5 applicable gpu modules green") {
		t.Errorf("clean run: ok=%v s=%q", ok, s)
	}
	// Two all-skip modules: green is 3/5, and the line must say so rather than
	// claiming 5/5 as it used to.
	s, ok := deviceVerdict(5, []string{"gpu/visionmetal", "gpu/qwenmetal"}, "abc1234", "box", "metal", "today")
	if !ok {
		t.Fatalf("a partially-skipped run should still pass: %q", s)
	}
	if !strings.Contains(s, "3/5 applicable gpu modules green") {
		t.Errorf("want 3/5 green, got %q", s)
	}
	if !strings.Contains(s, "NOT counted green") || !strings.Contains(s, "gpu/visionmetal") {
		t.Errorf("verdict must name the skipped modules: %q", s)
	}
	// Everything skipped: not a pass at all.
	if s, ok := deviceVerdict(2, []string{"a", "b"}, "abc1234", "box", "metal", "today"); ok ||
		!strings.Contains(s, "every test skipped") {
		t.Errorf("all-skip run should be inconclusive: ok=%v s=%q", ok, s)
	}
}
