package main

import (
	"errors"
	"strings"
	"testing"
)

func TestClassify(t *testing.T) {
	fail := errors.New("exit status 1")
	for _, tc := range []struct {
		name string
		out  string
		err  error
		to   bool
		want outcome
	}{
		{"pass", "=== RUN   TestX\n--- PASS: TestX (0.00s)\nPASS\n", nil, false, outOK},
		{"skip", "=== RUN   TestX\n--- SKIP: TestX (0.00s)\nPASS\n", nil, false, outSkip},
		{"ordinary failure", "--- FAIL: TestX (0.00s)\nFAIL\n", fail, false, outFail},
		{"go runtime SIGILL", "SIGILL: illegal instruction\nPC=0x1234 m=0 sigcode=1\n", errors.New("exit status 2"), false, outSIGILL},
		{"qemu SIGILL", "qemu: uncaught target signal 4 (Illegal instruction) - core dumped\n", errors.New("signal: illegal instruction"), false, outSIGILL},
		{"timeout beats everything", "SIGILL", fail, true, outTimeout},
		{"name matched nothing", "testing: warning: no tests to run\nPASS\n", nil, false, outNone},
	} {
		if got := classify(tc.out, tc.err, tc.to); got != tc.want {
			t.Errorf("%s: classify = %s, want %s", tc.name, got, tc.want)
		}
	}
}

func TestCheckKernels(t *testing.T) {
	ok := "    selfcheck_test.go:129: arch arm64 detected [] active [] forced []\n--- PASS\n"
	if err := checkKernels(ok); err != nil {
		t.Errorf("an arm64 CPU with no dotprod was rejected: %v", err)
	}
	for name, out := range map[string]string{
		"dotprod detected":    "arch arm64 detected [dotprod] active [] forced []",
		"dotprod active":      "arch arm64 detected [dotprod] active [dotprod] forced []",
		"amd64 binary":        "arch amd64 detected [avx2] active [avx2] forced []",
		"no report line":      "--- PASS: TestActiveKernels_isConsistent",
		"forced does not fix": "arch arm64 detected [dotprod] active [] forced [nodotprod]",
	} {
		if err := checkKernels(out); err == nil {
			t.Errorf("%s: accepted, but this must fail (a green run on the wrong CPU)", name)
		}
	}
}

func TestTailOf(t *testing.T) {
	out := "=== RUN   TestX\nSIGILL: illegal instruction\nPC=0x1 m=0\ngoroutine 6 [running]:\nlinalg.dotI8SDOT(...)\nr0 0x0\nr1 0x1\n"
	got := tailOf(out, outSIGILL)
	if got[:6] != "SIGILL" || !contains(got, "dotI8SDOT") {
		t.Errorf("a SIGILL tail should start at the fault and keep the stack, got %q", got)
	}
	if got := tailOf("a\nb\nc", outFail); got != "a\nb\nc" {
		t.Errorf("short failing output should come back whole, got %q", got)
	}
}

func contains(s, sub string) bool { return strings.Contains(s, sub) }
