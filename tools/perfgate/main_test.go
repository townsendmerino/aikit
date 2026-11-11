package main

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/tools/gate"
)

// parseBench must key on the shape (family + K/N), so a sub-benchmark renamed between tags —
// v1.17.1's `K768_N8192` vs the working tree's `K768_N8192_resident` — still matches.
func TestParseBench_keysOnShapeAcrossRename(t *testing.T) {
	cur := parseBench("BenchmarkW8A8SpanShapes/K768_N8192_resident-8   \t   6   \t   224084 ns/op\n")
	prev := parseBench("BenchmarkW8A8SpanShapes/K768_N8192-8   \t   6   \t   230000 ns/op\n")
	const key = "BenchmarkW8A8SpanShapes/K768_N8192"
	if _, ok := cur[key]; !ok {
		t.Fatalf("cur keys = %v, want %q", cur, key)
	}
	if _, ok := prev[key]; !ok {
		t.Fatalf("prev keys = %v, want %q", prev, key)
	}
	if cur[key] != 224084 || prev[key] != 230000 {
		t.Errorf("ns/op parsed wrong: cur=%v prev=%v", cur[key], prev[key])
	}
}

func TestParseBench_ignoresNonBenchLines(t *testing.T) {
	m := parseBench("=== RUN   TestFoo\nPASS\nok  \tpkg\t1.2s\n")
	if len(m) != 0 {
		t.Errorf("parsed %v from non-benchmark output; want empty", m)
	}
}

// visits builds a []map series for one shape: a warm-up sample (discarded) followed by the
// steady-state samples.
func visits(shape string, ns ...float64) []map[string]float64 {
	out := make([]map[string]float64, len(ns))
	for i, v := range ns {
		out[i] = map[string]float64{shape: v}
	}
	return out
}

func TestJudge_threeBranches(t *testing.T) {
	const sh = "B/K1_N1"
	cfg := config{minFloor: 2.0, floorK: 3.0}

	// flat: current within the floor of prev (warm-up 999 discarded on each arm).
	flat := judge(sh, visits(sh, 999, 100, 101, 100), visits(sh, 999, 100, 100, 101), 3.0, cfg)
	if flat.Outcome != gate.OK {
		t.Errorf("flat: outcome=%v, want ok", flat.Outcome)
	}
	if b := flat.Field("branch"); b != "flat" {
		t.Errorf("flat: branch=%q", b)
	}

	// regression: current ~10% slower than prev, floor 3% → FAIL.
	reg := judge(sh, visits(sh, 999, 110, 110, 111), visits(sh, 999, 100, 100, 100), 3.0, cfg)
	if reg.Outcome != gate.Fail {
		t.Errorf("regression: outcome=%v, want FAIL", reg.Outcome)
	}

	// faster: current ~10% faster than prev, floor 3% → ok but branch names the win.
	fast := judge(sh, visits(sh, 999, 90, 90, 91), visits(sh, 999, 100, 100, 100), 3.0, cfg)
	if fast.Outcome != gate.OK {
		t.Errorf("faster: outcome=%v, want ok", fast.Outcome)
	}
	if b := fast.Field("branch"); b == "flat" || b == "REGRESSION" {
		t.Errorf("faster: branch=%q, want the win text", b)
	}
}

func TestDeriveFloors_kSigmaFloorNeverBelowMin(t *testing.T) {
	cfg := config{minFloor: 2.0, floorK: 3.0}

	// a quiet shape: samples ~1% spread → k·σ below the 2% minimum → floored at 2%.
	quiet := deriveFloors(visits("s", 999, 100, 100.5, 100), visits("s", 999, 100, 99.5, 100.5), cfg)
	if quiet["s"] != 2.0 {
		t.Errorf("quiet floor = %.3f, want the 2.0%% minimum", quiet["s"])
	}

	// a noisy shape: samples spread ~10% → k·σ dominates and the floor rises above the min.
	noisy := deriveFloors(visits("s", 999, 100, 130, 90), visits("s", 999, 110, 85, 125), cfg)
	if noisy["s"] <= 2.0 {
		t.Errorf("noisy floor = %.3f, want > 2.0%% (k·σ should dominate)", noisy["s"])
	}
}

func TestStats(t *testing.T) {
	if m := median([]float64{3, 1, 2}); m != 2 {
		t.Errorf("median odd = %v, want 2", m)
	}
	if m := median([]float64{1, 2, 3, 4}); m != 2.5 {
		t.Errorf("median even = %v, want 2.5", m)
	}
	if s := stddev([]float64{2, 4, 4, 4, 5, 5, 7, 9}); math.Abs(s-2.138) > 0.01 {
		t.Errorf("stddev = %v, want ~2.138", s)
	}
}

// A flag value that does not parse used to become zero without a word (zero visits,
// a zero target). It is INCONCLUSIVE now: exit 2 before anything is measured.
func TestRun_rejectsUnparsableFlagValues(t *testing.T) {
	for _, args := range [][]string{
		{"--visits", "many"}, {"--visits", "0"}, {"--visits", "-3"},
		{"--target", "five"}, {"--target", "0"},
	} {
		if rc := run(args); rc != 2 {
			t.Errorf("run(%v) = %d, want 2", args, rc)
		}
	}
}

// The line perfgate prints is the line the release gate reads: pasted as printed, it passes.
func TestVerdict_isPasteReady(t *testing.T) {
	got := verdict(config{}, gate.OK, "no regression vs v1.76.0 above each shape's floor")
	if want := "`perfgate` VERDICT: PASS — no regression vs v1.76.0 above each shape's floor"; got != want {
		t.Errorf("verdict = %q, want %q", got, want)
	}
	dirty := verdict(config{dirtyOutside: true}, gate.OK, "x")
	if !strings.HasSuffix(dirty, "NOT RELEASE EVIDENCE") {
		t.Errorf("a verdict measured with --allow-dirty is not marked: %q", dirty)
	}
}

// CHANGELOG.md is dirty by design when perfgate runs in the release steps; nothing else may be.
func TestDirtOutsideChangelog(t *testing.T) {
	for _, tc := range []struct {
		name, porcelain string
		want            []string
	}{
		{"clean", "", nil},
		{"the release section", " M CHANGELOG.md\n", nil},
		{"paths the go tool ignores", "?? .claude/settings.local.json.bak\n?? _to_delete/x.go\n M CHANGELOG.md\n", nil},
		{"a kernel edit", " M CHANGELOG.md\n M linalg/quant.go\n", []string{"linalg/quant.go"}},
		{"an untracked source file", "?? linalg/new_kernel.go\n", []string{"linalg/new_kernel.go"}},
		{"a rename", "R  linalg/a.go -> linalg/b.go\n", []string{"linalg/b.go"}},
		{"another changelog is not the changelog", " M docs/CHANGELOG.md\n", []string{"docs/CHANGELOG.md"}},
	} {
		if got := dirtOutsideChangelog(tc.porcelain); !slices.Equal(got, tc.want) {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
}
