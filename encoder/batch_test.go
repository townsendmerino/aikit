package encoder

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"
)

// TestEncodeBatch_matchesEncode: per-position output of EncodeBatch
// must match a sequential Encode loop bit-for-bit. The model is
// immutable and per-call buffers are local, so this should hold
// regardless of worker count — a deviation here means a data race.
func TestEncodeBatch_matchesEncode(t *testing.T) {
	if _, err := os.Stat(modelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no model at %s — symlink testdata/encoder-model -> HF snapshot", modelDir)
	}
	m, err := Load(modelDir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	texts := []string{
		"how do i parse json",
		"def add(a, b):\n    return a + b",
		"compute the sha256 hash of a file",
		"recursive directory walk that respects gitignore",
	}
	isQ := []bool{true, false, true, true}

	// Reference: one-at-a-time.
	want := make([][]float32, len(texts))
	for i := range texts {
		v, err := m.Encode(texts[i], isQ[i])
		if err != nil {
			t.Fatalf("Encode[%d]: %v", i, err)
		}
		want[i] = v
	}

	// EncodeBatch at several worker counts (1 = degenerate sequential, NumCPU = full fan-out).
	for _, c := range []int{1, 2, runtime.NumCPU()} {
		t.Run(fmt.Sprintf("conc=%d", c), func(t *testing.T) {
			got, err := m.EncodeBatch(texts, isQ, c)
			if err != nil {
				t.Fatalf("EncodeBatch: %v", err)
			}
			if len(got) != len(want) {
				t.Fatalf("len: got %d want %d", len(got), len(want))
			}
			for i := range want {
				if len(got[i]) != len(want[i]) {
					t.Fatalf("[%d] len: got %d want %d", i, len(got[i]), len(want[i]))
				}
				for j := range want[i] {
					if got[i][j] != want[i][j] {
						t.Fatalf("[%d][%d]: got %v want %v", i, j, got[i][j], want[i][j])
					}
				}
			}
		})
	}
}

// TestEncodeBatch_errorPath: an empty-input contract sanity + length-mismatch error.
func TestEncodeBatch_errorPath(t *testing.T) {
	if _, err := os.Stat(modelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no model at %s", modelDir)
	}
	m, err := Load(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	// Empty: no work, no error.
	got, err := m.EncodeBatch(nil, nil, 0)
	if err != nil || got != nil {
		t.Errorf("empty: got=%v err=%v want nil/nil", got, err)
	}
	// Length mismatch: clear error.
	_, err = m.EncodeBatch([]string{"a", "b"}, []bool{true}, 0)
	if err == nil {
		t.Errorf("expected length-mismatch error")
	}
}

// TestEncodeBatch_speedup: report the wall-time ratio of sequential
// (Encode in a loop) vs parallel (EncodeBatch with NumCPU workers) on
// a realistic rerankN=8 batch of ~80-token inputs. This is the M3
// latency-gate input — independent forward passes should scale near
// linearly with some Amdahl loss.
// The 2× floor is GATED on the host, not asserted blindly: skipped below a
// parallel width of 4 (min(GOMAXPROCS, batch)), because a fixed 2× is
// unreachable on a 2-core box where the batch tops out at ~1.99×. It is not
// scaled UP with core count — this 8-short-text batch is overhead-bound and
// saturates at ~2× even on 8 cores (measured), so the Amdahl-curve speedup does
// not apply. See §1.20 in docs/internal/measuring-performance.md.
func TestEncodeBatch_speedup(t *testing.T) {
	if testing.Short() {
		t.Skip("speedup measurement runs ~50s; skipped under -short")
	}
	if _, err := os.Stat(modelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no model at %s", modelDir)
	}
	m, err := Load(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	// 8 candidates of roughly-uniform length (~70-90 tokens), so the
	// fan-out has minimal load-imbalance noise.
	texts := []string{
		"def add(a, b):\n    return a + b\n# comment line\nprint('hello')\n# more lines to pad",
		"import json\ndef load(s):\n    return json.loads(s)\n# more code\nx = 1; y = 2",
		"class Dog:\n    def bark(self):\n        print('woof')\n        return None\n# pad",
		"def fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)",
		"def sha256_of(path):\n    import hashlib\n    h = hashlib.sha256()\n    return h.hexdigest()",
		"def parse_url(s):\n    scheme, _, rest = s.partition('://')\n    return scheme, rest",
		"async def fetch(url):\n    async with httpx.AsyncClient() as c:\n        return await c.get(url)",
		"def walk(root):\n    for d, _, fs in os.walk(root):\n        for f in fs:\n            yield f",
	}
	isQ := make([]bool, len(texts))

	// The assertion is a CAPABILITY claim — this machine reaches the parallel speedup
	// it CAN — so the bar must account for the host, not be a bare constant. Two
	// separate failure modes (§1.20):
	//
	//  - CONTENTION → best-of-N. `go test ./...` runs packages concurrently, so this
	//    routinely shares the box with a dozen other tests; observed failing at 1.82×
	//    and 1.97× that way while measuring 2.6–2.8× alone. Best-of-N clears the flake;
	//    genuinely broken parallelism still fails every attempt.
	//  - A HOST THAT CANNOT REACH THE BAR → skip, don't assert. The achievable speedup
	//    is capped by the parallel WIDTH — min(GOMAXPROCS, batch size). At 2 effective
	//    cores EncodeBatch tops out near ~1.9–2.0× (apple-m1pro Amdahl §3: 2 workers =
	//    99.7% of linear), so a fixed 2.0× floor sits AT/ABOVE the ceiling and no retry
	//    count can clear it — the victim is a constrained container, a throttled laptop,
	//    or a pinned GOMAXPROCS, not CI (which skips this: the model is an HF-snapshot
	//    symlink). Use GOMAXPROCS, not NumCPU: a pinned GOMAXPROCS is exactly the case,
	//    and EncodeBatch's fan-out can only run as wide as the OS threads it gets.
	//
	// The floor is NOT scaled up with core count, deliberately. This realistic rerank
	// batch is 8 SHORT (~80-token) forwards, so it is OVERHEAD-bound, not width-bound:
	// measured on apple-m1pro it saturates at ~2.0–2.3× steady-state (median 2.06×) on
	// 8 cores, NOT the 5.23× the Amdahl §3 curve records — that curve is a larger
	// workload. A core-count-scaled bar (e.g. 0.5×width = 4× at 8 cores) would flake
	// here every warm run. So gate on whether the host can parallelize at all (width ≥
	// 4), then keep the honest ~2× floor that the workload actually reaches.
	width := min(runtime.GOMAXPROCS(0), len(texts))
	if width < 4 {
		t.Skipf("parallel width %d (min(GOMAXPROCS=%d, batch=%d)) is below 4 — this "+
			"overhead-bound batch cannot clear a 2× floor with so few effective cores, "+
			"and no retry count changes that", width, runtime.GOMAXPROCS(0), len(texts))
	}
	const floor = 2.0 // the ~2× this overhead-bound workload reaches when fan-out works.
	const attempts = 3
	var best float64
	for a := range attempts {
		t0 := time.Now()
		for i := range texts {
			_, err := m.Encode(texts[i], isQ[i])
			if err != nil {
				t.Fatalf("Encode: %v", err)
			}
		}
		seq := time.Since(t0)

		t1 := time.Now()
		_, err = m.EncodeBatch(texts, isQ, width) // match the gate above; EncodeBatch's own <=0 default is NumCPU, not GOMAXPROCS
		if err != nil {
			t.Fatal(err)
		}
		par := time.Since(t1)

		speedup := float64(seq) / float64(par)
		t.Logf("attempt %d: N=%d candidates, width=%d (GOMAXPROCS=%d, NumCPU=%d): sequential=%v parallel=%v -> %.2fx (floor %.1fx)",
			a+1, len(texts), width, runtime.GOMAXPROCS(0), runtime.NumCPU(), seq, par, speedup, floor)
		best = max(best, speedup)
		if best >= floor {
			return
		}
	}
	// BEFORE BLAMING THE CODE, ASK WHETHER THE HOST CAN FAN OUT AT ALL RIGHT NOW.
	// The message this test used to end on — "genuine fan-out failure, not a small
	// host" — names two causes and misses the third. A BUSY host produces exactly
	// this signature: the sequential arm is unaffected (it wants one core and gets
	// one), while the parallel arm cannot find `width` free cores, so the ratio
	// collapses without anything being wrong with EncodeBatch.
	//
	// Observed 2026-09-01, and it is why this guard exists: on an 8-core Mac with a
	// neighbouring decode benchmark holding ~4 cores (load 29), this reported
	// 1.87x and asserted a "genuine fan-out failure". The same commit on an idle
	// 16-core box reached 4.68x on the first attempt. Nothing was wrong with the
	// code, and the failure message said there was.
	//
	// Note the sequential arm does NOT reveal this — it was 443ms and 442ms on
	// attempts 2 and 3 of the loaded run, i.e. perfectly stable. Only a direct
	// measurement of available parallelism separates the two, so that is what this
	// does: a pure-CPU probe at the same width, with no encoder involved. If the
	// probe cannot clear the floor, the machine cannot, and this test has nothing
	// to say about the code.
	if probe := hostFanOut(width); probe < floor {
		t.Skipf("INCONCLUSIVE: a pure-CPU probe at width %d reached only %.2fx (floor %.1fx), so this "+
			"host cannot currently fan out — EncodeBatch's %.2fx says nothing about the code. "+
			"Re-run on a quiet machine.", width, probe, floor, best)
	}
	t.Errorf("best parallelism speedup over %d attempts was %.2fx, below floor %.1fx "+
		"(width %d — genuine fan-out failure: a pure-CPU probe at the same width DID clear "+
		"the floor on this host just now, so the cores were available and EncodeBatch did not "+
		"use them)", attempts, best, floor, width)
}

// hostFanOut measures the parallel speedup this machine can deliver RIGHT NOW at
// the given width, using a pure-CPU loop with no allocation, no encoder and no
// shared state — so the only thing it can be limited by is available cores.
//
// It is a canary for the measurement rather than a test of anything: its job is to
// tell TestEncodeBatch_speedup's failure branch whether to blame the code or the
// box, which that test could not previously distinguish.
func hostFanOut(width int) float64 {
	const span = 1 << 22
	burn := func(lo, hi int) uint64 {
		var acc uint64
		for i := lo; i < hi; i++ {
			acc = acc*1664525 + uint64(i) + 1013904223
		}
		return acc
	}
	t0 := time.Now()
	sink := burn(0, span)
	seq := time.Since(t0)

	// Each worker owns its own slot: no shared writes, so this stays clean under
	// -race (which CI runs) and the probe measures cores rather than contention on
	// its own accumulator.
	out := make([]uint64, width)
	t1 := time.Now()
	var wg sync.WaitGroup
	chunk := span / width
	for w := range width {
		wg.Add(1)
		go func(w, lo int) {
			defer wg.Done()
			hi := min(lo+chunk, span)
			out[w] = burn(lo, hi)
		}(w, w*chunk)
	}
	wg.Wait()
	par := time.Since(t1)
	_, _ = sink, out
	if par <= 0 {
		return 0
	}
	return float64(seq) / float64(par)
}

// BenchmarkEncodeBatch_rerankN50 wraps the same workload as
// TestEncodeBatch_rerankN50 in a Benchmark so `go test -bench` can
// drive it with -cpuprofile / -memprofile / -trace flags. Skipped when
// the model isn't present (CI / fresh checkouts). The benchmark runs
// the WARM path — b.N iterations share a single Model and re-encode
// the same texts each iter. For COLD-cache numbers use the test.
func BenchmarkEncodeBatch_rerankN50(b *testing.B) {
	if _, err := os.Stat(modelDir); errors.Is(err, fs.ErrNotExist) {
		b.Skipf("no model at %s", modelDir)
	}
	m, err := Load(modelDir)
	if err != nil {
		b.Fatal(err)
	}
	templates := []string{
		"def %s(x):\n    if x < 2:\n        return x\n    return %s(x-1) + %s(x-2)",
		"class %s:\n    def __init__(self, name):\n        self.name = name\n    def %s(self):\n        return self.name",
		"async def %s(url):\n    async with httpx.AsyncClient() as c:\n        r = await c.get(url)\n        return r.json()",
		"func %s(in []byte) ([]byte, error) {\n    if len(in) == 0 {\n        return nil, fmt.Errorf(\"empty\")\n    }\n    return in, nil\n}",
		"impl %s for Foo {\n    fn bar(&self) -> Result<()> {\n        self.inner.lock().unwrap().do_thing();\n        Ok(())\n    }\n}",
	}
	names := []string{"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
		"iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
		"rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
		"add", "sub", "mul", "div", "mod", "pow", "log", "exp",
		"sin", "cos", "tan", "asin", "acos", "atan", "min", "max",
		"sort", "reverse", "shuffle", "merge", "split", "concat", "filter", "map",
		"reduce", "fold"}
	N := 50
	texts := make([]string, N)
	isQ := make([]bool, N)
	for i := range N {
		tmpl := templates[i%len(templates)]
		name := names[i%len(names)]
		texts[i] = fmt.Sprintf(tmpl, name, name, name)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := m.EncodeBatch(texts, isQ, 0)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// TestEncodeBatch_rerankN50: the M0 reference workload — 50
// candidates of ~80 tokens, cold cache. This is the headline M3
// latency number for the GO/NO-GO call: how long does a fresh
// hybrid-rerank query take end-to-end? Single repetition; logs the
// wall time. Skipped under -short (~20s on an 8-core M1 Pro).
func TestEncodeBatch_rerankN50(t *testing.T) {
	if testing.Short() {
		t.Skip("rerankN=50 measurement runs ~20s; skipped under -short")
	}
	if _, err := os.Stat(modelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no model at %s", modelDir)
	}
	m, err := Load(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	// 50 distinct candidate chunks of ~80-100 tokens each — synthesized
	// from a single template per category, varied enough that the dedup
	// cache (which this test deliberately doesn't model) wouldn't help.
	templates := []string{
		"def %s(x):\n    if x < 2:\n        return x\n    return %s(x-1) + %s(x-2)",
		"class %s:\n    def __init__(self, name):\n        self.name = name\n    def %s(self):\n        return self.name",
		"async def %s(url):\n    async with httpx.AsyncClient() as c:\n        r = await c.get(url)\n        return r.json()",
		"func %s(in []byte) ([]byte, error) {\n    if len(in) == 0 {\n        return nil, fmt.Errorf(\"empty\")\n    }\n    return in, nil\n}",
		"impl %s for Foo {\n    fn bar(&self) -> Result<()> {\n        self.inner.lock().unwrap().do_thing();\n        Ok(())\n    }\n}",
	}
	names := []string{
		"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
		"iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
		"rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
		"add", "sub", "mul", "div", "mod", "pow", "log", "exp",
		"sin", "cos", "tan", "asin", "acos", "atan", "min", "max",
		"sort", "reverse", "shuffle", "merge", "split", "concat", "filter", "map",
		"reduce", "fold",
	}
	N := 50
	texts := make([]string, N)
	isQ := make([]bool, N)
	for i := range N {
		tmpl := templates[i%len(templates)]
		name := names[i%len(names)]
		texts[i] = fmt.Sprintf(tmpl, name, name, name)
	}

	t0 := time.Now()
	out, err := m.EncodeBatch(texts, isQ, 0) // NumCPU
	if err != nil {
		t.Fatal(err)
	}
	took := time.Since(t0)
	t.Logf("rerankN=%d cold, NumCPU=%d: %v wall (%.0fms/candidate amortized)",
		N, runtime.NumCPU(), took, float64(took.Milliseconds())/float64(N))
	if len(out) != N {
		t.Fatalf("got %d outputs, want %d", len(out), N)
	}
}
