package main

import (
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/tools/gpumod"
)

// A planted box: 128 GFLOP/s FMA (64 GMAC/s f32, 32 f64, implying 4.0 GHz), 64 GMAC/s int8,
// 20 GB/s streaming, 80 GB/s resident.
var planted = roofs{fmaGFLOPS: 128, int8GMACs: 64, int8Source: "planted", streamGBs: 20, residentGBs: 80, streamPassed: true}

func near(a, b float64) bool { return math.Abs(a-b) <= 1e-9*math.Max(1, math.Abs(b)) }

func TestComputeRow_planted(t *testing.T) {
	for _, tc := range []struct {
		name     string
		s        sample
		r        roofs
		gbs      float64
		gmacs    float64
		binding  string
		fraction float64 // NaN: none
		laggard  bool
	}{
		// 1e6 MACs in 31250 ns = 32 GMAC/s of a 64 f32 roof.
		{"resident f32 against the FMA roof", sample{class: "f32", regime: "resident", ns: []float64{31250}, bytes: 1e5, macs: 1e6},
			planted, 3.2, 32, "compute", 0.5, false},
		// AI = 10 MAC/B (dense): 12.8 GMAC/s of 64 is 20%, under the resident line.
		{"dense resident below its line", sample{class: "f32", regime: "resident", ns: []float64{78125}, bytes: 1e5, macs: 1e6},
			planted, 1.28, 12.8, "compute", 0.2, true},
		// AI = 1: the same 20% is not judged at resident.
		{"sparse resident is not judged", sample{class: "int8", regime: "resident", ns: []float64{78125}, bytes: 1e6, macs: 1e6},
			planted, 12.8, 12.8, "compute", 0.2, false},
		// int8 M=1, AI = 1 MAC/B: bound = min(64, 1·20) = 20 GMAC/s; 10 achieved is 50%, on the line.
		{"streamed int8 bound by the stream", sample{class: "int8", regime: "streamed", ns: []float64{1e8}, bytes: 1e9, macs: 1e9},
			planted, 10, 10, "stream", 0.5, false},
		// f64 at AI = 0.25 MAC/B: bound 5 GMAC/s; 2 achieved is 40%, below the streamed line.
		{"streamed f64 below the line", sample{class: "f64", regime: "streamed", ns: []float64{5e8}, bytes: 4e9, macs: 1e9},
			planted, 8, 2, "stream", 0.4, true},
		// 9.2 of a 20 GMAC/s bound is 46%: under 0.5 by less than the 0.05 band, so parked.
		{"streamed just under the line is parked", sample{class: "int8", regime: "streamed", ns: []float64{1e9 / 9.2}, bytes: 1e9, macs: 1e9},
			planted, 9.2, 9.2, "stream", 0.46, false},
		// AI = 32: min(64, 640) = 64, compute-bound; 48 achieved is 75%.
		{"streamed f32 bound by compute", sample{class: "f32", regime: "streamed", ns: []float64{1e6}, bytes: 1.5e6, macs: 4.8e7},
			planted, 1.5, 48, "compute", 0.75, false},
		{"stream class", sample{class: "stream", regime: "streamed", ns: []float64{1e8}, bytes: 1.5e9},
			planted, 15, 0, "stream", 0.75, false},
		{"stream class resident has no roof", sample{class: "stream", regime: "resident", ns: []float64{1e3}, bytes: 1e5},
			planted, 100, 0, "n/a: no cache roof measured", math.NaN(), false},
		{"no int8 roof at resident", sample{class: "int8", regime: "resident", ns: []float64{1e4}, bytes: 1e5, macs: 1e5},
			roofs{fmaGFLOPS: 128, streamGBs: 20, residentGBs: 80, streamPassed: true}, 10, 10, "n/a: no int8 compute roof", math.NaN(), false},
		{"no int8 roof streamed falls back to the stream", sample{class: "int8", regime: "streamed", ns: []float64{1e8}, bytes: 1e9, macs: 1e9},
			roofs{fmaGFLOPS: 128, streamGBs: 20, residentGBs: 80, streamPassed: true}, 10, 10, "stream (no int8 compute roof)", 0.5, false},
		{"a failed stream probe binds nothing streamed", sample{class: "int8", regime: "streamed", ns: []float64{1e8}, bytes: 1e9, macs: 1e9},
			roofs{fmaGFLOPS: 128, int8GMACs: 64, streamGBs: 20, residentGBs: 30}, 10, 10, "n/a: the streaming probe failed its cross-check", math.NaN(), false},
		{"median of three passes", sample{class: "f32", regime: "resident", ns: []float64{20000, 31250, 1e9}, bytes: 1e5, macs: 1e6},
			planted, 3.2, 32, "compute", 0.5, false},
	} {
		got := computeRow(tc.s, tc.r, registeredLines)
		if !near(got.gbs, tc.gbs) || !near(got.gmacs, tc.gmacs) || got.binding != tc.binding || got.laggard != tc.laggard {
			t.Errorf("%s: GB/s %v GMAC/s %v binding %q laggard %v; want %v %v %q %v",
				tc.name, got.gbs, got.gmacs, got.binding, got.laggard, tc.gbs, tc.gmacs, tc.binding, tc.laggard)
		}
		if wantParked := strings.HasSuffix(tc.name, "is parked"); got.parked != wantParked {
			t.Errorf("%s: parked %v, want %v", tc.name, got.parked, wantParked)
		}
		if math.IsNaN(tc.fraction) != math.IsNaN(got.fraction) || (!math.IsNaN(tc.fraction) && !near(got.fraction, tc.fraction)) {
			t.Errorf("%s: fraction %v, want %v", tc.name, got.fraction, tc.fraction)
		}
	}
}

func TestRoofs_classesAndClock(t *testing.T) {
	if g := planted.computeGMACs("f32"); g != 64 {
		t.Errorf("f32 roof %v, want 64", g)
	}
	if g := planted.computeGMACs("f64"); g != 32 {
		t.Errorf("f64 roof %v, want 32", g)
	}
	if g := planted.computeGMACs("int8"); g != 64 {
		t.Errorf("int8 roof %v, want 64", g)
	}
	if g := planted.computeGMACs("stream"); g != 0 {
		t.Errorf("stream class has a compute roof: %v", g)
	}
	if g := planted.impliedGHz(); g != 4 {
		t.Errorf("implied clock %v, want 4", g)
	}
}

const plantedRoofOut = `goos: linux
goarch: amd64
cpu: AMD Ryzen 7 3700X 8-Core Processor
BenchmarkRooflineRoofs/fma-16         	       1	  99792289 ns/op	       128.0 GFLOP/s
BenchmarkRooflineRoofs/int8-16
    roofline_census_test.go:9: no int8 dot peak probe on this arch or CPU
BenchmarkRooflineRoofs/stream-16      	       1	 565524733 ns/op	        20.00 GB/s	        80.00 residentGB/s
PASS
`

func plantedCensusOut(nsResident, nsStreamed string) string {
	return `cpu: AMD Ryzen 7 3700X 8-Core Processor
BenchmarkRooflineCensus/int8/MatmulBTW8A8Into_M1/resident-16 	   10000	     ` + nsResident + ` ns/op	   131072 MAC/op	   133120 rdB/op
--- BENCH: BenchmarkRooflineCensus/int8/MatmulBTW8A8Into_M1/resident-16
    roofline_census_test.go:355: shape M1_K512_N256
BenchmarkRooflineCensus/int8/MatmulBTW8A8Into_M1/streamed-16 	      10	     ` + nsStreamed + ` ns/op	134217728 MAC/op	134266880 rdB/op
--- BENCH: BenchmarkRooflineCensus/int8/MatmulBTW8A8Into_M1/streamed-16
    roofline_census_test.go:355: shape M1_K4096_N32768
PASS
`
}

func TestParse_plantedOutput(t *testing.T) {
	rf := roofsFrom(parseOutput(plantedRoofOut), 69, "docs/internal/roofline-2026-08.md §0")
	if rf.fmaGFLOPS != 128 || rf.streamGBs != 20 || rf.residentGBs != 80 || !rf.streamPassed {
		t.Errorf("roofs = %+v", rf)
	}
	if rf.int8GMACs != 69 || !strings.HasPrefix(rf.int8Source, "cited: ") {
		t.Errorf("int8 roof %v from %q, want the cited 69", rf.int8GMACs, rf.int8Source)
	}
	if rf := roofsFrom(parseOutput(plantedRoofOut), 0, ""); rf.int8GMACs != 0 {
		t.Errorf("an int8 roof appeared from nowhere: %v", rf.int8GMACs)
	}

	passes := []parsed{
		parseOutput(plantedCensusOut("2048", "13421772.8")),
		parseOutput(plantedCensusOut("4096", "6710886.4")),
		parseOutput(plantedCensusOut("1024", "1e9")),
	}
	ss := samplesFrom(passes)
	if len(ss) != 2 {
		t.Fatalf("samples = %+v", ss)
	}
	res, str := ss[0], ss[1]
	if res.class != "int8" || res.label != "MatmulBTW8A8Into_M1" || res.regime != "resident" || res.shape != "M1_K512_N256" {
		t.Errorf("resident sample = %+v", res)
	}
	if res.bytes != 133120 || res.macs != 131072 || len(res.ns) != 3 {
		t.Errorf("resident counts = %+v", res)
	}
	if passes[0].cpu != "AMD Ryzen 7 3700X 8-Core Processor" {
		t.Errorf("cpu = %q", passes[0].cpu)
	}
	// Median ns/op 2048: 131072 MACs / 2048 ns = 64 GMAC/s of the planted 64: 100%.
	if rw := computeRow(res, planted, registeredLines); !near(rw.fraction, 1) {
		t.Errorf("resident fraction = %v, want 1", rw.fraction)
	}
	// Median 13421772.8 ns: 134266880 B / ns = 10.004 GB/s of 20.
	if rw := computeRow(str, planted, registeredLines); rw.binding != "stream" || math.Abs(rw.fraction-0.5002) > 1e-3 {
		t.Errorf("streamed row = %q %v, want stream 0.5002", rw.binding, rw.fraction)
	}
}

func TestRecord_verdictAndMarkdown(t *testing.T) {
	rows := []row{
		computeRow(sample{name: "f64/MatmulQKAcc64/streamed", class: "f64", label: "MatmulQKAcc64", regime: "streamed", shape: "G1", ns: []float64{5e8}, bytes: 4e9, macs: 1e9}, planted, registeredLines),
		computeRow(sample{name: "f32/MatmulBT_M1/resident", class: "f32", label: "MatmulBT_M1", regime: "resident", shape: "M1", ns: []float64{31250}, bytes: 1e5, macs: 1e6}, planted, registeredLines),
		computeRow(sample{name: "stream/SiLUContractInto/resident", class: "stream", label: "SiLUContractInto", regime: "resident", shape: "n1", ns: []float64{1e3}, bytes: 1e5}, planted, registeredLines),
	}
	rec := record{cpu: "Planted CPU", goos: "linux", goarch: "amd64", commit: "abc1234", date: "2026-11-24T00:00:00Z", passes: 3, benchtime: "300ms", roofs: planted, rows: rows}
	if why := rec.inconclusive(); why != "" {
		t.Errorf("planted record inconclusive: %s", why)
	}
	if lag, park := rec.laggards(); lag != 1 || park != 0 {
		t.Errorf("laggards, parked = %d, %d; want 1, 0 (the streamed QK at 40%%)", lag, park)
	}
	md := rec.markdown()
	for _, want := range []string{"abc1234", "Planted CPU", "| f32 FMA peak | 128.0 GFLOP/s", "| MatmulQKAcc64 | f64 | streamed |", "40% | laggard |", "| 50% |"} {
		if !strings.Contains(md, want) {
			t.Errorf("markdown lacks %q:\n%s", want, md)
		}
	}
	if strings.Index(md, "| MatmulQKAcc64 |") > strings.Index(md, "| MatmulBT_M1 |") || strings.Index(md, "| MatmulBT_M1 |") > strings.Index(md, "| SiLUContractInto |") {
		t.Errorf("rows are not lowest fraction first, unbound last:\n%s", md)
	}

	for _, tc := range []struct {
		name string
		mut  func(*record)
		want string
	}{
		{"no FMA peak", func(r *record) { r.roofs.fmaGFLOPS = 0 }, "no FMA peak"},
		{"an impossible clock", func(r *record) { r.roofs.fmaGFLOPS = 400 }, "implies 12.50 GHz"},
		{"a failed stream probe", func(r *record) { r.roofs.streamPassed = false }, "cross-check"},
		{"an unmeasured row", func(r *record) { r.rows = append(r.rows, row{name: "f32/x/resident"}) }, "f32/x/resident was not measured"},
		{"int8 rows with no int8 roof", func(r *record) {
			r.roofs.int8GMACs = 0
			r.rows = append(r.rows, row{name: "int8/y/resident", class: "int8", medianNs: 1})
		}, "no int8 dot roof"},
	} {
		r := rec
		r.rows = append([]row(nil), rec.rows...)
		tc.mut(&r)
		if why := r.inconclusive(); !strings.Contains(why, tc.want) {
			t.Errorf("%s: inconclusive() = %q, want it to say %q", tc.name, why, tc.want)
		}
	}
}

func TestRun_rejectsBadArguments(t *testing.T) {
	for _, args := range [][]string{
		{"--passes", "0"}, {"--passes", "x"}, {"--int8-roof", "-1"}, {"--int8-roof", "69"}, {"--bogus"},
	} {
		var b strings.Builder
		if rc := run(args, &b); rc != 2 || !strings.Contains(b.String(), "VERDICT: INCONCLUSIVE") {
			t.Errorf("run(%v) = %d, %q; want 2 and an INCONCLUSIVE verdict", args, rc, b.String())
		}
	}
}

func TestSlug(t *testing.T) {
	if s := slug("AMD Ryzen 7 3700X 8-Core Processor"); s != "amd-ryzen-7-3700x-8-core-processor" {
		t.Errorf("slug = %q", s)
	}
	if s := slug(""); s != "unknown-cpu" {
		t.Errorf("empty slug = %q", s)
	}
}

// The census names every kernel this tool must cover: the ones perfgate's families exercise and
// the matmul, attention and elementwise kernels goinfer's kernelprint fingerprints.
func TestCensus_coversTheNamedKernels(t *testing.T) {
	root, err := gpumod.RepoRoot()
	if err != nil {
		t.Fatal(err)
	}
	src, err := os.ReadFile(filepath.Join(root, "linalg", "roofline_census_test.go"))
	if err != nil {
		t.Fatal(err)
	}
	labels := map[string]bool{}
	for _, m := range regexp.MustCompile(`e\("(\w+)", "([\w.]+)"|class: "(\w+)", label: "([\w.]+)"`).FindAllStringSubmatch(string(src), -1) {
		labels[m[2]+m[4]] = true
	}
	for _, want := range []string{
		"MatmulBT_M1", "MatmulBT_M32", "MatmulBTQ8Into_M1", "MatmulBTW8A8Into_M1", "MatmulBTW8A8Into_M32",
		"MatmulBTW4A8F16Into_M1", "WeightMat.int4_M1", "WeightMat.int4_M32",
		"WeightMat.q4k_M1", "WeightMat.q5k_M1", "WeightMat.q6k_M1",
		"MatmulQKAcc64", "MatmulAVAcc64", "MatmulQKAcc64F16", "MatmulAVAcc64F16", "MatmulQKAcc64Group_G4", "MatmulAVAcc64Group_G4",
		"AttendTileFused", "AttendTileFusedContractExp",
		"SoftmaxRowContractInto", "SiLUContractInto", "GELUContractInto", "QuantizeRowInt8",
		"F16ToF32Slice", "F32ToF16Slice", "DequantizeRowsInt8Into", "HammingRows",
	} {
		if !labels[want] {
			t.Errorf("the census has no row for %s (labels: %v)", want, labels)
		}
	}
}
