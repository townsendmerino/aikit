package kernreg

import (
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/internal/guardmem"
)

var goodRows = []Row{
	{Symbol: "sumK", Arch: "amd64", Dispatch: "Sum", Oracle: "TestSum"},
	{Symbol: "probe", Arch: "amd64", NoSlices: "a CPU probe"},
	{Symbol: "sumK", Arch: "arm64", Dispatch: "Sum", Oracle: "TestSum"},
	{Symbol: "armK", Arch: "amd64", Undispatched: "a measurement arm", Oracle: "TestArm"},
}

var goodAdapters = []Adapter{{Symbol: "sumK"}, {Symbol: "armK"}}

func TestProblems_cleanRegistry(t *testing.T) {
	for arch, adapters := range map[string][]Adapter{"amd64": goodAdapters, "arm64": goodAdapters[:1]} {
		probs, err := Problems("testdata/pkg", arch, goodRows, adapters)
		if err != nil {
			t.Fatal(err)
		}
		if len(probs) != 0 {
			t.Fatalf("%s: problems on a complete registry: %q", arch, probs)
		}
	}
}

// Break-it-first: each way a registry can drift from the tree is reported.
func TestProblems_reportsEachDrift(t *testing.T) {
	cases := []struct {
		name     string
		rows     []Row
		adapters []Adapter
		want     string
	}{
		{"missing row", goodRows[1:], goodAdapters, "amd64/sumK (k_amd64.s): a TEXT symbol with no row"},
		{"stale row", append(slices.Clone(goodRows), Row{Symbol: "gone", Arch: "amd64", NoSlices: "x"}), goodAdapters, "amd64/gone: a row with no TEXT symbol"},
		{"duplicate row", append(slices.Clone(goodRows), goodRows[0]), goodAdapters, "amd64/sumK: two rows"},
		{"wrong dispatcher", replace(goodRows, 0, Row{Symbol: "sumK", Arch: "amd64", Dispatch: "Other", Oracle: "TestSum"}), goodAdapters, "Dispatch Other is not a non-test function"},
		{"missing oracle", replace(goodRows, 0, Row{Symbol: "sumK", Arch: "amd64", Dispatch: "Sum", Oracle: "TestNope"}), goodAdapters, "Oracle TestNope is not a Test function"},
		{"unrelated oracle", replace(goodRows, 0, Row{Symbol: "sumK", Arch: "amd64", Dispatch: "Sum", Oracle: "TestUnrelated"}), goodAdapters, "Oracle TestUnrelated reaches sumK neither directly"},
		{"no adapter", goodRows, nil, "amd64/sumK: no adapter"},
		{"two adapters", goodRows, append(slices.Clone(goodAdapters), Adapter{Symbol: "sumK"}), "amd64/sumK: 2 adapters"},
		{"adapter without row", goodRows, append(slices.Clone(goodAdapters), Adapter{Symbol: "extra"}), "amd64/extra: an adapter with no row"},
		{"adapter for a NoSlices row", goodRows, append(slices.Clone(goodAdapters), Adapter{Symbol: "probe"}), "amd64/probe: has an adapter and a NoSlices reason"},
		{"no dispatch", replace(goodRows, 0, Row{Symbol: "sumK", Arch: "amd64", Oracle: "TestSum"}), goodAdapters, "amd64/sumK: no Dispatch and no Undispatched reason"},
		{"undispatched but called", replace(goodRows, 0, Row{Symbol: "sumK", Arch: "amd64", Undispatched: "x", Oracle: "TestSum"}), goodAdapters, "amd64/sumK: listed Undispatched, but"},
	}
	for _, c := range cases {
		probs, err := Problems("testdata/pkg", "amd64", c.rows, c.adapters)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.ContainsFunc(probs, func(p string) bool { return strings.Contains(p, c.want) }) {
			t.Errorf("%s: problems %q do not include %q", c.name, probs, c.want)
		}
	}
}

func replace(rows []Row, i int, r Row) []Row {
	out := slices.Clone(rows)
	out[i] = r
	return out
}

func TestSameBits(t *testing.T) {
	nan := float32(0)
	nan = nan / nan
	if !sameBits([]float32{nan, 1}, []float32{nan, 1}) {
		t.Error("the same NaN compared unequal")
	}
	negZero := float32(0)
	negZero = -negZero
	if sameBits([]float32{0}, []float32{negZero}) {
		t.Error("+0 and -0 compared equal")
	}
	type hidden struct {
		n int
		f []float32
		m map[string]int
	}
	if !sameBits(hidden{3, []float32{1}, map[string]int{"a": 1, "b": 2}}, hidden{3, []float32{1}, map[string]int{"b": 2, "a": 1}}) {
		t.Error("equal structs with unexported fields compared unequal")
	}
	if sameBits(hidden{n: 3}, hidden{n: 4}) {
		t.Error("structs differing in an unexported field compared equal")
	}
	if sameBits([]int8{1, 2}, []int8{1, 2, 0}) || sameBits([]int32{1}, []int32{2}) {
		t.Error("different results compared equal")
	}
	if !sameBits(struct {
		A [2]float32
		B int32
	}{[2]float32{1, 2}, 3}, struct {
		A [2]float32
		B int32
	}{[2]float32{1, 2}, 3}) {
		t.Error("equal structs compared unequal")
	}
}

// sumPast is a fake kernel: it sums x[0:n+over] with no bounds check.
func sumPast(x []float32, n, over int) float32 {
	var s float32
	for _, v := range unsafe.Slice(unsafe.SliceData(x), n+over) {
		s += v
	}
	return s
}

func inputs(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i)
	}
	return x
}

func TestSweep_cleanKernelPasses(t *testing.T) {
	Sweep(t, []Adapter{{
		Symbol:  "sumExact",
		Lengths: []int{1, 7, 8, 9, 1024, 1025},
		Call:    func(p *Placer, n int) any { return sumPast(Put(p, "x", inputs(n)), n, 0) },
	}})
}

// childAdapters are the faulty kernels the break-it-first tests run in a child process,
// where a failing Sweep is the expected outcome.
var childAdapters = map[string]Adapter{
	"overread": {
		Symbol:  "sumPastEnd",
		Lengths: []int{5},
		Call:    func(p *Placer, n int) any { return sumPast(Put(p, "x", inputs(n)), n, 1) },
	},
	"placement": {
		Symbol:  "addressDependent",
		Lengths: []int{3},
		Call: func(p *Placer, n int) any {
			x := Put(p, "x", inputs(n))
			return uintptr(unsafe.Pointer(&x[0])) % 4096 // differs between heap and guard placement
		},
	},
}

func TestSweepChild(t *testing.T) {
	ad, ok := childAdapters[os.Getenv("KERNREG_CHILD")]
	if !ok {
		t.Skip("run only as the child of TestSweep_reportsFaultyKernels")
	}
	Sweep(t, []Adapter{ad})
}

// Break-it-first: a kernel that reads one element past its input, and one whose result
// depends on where its input sits, each fail the sweep with a message naming the kernel.
func TestSweep_reportsFaultyKernels(t *testing.T) {
	if !guardmem.Supported {
		t.Skip("guard pages are mapped on linux and darwin only: the faulty kernels cannot fault here")
	}
	for name, want := range map[string]string{
		"overread":  "sumPastEnd n=5 (inputs at the end guard): read outside input x: len 5 float32, byte 0 past its end",
		"placement": "addressDependent n=3, inputs at the end guard: result differs from the heap call",
	} {
		cmd := exec.Command(os.Args[0], "-test.run=^TestSweepChild$", "-test.v")
		cmd.Env = append(os.Environ(), "KERNREG_CHILD="+name)
		out, err := cmd.CombinedOutput()
		if err == nil {
			t.Errorf("%s: the child's sweep passed\n%s", name, out)
			continue
		}
		if !strings.Contains(string(out), want) {
			t.Errorf("%s: child output does not contain %q\n%s", name, want, out)
		}
	}
}

// A dispatcher must build for its row's architecture: the arm64 row may not borrow an
// amd64-only caller.
func TestProblems_dispatcherOnTheRowsArch(t *testing.T) {
	rows := replace(goodRows, 2, Row{Symbol: "sumK", Arch: "arm64", Dispatch: "SumAMD64", Oracle: "TestSum"})
	probs, err := Problems("testdata/pkg", "arm64", rows, goodAdapters)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.ContainsFunc(probs, func(p string) bool { return strings.Contains(p, "arm64/sumK: Dispatch SumAMD64 is not") }) {
		t.Fatalf("problems %q do not reject an amd64-only dispatcher on an arm64 row", probs)
	}
	rows = replace(goodRows, 0, Row{Symbol: "sumK", Arch: "amd64", Dispatch: "SumAMD64", Oracle: "TestSum"})
	if probs, _ := Problems("testdata/pkg", "amd64", rows, goodAdapters); slices.ContainsFunc(probs, func(p string) bool { return strings.Contains(p, "Dispatch") }) {
		t.Fatalf("an amd64 dispatcher on an amd64 row was rejected: %q", probs)
	}
}

func TestExported(t *testing.T) {
	got, err := Exported("testdata/pkg")
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"Config", "Mat.Rows", "Sum", "SumAMD64", "Wrapper"}; !slices.Equal(got, want) {
		t.Fatalf("Exported = %q, want %q", got, want)
	}
}

// Break-it-first: an uncovered, doubly covered or stale name is reported.
func TestExportedProblems(t *testing.T) {
	ok := map[string]string{"Config": "no shape"}
	swept := []string{"Mat.Rows", "Sum", "SumAMD64", "Wrapper"}
	if probs, err := ExportedProblems("testdata/pkg", swept, ok); err != nil || len(probs) != 0 {
		t.Fatalf("clean coverage: %q, %v", probs, err)
	}
	for name, c := range map[string]struct {
		swept  []string
		exempt map[string]string
		want   string
	}{
		"uncovered": {swept[1:], ok, "Mat.Rows: an exported entry point with no tiny-shape adapter"},
		"both":      {append(slices.Clone(swept), "Config"), ok, "Config: has an adapter and a listed reason"},
		"twice":     {append(slices.Clone(swept), "Sum"), ok, "Sum: 2 adapters"},
		"stale":     {append(slices.Clone(swept), "Gone"), ok, "Gone: listed but not an exported name"},
	} {
		probs, err := ExportedProblems("testdata/pkg", c.swept, c.exempt)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.ContainsFunc(probs, func(p string) bool { return strings.Contains(p, c.want) }) {
			t.Errorf("%s: problems %q do not include %q", name, probs, c.want)
		}
	}
}

func TestRefused(t *testing.T) {
	if msg := Refused(func() {}); msg != "" {
		t.Fatalf("a call that returned reads as refused: %q", msg)
	}
	if msg := Refused(func() { panic("linalg: K must be a multiple of 4") }); msg != "refused: linalg: K must be a multiple of 4" {
		t.Fatalf("an explicit panic reads %q", msg)
	}
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("a runtime error was swallowed as a refusal")
		}
	}()
	var x []int
	i := 3
	Refused(func() { _ = x[i] })
}

// An oracle may reach its dispatcher through a test table or helper it names.
func TestProblems_oracleThroughHelperOrTable(t *testing.T) {
	for _, oracle := range []string{"TestViaTable", "TestViaHelper", "TestViaWrapper"} {
		rows := replace(goodRows, 0, Row{Symbol: "sumK", Arch: "amd64", Dispatch: "Sum", Oracle: oracle})
		probs, err := Problems("testdata/pkg", "amd64", rows, goodAdapters)
		if err != nil {
			t.Fatal(err)
		}
		if len(probs) != 0 {
			t.Errorf("%s: %q", oracle, probs)
		}
	}
}
