package main

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
)

// family is one benchmark function perfgate judges. arches lists the GOARCH values whose build
// defines it (nil: every arch); on any other arch it is not expected, and on a listed one its
// absence fails the gate.
type family struct {
	name   string
	arches []string
}

func (f family) appliesTo(goarch string) bool {
	return f.arches == nil || slices.Contains(f.arches, goarch)
}

// defaultPackages are the packages perfgate builds in each arm. A family is found by its
// benchmark name in whichever of them defines it, so a family that moves between packages
// across the tag boundary is still compared. A package missing at the previous tag is skipped
// there.
var defaultPackages = []string{"./linalg", "./linalg/attn", "./linalg/elem", "./linalg/bitvec"}

// defaultFamilies is the gate's coverage. Each family is here because a regression in it would
// otherwise ship green; TestDefaultFamilies_resolveInSource holds the arches column to the
// source.
var defaultFamilies = []family{
	{name: "BenchmarkW8A8SpanShapes"},                                   // W8A8 M=1, resident and streamed
	{name: "BenchmarkGEMV_W8A8_baseline"},                               // W8A8 M=1 through the f32 entry
	{name: "BenchmarkMatmulBTW8A8Batch_prefill"},                        // W8A8 M>1, the batched tile
	{name: "BenchmarkW4A8_CanonicalVsSplitHalf", arches: amd64AndArm64}, // W4A8 M=1, the int4 decode kernel
	{name: "BenchmarkMatmulBTW4A8SplitHalfTile", arches: amd64Only},     // W4A8 M>1 on amd64 split-half-only
	{name: "BenchmarkMatmulQKAcc64"},                                    // the f64 attention kernels at depth
	{name: "BenchmarkMatmulAVAcc64"},
	{name: "BenchmarkMatmulQKAcc64F16"}, // the same over a binary16 cache
	{name: "BenchmarkMatmulAVAcc64F16"},
	{name: "BenchmarkAttendTileFused_expKind"},              // the fused ViT attention schedule
	{name: "BenchmarkSoftmaxRowKernels", arches: arm64Only}, // the S-06 contract transcendentals, per arch
	{name: "BenchmarkSiLUKernels", arches: arm64Only},
	{name: "BenchmarkExpF32KernelsAMD64", arches: amd64Only},
	{name: "BenchmarkSiLUAMD64", arches: amd64Only},
	{name: "BenchmarkQuantizeRowInt8"},                   // the activation quantiser
	{name: "BenchmarkQ56KvsInt8", arches: amd64Only},     // the Q5_K and Q6_K kinds, against int8
	{name: "BenchmarkQ4KvsSplitHalf", arches: amd64Only}, // the Q4_K kind, against split-half int4
}

var (
	amd64Only     = []string{"amd64"}
	arm64Only     = []string{"arm64"}
	amd64AndArm64 = []string{"amd64", "arm64"}
)

// familyNames returns the names of the families that apply to goarch, in table order.
func familyNames(fams []family, goarch string) []string {
	var out []string
	for _, f := range fams {
		if f.appliesTo(goarch) {
			out = append(out, f.name)
		}
	}
	return out
}

// parseFamilies reads --bench's value: benchmark names separated by commas, each applying to
// every arch.
func parseFamilies(s string) ([]family, error) {
	var out []family
	for name := range strings.SplitSeq(s, ",") {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		if !reBenchName.MatchString(name) {
			return nil, fmt.Errorf("%q is not a benchmark function name", name)
		}
		out = append(out, family{name: name})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no benchmark names in %q", s)
	}
	return out, nil
}

var reBenchName = regexp.MustCompile(`^Benchmark[A-Za-z0-9_]*$`)

// resolution is where one arm's families live: the families each package defines, in the order
// asked, and the families no package defines.
type resolution struct {
	byPkg   map[string][]string
	missing []string
}

// resolveFamilies finds each wanted family among the benchmarks each package defines (listed:
// package → benchmark function names). A family defined by more than one package is an error:
// its rows could not be told apart.
func resolveFamilies(wanted []string, listed map[string][]string) (resolution, error) {
	res := resolution{byPkg: map[string][]string{}}
	for _, name := range wanted {
		var in []string
		for pkg, benches := range listed {
			if slices.Contains(benches, name) {
				in = append(in, pkg)
			}
		}
		slices.Sort(in)
		switch len(in) {
		case 0:
			res.missing = append(res.missing, name)
		case 1:
			res.byPkg[in[0]] = append(res.byPkg[in[0]], name)
		default:
			return resolution{}, fmt.Errorf("%s is defined in more than one package (%s)", name, strings.Join(in, ", "))
		}
	}
	return res, nil
}

// benchRegex is the -test.bench pattern that runs exactly these top-level benchmarks and all
// their sub-benchmarks.
func benchRegex(names []string) string {
	q := make([]string, len(names))
	for i, n := range names {
		q[i] = regexp.QuoteMeta(n)
	}
	return "^(" + strings.Join(q, "|") + ")$"
}

// parseTestList reads a test binary's -test.list output: one name per line.
func parseTestList(out string) []string {
	var names []string
	for ln := range strings.SplitSeq(out, "\n") {
		if ln = strings.TrimSpace(ln); strings.HasPrefix(ln, "Benchmark") {
			names = append(names, ln)
		}
	}
	return names
}

// silentFamilies returns the families that ran in an arm but produced no row in any visit: a
// benchmark that skipped on this host. Rows are keyed by family first (parseBench).
func silentFamilies(names []string, visits []map[string]float64) []string {
	var out []string
	for _, name := range names {
		found := false
		for k := range keySet(visits) {
			if k == name || strings.HasPrefix(k, name+"/") {
				found = true
				break
			}
		}
		if !found {
			out = append(out, name)
		}
	}
	return out
}
