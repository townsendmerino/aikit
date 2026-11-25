package main

import (
	"cmp"
	"fmt"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// roofs are one box's ceilings, single core. A zero value is a roof that was not measured or
// cited; a row bound by it gets no fraction.
type roofs struct {
	fmaGFLOPS    float64 // the f32 FMA peak probe
	int8GMACs    float64 // the int8 dot peak: the probe where the arch has one, else --int8-roof
	int8Source   string  // where int8GMACs came from
	streamGBs    float64 // the streaming-read probe over 512 MiB
	residentGBs  float64 // the same probe over 32 KiB: the cross-check below
	streamPassed bool    // residentGBs ≥ 2·streamGBs: the probe is not instruction-bound
}

// fmaPeakFlopsPerCycle is the f32 flops a core retires per cycle at the FMA peak: two 8-lane YMM
// FMA pipes on the amd64 boxes, four 4-lane NEON pipes on the M1 Pro, two flops an FMA.
const fmaPeakFlopsPerCycle = 32

// impliedGHz is the clock the FMA probe implies; one outside a part's range means the probe did
// not measure what it claims.
func (r roofs) impliedGHz() float64 { return r.fmaGFLOPS / fmaPeakFlopsPerCycle }

// computeGMACs is the compute roof of a class in GMAC/s, or 0 when it has none.
func (r roofs) computeGMACs(class string) float64 {
	switch class {
	case "f32":
		return r.fmaGFLOPS / 2
	case "f64":
		return r.fmaGFLOPS / 4
	case "int8":
		return r.int8GMACs
	}
	return 0
}

// sample is one census row as measured: ns/op from each pass, and the counts it reported.
type sample struct {
	name         string // <class>/<label>/<regime>
	ns           []float64
	bytes, macs  float64
	shape        string
	class, label string
	regime       string
}

// row is one line of the table.
type row struct {
	sample
	medianNs float64
	gbs      float64 // operand bytes per ns
	gmacs    float64
	bound    float64 // the roof that binds, in GMAC/s for a MAC row and GB/s for a stream row
	binding  string  // "compute", "stream", or "n/a: <why>"
	fraction float64 // achieved ÷ bound; NaN when there is no bound
	laggard  bool    // below its line by more than the band
	parked   bool    // below its line by no more than the band
}

// laggardLines are the registered lines (docs/bench-records/roofline-registration.md): a
// streamed row below streamed of its binding roof is a laggard, and so is a resident row of a
// compute-dense kernel (at least denseAI MACs per operand byte) below resident. A row less than
// band below its line is parked instead: listed, not queued.
type laggardLines struct {
	streamed, resident, denseAI, band float64
}

// computeRow applies the roofline: a resident row is held to its class's compute roof; a
// streamed row to the lower of that and its arithmetic intensity times the streaming roof; a
// row with no MACs, or a class with no compute roof, to the streaming roof alone.
func computeRow(s sample, r roofs, lines laggardLines) row {
	out := row{sample: s, medianNs: median(s.ns), fraction: math.NaN()}
	if out.medianNs <= 0 {
		out.binding = "n/a: not measured"
		return out
	}
	out.gbs = s.bytes / out.medianNs
	out.gmacs = s.macs / out.medianNs
	comp := r.computeGMACs(s.class)
	ai := 0.0
	if s.bytes > 0 {
		ai = s.macs / s.bytes
	}
	switch {
	case s.regime == "resident" && s.macs > 0 && comp > 0:
		out.bound, out.binding = comp, "compute"
		out.fraction = out.gmacs / comp
	case s.regime == "resident" && s.macs > 0:
		out.binding = "n/a: no " + s.class + " compute roof"
	case s.regime == "resident":
		out.binding = "n/a: no cache roof measured"
	case !r.streamPassed:
		out.binding = "n/a: the streaming probe failed its cross-check"
	case s.macs > 0 && comp > 0:
		if mem := ai * r.streamGBs; mem < comp {
			out.bound, out.binding = mem, "stream"
		} else {
			out.bound, out.binding = comp, "compute"
		}
		out.fraction = out.gmacs / out.bound
	default:
		out.bound, out.binding = r.streamGBs, "stream"
		if s.macs > 0 {
			out.binding = "stream (no " + s.class + " compute roof)"
		}
		out.fraction = out.gbs / r.streamGBs
	}
	line := math.NaN()
	switch {
	case s.regime == "streamed":
		line = lines.streamed
	case s.regime == "resident" && ai >= lines.denseAI:
		line = lines.resident
	}
	if !math.IsNaN(out.fraction) && !math.IsNaN(line) && out.fraction < line {
		out.laggard = out.fraction < line-lines.band
		out.parked = !out.laggard
	}
	return out
}

func median(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	c := slices.Sorted(slices.Values(s))
	n := len(c)
	if n%2 == 1 {
		return c[n/2]
	}
	return (c[n/2-1] + c[n/2]) / 2
}

var (
	reRow    = regexp.MustCompile(`^(BenchmarkRoofline\w+/\S+?)(?:-\d+)?\s+\d+\s+([\d.e+]+) ns/op(.*)$`)
	reMetric = regexp.MustCompile(`([\d.e+]+) (\S+)`)
	reBench  = regexp.MustCompile(`^--- BENCH: (BenchmarkRoofline\w+/\S+?)(?:-\d+)?$`)
	reShape  = regexp.MustCompile(`^\s+\S+\.go:\d+: shape (\S+)$`)
)

// parsed is one benchmark run's output: each row's ns/op and metrics, each row's shape, and the
// cpu line.
type parsed struct {
	ns      map[string]float64
	metrics map[string]map[string]float64
	shapes  map[string]string
	cpu     string
	order   []string
}

func parseOutput(out string) parsed {
	p := parsed{ns: map[string]float64{}, metrics: map[string]map[string]float64{}, shapes: map[string]string{}}
	cur := ""
	for ln := range strings.SplitSeq(out, "\n") {
		ln = strings.TrimRight(ln, "\r")
		if c, ok := strings.CutPrefix(ln, "cpu: "); ok {
			p.cpu = strings.TrimSpace(c)
			continue
		}
		if g := reBench.FindStringSubmatch(ln); g != nil {
			cur = g[1]
			continue
		}
		if g := reShape.FindStringSubmatch(ln); g != nil && cur != "" {
			p.shapes[cur] = g[1]
			continue
		}
		g := reRow.FindStringSubmatch(strings.TrimSpace(ln))
		if g == nil {
			continue
		}
		ns, err := strconv.ParseFloat(g[2], 64)
		if err != nil {
			continue
		}
		if _, seen := p.ns[g[1]]; !seen {
			p.order = append(p.order, g[1])
		}
		p.ns[g[1]] = ns
		m := map[string]float64{}
		for _, mm := range reMetric.FindAllStringSubmatch(g[3], -1) {
			if v, err := strconv.ParseFloat(mm[1], 64); err == nil {
				m[mm[2]] = v
			}
		}
		p.metrics[g[1]] = m
	}
	return p
}

// samplesFrom merges census passes into one sample per row, in first-seen order.
func samplesFrom(passes []parsed) []sample {
	byName := map[string]*sample{}
	var order []string
	for _, p := range passes {
		for _, full := range p.order {
			name, ok := strings.CutPrefix(full, "BenchmarkRooflineCensus/")
			if !ok {
				continue
			}
			s := byName[name]
			if s == nil {
				parts := strings.Split(name, "/")
				if len(parts) != 3 {
					continue
				}
				s = &sample{name: name, class: parts[0], label: parts[1], regime: parts[2]}
				byName[name] = s
				order = append(order, name)
			}
			s.ns = append(s.ns, p.ns[full])
			s.bytes, s.macs = p.metrics[full]["rdB/op"], p.metrics[full]["MAC/op"]
			if sh := p.shapes[full]; sh != "" {
				s.shape = sh
			}
		}
	}
	out := make([]sample, 0, len(order))
	for _, n := range order {
		out = append(out, *byName[n])
	}
	return out
}

// roofsFrom reads BenchmarkRooflineRoofs' metrics; int8 falls back to a cited figure.
func roofsFrom(p parsed, int8Cited float64, int8Source string) roofs {
	var r roofs
	r.fmaGFLOPS = p.metrics["BenchmarkRooflineRoofs/fma"]["GFLOP/s"]
	if v := p.metrics["BenchmarkRooflineRoofs/int8"]["GMAC/s"]; v > 0 {
		r.int8GMACs, r.int8Source = v, "measured: the SDOT issue probe (sdotIssuePeak)"
	} else if int8Cited > 0 {
		r.int8GMACs, r.int8Source = int8Cited, "cited: "+int8Source
	}
	st := p.metrics["BenchmarkRooflineRoofs/stream"]
	r.streamGBs, r.residentGBs = st["GB/s"], st["residentGB/s"]
	r.streamPassed = r.streamGBs > 0 && r.residentGBs >= 2*r.streamGBs
	return r
}

// sortRows orders the table by fraction, lowest first, rows without one last: the laggards lead.
func sortRows(rows []row) {
	slices.SortStableFunc(rows, func(a, b row) int {
		an, bn := math.IsNaN(a.fraction), math.IsNaN(b.fraction)
		switch {
		case an && bn:
			return 0
		case an:
			return 1
		case bn:
			return -1
		}
		return cmp.Compare(a.fraction, b.fraction)
	})
}

func fmtFrac(f float64) string {
	if math.IsNaN(f) {
		return "—"
	}
	return fmt.Sprintf("%.0f%%", 100*f)
}
