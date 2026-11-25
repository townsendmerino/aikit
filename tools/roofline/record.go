package main

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// record is one reading, rendered as the committed markdown table.
type record struct {
	cpu, goos, goarch, commit, date string
	passes                          int
	benchtime                       string
	wall                            time.Duration
	roofs                           roofs
	rows                            []row
}

// The FMA probe's implied clock must land where a desktop or laptop core can run.
const minGHz, maxGHz = 0.8, 6.0

// inconclusive says why the reading cannot be read against its registration, or "".
func (r record) inconclusive() string {
	var why []string
	if r.roofs.fmaGFLOPS <= 0 {
		why = append(why, "no FMA peak was measured")
	} else if g := r.roofs.impliedGHz(); g < minGHz || g > maxGHz {
		why = append(why, fmt.Sprintf("the FMA peak implies %.2f GHz, outside %.1f–%.1f: the probe did not measure what it claims", g, minGHz, maxGHz))
	}
	if !r.roofs.streamPassed {
		why = append(why, fmt.Sprintf("the streaming probe failed its cross-check (resident %.1f GB/s, streamed %.1f: resident must be at least twice streamed)", r.roofs.residentGBs, r.roofs.streamGBs))
	}
	needInt8 := false
	for _, rw := range r.rows {
		if rw.medianNs <= 0 {
			why = append(why, rw.name+" was not measured")
		}
		needInt8 = needInt8 || rw.class == "int8"
	}
	if needInt8 && r.roofs.int8GMACs <= 0 {
		why = append(why, "no int8 dot roof was measured or cited (--int8-roof)")
	}
	return strings.Join(why, "; ")
}

// laggards counts the rows below their line, and the parked rows just under it.
func (r record) laggards() (laggards, parked int) {
	for _, rw := range r.rows {
		if rw.laggard {
			laggards++
		}
		if rw.parked {
			parked++
		}
	}
	return laggards, parked
}

// markdown renders the record, rows sorted lowest fraction first.
func (r record) markdown() string {
	rows := append([]row(nil), r.rows...)
	sortRows(rows)
	var b strings.Builder
	fmt.Fprintf(&b, "# Roofline census: %s, %s\n\n", r.cpu, r.date[:10])
	fmt.Fprintf(&b, "Written by `go run -C tools ./roofline` at aikit `%s`, %s/%s, %s. %d passes at `-benchtime %s`, median ns/op; wall time %s. Single-threaded. Read it against [`roofline-registration.md`](roofline-registration.md).\n\n",
		r.commit, r.goos, r.goarch, r.date, r.passes, r.benchtime, r.wall.Round(time.Second))
	b.WriteString("## Roofs (one core)\n\n| roof | value | how |\n|---|--:|---|\n")
	fmt.Fprintf(&b, "| f32 FMA peak | %.1f GFLOP/s (%.1f GMAC/s f32, %.1f f64; implies %.2f GHz) | measured: `measureFMAPeak` |\n",
		r.roofs.fmaGFLOPS, r.roofs.computeGMACs("f32"), r.roofs.computeGMACs("f64"), r.roofs.impliedGHz())
	if r.roofs.int8GMACs > 0 {
		fmt.Fprintf(&b, "| int8 dot peak | %.1f GMAC/s | %s |\n", r.roofs.int8GMACs, r.roofs.int8Source)
	} else {
		b.WriteString("| int8 dot peak | — | neither measured nor cited |\n")
	}
	fmt.Fprintf(&b, "| streaming read | %.1f GB/s (32 KiB: %.1f; cross-check %s) | measured: `bytes.IndexByte` over 512 MiB, best of 5 |\n\n",
		r.roofs.streamGBs, r.roofs.residentGBs, passWord(r.roofs.streamPassed))
	b.WriteString("## Rows, lowest fraction first\n\n")
	b.WriteString("Bytes are each row's operands counted once, not its traffic. A resident row is held to its class's compute roof; a streamed row to the lower of that and its MACs per byte times the streaming roof.\n\n")
	b.WriteString("| kernel | class | regime | shape | MiB | ns/op | GB/s | GMAC/s | binding | fraction | |\n|---|---|---|---|--:|--:|--:|--:|---|--:|---|\n")
	for _, rw := range rows {
		mark := ""
		switch {
		case rw.laggard:
			mark = "laggard"
		case rw.parked:
			mark = "parked"
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s | %.2f | %s | %s | %s | %s | %s | %s |\n",
			rw.label, rw.class, rw.regime, rw.shape, rw.bytes/(1<<20), num(rw.medianNs), num(rw.gbs), gmacCell(rw), rw.binding, fmtFrac(rw.fraction), mark)
	}
	return b.String()
}

func gmacCell(rw row) string {
	if rw.macs <= 0 {
		return "—"
	}
	return num(rw.gmacs)
}

func num(v float64) string {
	switch {
	case v <= 0 || math.IsNaN(v):
		return "—"
	case v >= 1000:
		return fmt.Sprintf("%.0f", v)
	case v >= 10:
		return fmt.Sprintf("%.1f", v)
	default:
		return fmt.Sprintf("%.3g", v)
	}
}
