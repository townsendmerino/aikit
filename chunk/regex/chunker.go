// Package regex is the v1-default chunker (ken's DESIGN.md §2 Option C): one
// generic line-walking engine driven by per-language LanguageRules. It
// registers itself as "regex" via chunk.Register in init(); import it for
// side effects (internal/search does) — chunk must not import this package
// (import cycle), so registration is decoupled the database/sql way.
//
// This package moved here from ken (ADR-034); "DESIGN.md" refers to
// https://github.com/townsendmerino/ken/blob/main/docs/DESIGN.md.
//
// Algorithm (ken's DESIGN.md §2 "Build path"): walk lines, mark the start of each
// top-level (or member-level) definition as a candidate boundary, greedily
// accumulate lines into a chunk, and when the next line would exceed
// chunkSize snap the cut back to the latest candidate boundary that still
// fits. If a single definition is itself larger than chunkSize there is no
// boundary to snap to, so it is split at line boundaries (the line-chunker
// fallback rule). Chunks are a contiguous, non-overlapping partition, so
// concatenating Text in order reproduces the source byte-for-byte.
//
// This file holds the config types (LanguageRules and friends) and the
// public Chunker API (Chunk/chunkWith). The literal-prefix/first-byte
// prescreen precomputation lives in chunker_prescreen.go; the hand-rolled
// brace-depth state machine lives in chunker_scan.go.
package regex

import (
	"bytes"
	"regexp"
	"sort"

	"github.com/townsendmerino/aikit/chunk"
)

type strategy int

const (
	braceStrategy  strategy = iota // top-level ⇔ brace depth 0 (C-likes)
	indentStrategy                 // top-level ⇔ no leading whitespace (Python)
)

// scannerCfg tells the brace-depth scanner which literal/comment forms to
// skip so braces inside them do not perturb the depth count.
type scannerCfg struct {
	lineComment string // e.g. "//"
	dq          bool   // "double quoted" with \ escapes
	sq          bool   // 'single quoted' with \ escapes (char/rune/JS string)
	backtick    bool   // `raw / template` (Go raw, TS template)
	tripleQuote bool   // Java text blocks: """ ... """
	rustRaw     bool   // Rust raw strings: r"..." / r#"..."#
}

// LanguageRules is the per-language driver for the generic engine.
type LanguageRules struct {
	lang     string
	defs     []*regexp.Regexp // a (left-trimmed) line that starts a definition
	skip     []*regexp.Regexp // … unless it also matches one of these (control-flow lines that look defn-ish)
	attach   []*regexp.Regexp // lines that glue onto the following def (docs, annotations, attributes, decorators)
	strat    strategy
	maxDepth int        // brace strategy: a def is a boundary iff depthBefore ≤ maxDepth
	scan     scannerCfg // brace strategy only

	// Literal prefixes for the three rule sets above, one per regexp, filled by
	// register. An empty entry means "no prescreen available, run the regexp".
	defsPre, skipPre, attachPre [][]byte
	// First-byte sets, the weaker fallback screen for patterns with no literal
	// prefix (TypeScript's optional-group-led rules). One per regexp, nil when the
	// literal prefix already screens it or no useful set exists.
	defsFB, skipFB, attachFB []*[256]bool
}

// languageRules is populated by each per-language file's init().
var languageRules = map[string]LanguageRules{}

// Chunker implements chunk.Chunker.
type Chunker struct{ rules map[string]LanguageRules }

// New returns a Chunker over every registered language ruleset.
func New() *Chunker { return &Chunker{rules: languageRules} }

func init() { chunk.Register("regex", New()) }

func (*Chunker) Name() string { return "regex" }

func (c *Chunker) SupportedLanguages() []string {
	out := make([]string, 0, len(c.rules))
	for l := range c.rules {
		out = append(out, l)
	}
	sort.Strings(out)
	return out
}

func (c *Chunker) Chunk(source []byte, language string, chunkSize int) ([]chunk.Chunk, error) {
	if chunkSize <= 0 {
		chunkSize = chunk.DefaultChunkSize
	}
	r, ok := c.rules[language]
	if !ok {
		// Defensive: ChunkFile routes unsupported languages to the line
		// chunker, so this is rare. Degrade to size-only splitting (no
		// defs ⇒ no boundaries) — still a valid byte-exact partition.
		r = LanguageRules{lang: language, strat: braceStrategy, maxDepth: -1}
	}
	return chunkWith(r, source, chunkSize), nil
}

func chunkWith(r LanguageRules, src []byte, chunkSize int) []chunk.Chunk {
	if len(src) == 0 {
		return nil
	}

	// lineStart[i] = byte offset of line i. A trailing '\n' does not start
	// a phantom empty line (matches the Stage 1 line chunker).
	// Presized: the line count is one bytes.Count away, and that scan is an
	// assembly SIMD one — nearly free next to the 21 reallocating appends the
	// grow-by-append version paid on a 643 KB file (lens doc §4.8).
	lineStart := make([]int, 1, bytes.Count(src, []byte{'\n'})+1)
	for i := range src {
		if src[i] == '\n' && i+1 < len(src) {
			lineStart = append(lineStart, i+1)
		}
	}
	n := len(lineStart)
	off := func(k int) int {
		if k >= n {
			return len(src)
		}
		return lineStart[k]
	}
	rawLine := func(i int) []byte { return src[off(i):off(i+1)] }

	// Per-line "is this the start of a definition?" with attachment of the
	// preceding doc-comment / annotation / decorator block.
	var depth []int32
	if r.strat == braceStrategy {
		depth = scanDepth(src, lineStart, r.scan)
	}
	isBoundary := make([]bool, n)
	for i := range n {
		line := rawLine(i)
		var probe []byte
		switch r.strat {
		case indentStrategy:
			// Top-level only: a definition with no leading whitespace.
			if len(line) > 0 && (line[0] == ' ' || line[0] == '\t') {
				continue
			}
			probe = line
		default: // braceStrategy
			if r.maxDepth < 0 || int(depth[i]) > r.maxDepth {
				continue
			}
			probe = bytes.TrimLeft(line, " \t")
		}
		if !anyMatch(r.defs, r.defsPre, r.defsFB, probe) || anyMatch(r.skip, r.skipPre, r.skipFB, probe) {
			continue
		}
		// Snap the boundary up over a contiguous attach block (so a
		// function keeps its doc comment / @annotation / #[attr] / @decorator).
		b := i
		// b > 0, not b-1 >= 1: the latter halts at b==1, so an attach block
		// (license header, module doc) that starts at line 0 and runs into the
		// first definition marked isBoundary[1] — a line INSIDE the comment —
		// producing a single-line first chunk holding only line 0 (audit #22).
		for b > 0 && attachMatch(r, rawLine(b-1)) {
			b--
		}
		isBoundary[b] = true
	}

	boundaries := make([]int, 0, 16) // sorted line indices > 0 where a chunk may start
	for i := 1; i < n; i++ {
		if isBoundary[i] {
			boundaries = append(boundaries, i)
		}
	}
	// lastBoundaryLE: greatest boundary b with lo < b ≤ hi, else -1.
	lastBoundaryLE := func(lo, hi int) int {
		j := sort.Search(len(boundaries), func(k int) bool { return boundaries[k] > hi })
		if j == 0 {
			return -1
		}
		if b := boundaries[j-1]; b > lo {
			return b
		}
		return -1
	}

	// NOT DONE, deliberately: lens §4.8's third part proposes one `string(src)`
	// plus substrings in place of the K per-chunk copies — 430 allocations to 2,
	// latency-neutral, since the package guarantees a contiguous non-overlapping
	// partition so the copies already sum to len(src). It is left alone because
	// it changes RETENTION, not cost: every chunk would pin the whole source
	// string, which is right for index-everything and wrong for
	// filter-then-keep-a-few, and no test or benchmark here can tell those apart.
	// A caller holding three chunks of a 100 MB file would silently hold 100 MB.
	out := make([]chunk.Chunk, 0, len(src)/chunkSize+1)
	emit := func(a, b int) {
		out = append(out, chunk.Chunk{
			StartLine: a + 1,
			EndLine:   b,
			Text:      string(src[off(a):off(b)]),
		})
	}

	start := 0
	for start < n {
		end := start + 1
		for end < n && off(end+1)-off(start) <= chunkSize {
			end++
		}
		if end >= n {
			emit(start, n)
			break
		}
		if b := lastBoundaryLE(start, end); b > start {
			emit(start, b) // snap the cut back to a definition boundary
			start = b
			continue
		}
		emit(start, end) // oversized single unit ⇒ line-split fallback
		start = end
	}
	return out
}

// anyMatch reports whether line matches any of res, skipping the regexp engine
// for patterns whose literal prefix the line does not start with. See register
// for why that is exact rather than a heuristic.
func anyMatch(res []*regexp.Regexp, pre [][]byte, fb []*[256]bool, line []byte) bool {
	for i, re := range res {
		if p := pre[i]; len(p) > 0 {
			if !bytes.HasPrefix(line, p) {
				continue
			}
		} else if s := fb[i]; s != nil {
			// First-byte screen: an ^-anchored match starts at line[0], so a first
			// byte outside the set (or an empty line) provably cannot match.
			if len(line) == 0 || !s[line[0]] {
				continue
			}
		}
		if re.Match(line) {
			return true
		}
	}
	return false
}

// attachMatch reports whether line glues onto the following definition. A
// blank line never attaches (it separates a def from anything above it).
func attachMatch(r LanguageRules, line []byte) bool {
	probe := line
	if r.strat == braceStrategy {
		probe = bytes.TrimLeft(line, " \t")
	}
	if len(bytes.TrimSpace(probe)) == 0 {
		return false
	}
	return anyMatch(r.attach, r.attachPre, r.attachFB, probe)
}
