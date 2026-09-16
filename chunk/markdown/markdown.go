// Package markdown is ken's documentation-aware chunker, registered as
// "markdown". Targets `.md`, `.mdx`, and `.markdown` files; other files
// auto-fall-back to the line chunker so a mixed-content corpus (docs +
// code) routes cleanly when this chunker is selected at the corpus
// level.
//
// Algorithm (overview; details in scanner.go):
//
//  1. scanLines: one pass over source, classifying each line as heading,
//     fenced-code-fence, fenced-code-inside, frontmatter, table row,
//     table separator, list item, blank, or text. Fence and frontmatter
//     state are carried forward across lines so `#` inside a code block
//     is NOT treated as a heading.
//  2. Section pass: split into heading-bounded sections. A section
//     starts at file beginning OR at an ATX heading OR at a setext
//     heading (text line whose next line is `===`/`---`).
//  3. Size-bounded subdivision: a section larger than chunkSize is
//     subdivided at safe-split positions (blank lines that are NOT
//     inside an atomic block — fenced code, frontmatter, table, list).
//     Atomic blocks larger than chunkSize stay whole; splitting a code
//     fence or a table mid-way would corrupt the semantics retrieval
//     wants to surface.
//
// Load-bearing invariant: byte-fidelity. Concatenating Chunk.Text values
// for a single file reproduces the source byte-for-byte. Tests in
// markdown_test.go pin this across all 12 prompt-spec scenarios.
//
// No new third-party dependencies. The CommonMark/GFM corner cases we
// don't aim to handle (HTML blocks, footnotes, link reference
// definitions, def-list extensions) all degrade gracefully to lineText
// and either flow into the current section or trigger a paragraph split
// — they never corrupt fidelity.
package markdown

import (
	"bytes"
	"path"
	"strings"

	"github.com/townsendmerino/aikit/chunk"
)

// Chunker implements chunk.Chunker for Markdown documents.
type Chunker struct{}

// New returns a fresh chunker. Stateless; safe to share across goroutines.
func New() *Chunker { return &Chunker{} }

func init() { chunk.Register("markdown", New()) }

// Name is "markdown" (the chunker registry key).
func (*Chunker) Name() string { return "markdown" }

// SupportedLanguages returns the canonical language names this chunker
// claims. Only "markdown" — chunk.ChunkFile routes other languages to
// the line chunker via the SupportedLanguages-based fallback. The
// chunker also self-checks (via routesToLineFallback) so callers who
// invoke Chunk directly with a non-markdown language get the same
// fallback behavior.
func (*Chunker) SupportedLanguages() []string { return []string{"markdown"} }

// markdownExtensions are the file extensions for which the markdown
// chunker is preferred. Used by routesToLineFallback when language is
// empty (callers that don't set Chunk.File pre-stamping go through
// chunk.ChunkFile which does set it via Language(), so this is the
// belt-and-suspenders path for direct callers).
var markdownExtensions = map[string]bool{
	".md":       true,
	".mdx":      true,
	".markdown": true,
}

// Chunk produces chunks for a markdown source. For non-markdown
// languages (when called directly rather than via ChunkFile's
// SupportedLanguages dispatch), delegates to the registered "line"
// chunker for byte-fidelity-preserving fallback.
func (c *Chunker) Chunk(source []byte, language string, chunkSize int) ([]chunk.Chunk, error) {
	if len(source) == 0 {
		return nil, nil
	}
	if chunkSize <= 0 {
		chunkSize = chunk.DefaultChunkSize
	}
	if language != "" && language != "markdown" {
		return c.lineFallback(source, language, chunkSize)
	}
	return c.chunkMarkdown(source, chunkSize), nil
}

// lineFallback delegates to the registered "line" chunker for non-
// markdown languages. Matches the treesitter chunker's fallback shape.
func (c *Chunker) lineFallback(source []byte, language string, chunkSize int) ([]chunk.Chunk, error) {
	lc, err := chunk.Get("line")
	if err != nil {
		// chunk.init() registers "line" before any sub-package init runs,
		// so this branch is unreachable in practice. Return a single
		// whole-file chunk as a last-resort safety net.
		return []chunk.Chunk{wholeFileChunk(source)}, nil
	}
	return lc.Chunk(source, language, chunkSize)
}

var newlineByte = []byte{'\n'}

// chunkMarkdown is the core algorithm: scan → section → subdivide.
func (c *Chunker) chunkMarkdown(source []byte, chunkSize int) []chunk.Chunk {
	lines := scanLines(source)
	if len(lines) == 0 {
		return nil
	}

	// Identify section boundaries: indices where a new section starts.
	// Index 0 is always a boundary; ATX headings and setext headings start new sections.
	// A setext underline at line N promotes line N-1 (if lineText) to a heading.
	boundaries := make([]int, 1, 16)
	boundaries[0] = 0
	for i := 1; i < len(lines); i++ {
		ln := lines[i]
		if ln.kind == lineHeadingATX || (ln.kind == lineText && i+1 < len(lines) && lines[i+1].kind == lineSetextUnderline) {
			boundaries = append(boundaries, i)
		}
	}

	// For each section [boundaries[k], boundaries[k+1]), produce chunks.
	out := make([]chunk.Chunk, 0, len(boundaries))
	for k, startIdx := range boundaries {
		endIdx := len(lines)
		if k+1 < len(boundaries) {
			endIdx = boundaries[k+1]
		}
		sectionLines := lines[startIdx:endIdx]
		sectionBytes := source[sectionLines[0].start:sectionLines[len(sectionLines)-1].end]
		// lines are newline-split, so a line's 1-based number IS its index into
		// `lines` + 1 — no from-byte-0 rescan (M5: lineNumber was O(N²/chunkSize)).
		startLine := startIdx + 1
		if len(sectionBytes) <= chunkSize {
			out = append(out, makeChunk(source, sectionLines[0].start, sectionLines[len(sectionLines)-1].end, startLine))
			continue
		}
		out = append(out, subdivideSection(source, sectionLines, chunkSize, startLine)...)
	}
	return out
}

// subdivideSection splits an oversized section at safe-split positions.
// A "safe split" is a blank line that is NOT inside an atomic block
// (fenced code, frontmatter, table, list). Atomic blocks larger than
// chunkSize stay whole (splitting one mid-fence would corrupt the
// content), which means a single chunk can legitimately exceed
// chunkSize — the size target is best-effort under that invariant.
// baseLine is the 1-based line number of lines[0] in the full source, so a
// section-relative index j maps to the absolute line baseLine+j (M5: replaces
// the per-chunk from-byte-0 lineNumber rescan with O(1) index arithmetic).
func subdivideSection(source []byte, lines []scannedLine, chunkSize int, baseLine int) []chunk.Chunk {
	if len(lines) == 0 {
		return nil
	}

	// prevSplit[i] = the greatest index s ≤ i with splittable[s], else -1. This
	// O(L) prefix pass replaces the per-i backward rescan below: when a window had
	// no splittable line (a section whose blank lines are all inside one big fenced
	// block — a generated artifact in a docs corpus), chunkStartIdx didn't advance
	// and the next i rescanned the whole range, making it Σi ≈ L²/2 (audit #7).
	// prevSplit[i] > chunkStartIdx is exactly the old loop's "greatest splittable in
	// (chunkStartIdx, i]" result.
	var prevSplitBuf [128]int
	var prevSplit []int
	if len(lines) <= len(prevSplitBuf) {
		prevSplit = prevSplitBuf[:len(lines)]
	} else {
		prevSplit = make([]int, len(lines))
	}
	last := -1
	for i, ln := range lines {
		if ln.kind == lineBlank {
			last = i
		}
		prevSplit[i] = last
	}

	var out []chunk.Chunk
	chunkStartIdx := 0
	for i := 1; i < len(lines); i++ {
		// Bytes accumulated from lines[chunkStartIdx..i] inclusive.
		accumBytes := lines[i].end - lines[chunkStartIdx].start
		if accumBytes >= chunkSize && i > chunkStartIdx {
			splitIdx := prevSplit[i]
			if splitIdx > chunkStartIdx {
				// Emit [chunkStartIdx, splitIdx) — splitIdx is a blank
				// line; we attach blanks to the chunk that came before
				// for byte fidelity.
				out = append(out, makeChunk(source,
					lines[chunkStartIdx].start,
					lines[splitIdx].end,
					baseLine+chunkStartIdx))
				chunkStartIdx = splitIdx + 1
			}
			// If no safe split exists in the window, we have to keep
			// accumulating — an atomic block bigger than chunkSize
			// legitimately overflows.
		}
	}
	// Flush remaining lines.
	if chunkStartIdx < len(lines) {
		out = append(out, makeChunk(source,
			lines[chunkStartIdx].start,
			lines[len(lines)-1].end,
			baseLine+chunkStartIdx))
	}
	return out
}

// makeChunk constructs a single chunk.Chunk from a byte range. The
// File field is left empty (chunk.ChunkFile stamps it). EndLine is
// derived from the chunk's last byte: a chunk ending with '\n' spans
// N lines, not N+1, so the trailing-newline correction is applied.
func makeChunk(source []byte, byteStart, byteEnd, startLine int) chunk.Chunk {
	text := source[byteStart:byteEnd]
	endLine := startLine
	n := len(text)
	if n > 0 && text[n-1] == '\n' {
		endLine += bytes.Count(text[:n-1], newlineByte)
	} else if n > 0 {
		endLine += bytes.Count(text, newlineByte)
	}
	// If the chunk ends with a newline, the "last line" count is correct
	// as-is; if not, we have a partial trailing line that still counts.
	return chunk.Chunk{
		StartLine: startLine,
		EndLine:   endLine,
		Text:      string(text),
	}
}

// wholeFileChunk is a last-resort fallback if even the line chunker
// can't be looked up (registry-init pathology — should not happen).
func wholeFileChunk(source []byte) chunk.Chunk {
	endLine := 1
	for _, b := range source {
		if b == '\n' {
			endLine++
		}
	}
	if len(source) > 0 && source[len(source)-1] == '\n' && endLine > 1 {
		endLine--
	}
	return chunk.Chunk{StartLine: 1, EndLine: endLine, Text: string(source)}
}

// IsMarkdownPath reports whether the path has a markdown file extension.
// Exposed so external callers (e.g. cmd/ken-mcp-docs) can decide whether
// to set ChunkerName=markdown for their corpus. Recognizes .md, .mdx,
// and .markdown (case-insensitive).
func IsMarkdownPath(p string) bool {
	ext := strings.ToLower(path.Ext(p))
	return markdownExtensions[ext]
}
