package treesitter

import (
	"strings"
	"testing"

	"github.com/odvcencio/gotreesitter"
	"github.com/odvcencio/gotreesitter/grammars"
)

// TestParserPoolTimeout_stoppedEarlyContract (review2 F4) pins the timeout
// contract the chunker's runaway-parse guard depends on across the gotreesitter
// 0.20→0.40 bump — invisible from the call sites, and which the bump silently
// changed:
//
//   - 0.20: pool.Parse returned a non-nil ERROR on timeout.
//   - 0.40: pool.Parse returns a TRUNCATED tree with a NIL error and
//     tree.ParseStoppedEarly() == true.
//
// Chunk() now detects the early stop via ParseStoppedEarly() (relying on the
// error alone would silently chunk a partial tree and disable the guard). This
// test fails loudly if a future bump changes that signal.
func TestParserPoolTimeout_stoppedEarlyContract(t *testing.T) {
	entry := grammars.DetectLanguageByName("go")
	if entry == nil {
		t.Skip("gotreesitter has no \"go\" grammar")
	}
	lang := entry.Language()

	var b strings.Builder
	b.WriteString("package big\n")
	for i := range 12000 {
		b.WriteString("func f")
		b.WriteString(itoa(i))
		b.WriteString("(a, b int) int { return a + b }\n")
	}
	src := []byte(b.String())

	// A 1 µs budget can't finish a ~500 KB parse → truncated tree, nil error,
	// ParseStoppedEarly true. (Confirms the timeout still bounds the parse AND
	// that relying on the error would miss it — the exact regression F4 caught.)
	tight, err := gotreesitter.NewParserPool(lang, gotreesitter.WithParserPoolTimeoutMicros(1)).Parse(src)
	if err != nil {
		t.Fatalf("tight parse returned a hard error %v (contract expects nil + ParseStoppedEarly)", err)
	}
	if tight == nil || !tight.ParseStoppedEarly() {
		t.Errorf("1 µs budget: ParseStoppedEarly=%v, want true (timeout no longer detectable → chunker guard broken)",
			tight != nil && tight.ParseStoppedEarly())
	}

	// A generous budget parses the same source completely → ParseStoppedEarly
	// false (no false positive; a completed parse isn't mistaken for a timeout).
	// Uses 60 s rather than the chunker's real 1 s budget so the -race build,
	// which slows the parse ~10×, doesn't legitimately time out and flake.
	full, err := gotreesitter.NewParserPool(lang, gotreesitter.WithParserPoolTimeoutMicros(60_000_000)).Parse(src)
	if err != nil || full == nil {
		t.Fatalf("generous budget failed: tree=%v err=%v", full != nil, err)
	}
	if full.ParseStoppedEarly() {
		t.Error("generous budget: ParseStoppedEarly=true on a completed parse (false positive → every file would fall back)")
	}
}

// itoa avoids importing strconv for one call.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var d [8]byte
	i := len(d)
	for n > 0 {
		i--
		d[i] = byte('0' + n%10)
		n /= 10
	}
	return string(d[i:])
}
