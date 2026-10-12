package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const apidiffSample = `Incompatible changes:
- Old: removed
Compatible changes:
- WhisperDevice: added
- WhisperLayerWeights: added
- WhisperWeights: added
- (*WhisperEncoder).SetDevice: added
- (*WhisperEncoder).Weights: added
- Tensor.SubF32: added
`

func TestParseAPIDiff_readsBothSections(t *testing.T) {
	cs := parseAPIDiff(apidiffSample)
	if len(cs) != 7 {
		t.Fatalf("parsed %d changes, want 7: %+v", len(cs), cs)
	}
	if cs[0].Section != "Incompatible changes" || cs[0].Symbol != "Old" || cs[0].What != "removed" {
		t.Errorf("first change = %+v", cs[0])
	}
	if cs[5].Section != "Compatible changes" || cs[5].Symbol != "(*WhisperEncoder).Weights" {
		t.Errorf("sixth change = %+v", cs[5])
	}
	if got := parseAPIDiff("- stray: added\n"); len(got) != 0 {
		t.Errorf("a line before any section heading was read: %+v", got)
	}
}

func TestSplitSymbol(t *testing.T) {
	for in, want := range map[string][2]string{
		"Foo":                         {"", "Foo"},
		"(*WhisperEncoder).SetDevice": {"WhisperEncoder", "SetDevice"},
		"Tensor.SubF32":               {"Tensor", "SubF32"},
	} {
		if typ, m := splitSymbol(in); typ != want[0] || m != want[1] {
			t.Errorf("splitSymbol(%q) = %q, %q; want %q, %q", in, typ, m, want[0], want[1])
		}
	}
}

func noWaivers(pkg, sym string) bool { return false }

// Every added symbol named: nothing owed. Dropping one mention owes exactly that symbol.
func TestUncoveredAdditions(t *testing.T) {
	changes := parseAPIDiff(apidiffSample)
	words := map[string]string{ // the symbol owed when its word is not in the section
		"WhisperDevice": "audio WhisperDevice", "WhisperWeights": "audio WhisperWeights", "WhisperLayerWeights": "audio WhisperLayerWeights",
		"SetDevice": "audio (*WhisperEncoder).SetDevice", "Weights": "audio (*WhisperEncoder).Weights", "SubF32": "audio Tensor.SubF32",
	}
	section := func(except string) string {
		var parts []string
		for w := range words {
			if w != except {
				parts = append(parts, "`"+w+"`")
			}
		}
		return strings.Join(parts, ", ")
	}
	if got := uncoveredAdditions("audio", changes, section(""), noWaivers); len(got) != 0 {
		t.Fatalf("every symbol is named, yet owed: %v", got)
	}
	for w, owed := range words {
		got := uncoveredAdditions("audio", changes, section(w), noWaivers)
		if len(got) != 1 || got[0] != owed {
			t.Errorf("without %q the owed set is %v, want exactly [%s]", w, got, owed)
		}
	}
}

// A word is a whole word: `Weights` inside `WhisperWeights` is not a mention of `Weights`.
func TestMentions_wholeWordsOnly(t *testing.T) {
	if mentions("only WhisperWeights here", "Weights") {
		t.Error("`Weights` was found inside `WhisperWeights`")
	}
	if !mentions("the (*E).Weights method", "Weights") || !mentions("`Weights`", "Weights") || !mentions("Weights", "Weights") {
		t.Error("a whole-word mention was missed")
	}
}

// A new type's members are described with the type; a member of an existing type is not covered by the type's name.
func TestUncoveredAdditions_newTypeCoversItsMembers(t *testing.T) {
	changes := parseAPIDiff("Compatible changes:\n- Brand: added\n- Brand.Run: added\n- Old.Extra: added\n")
	got := uncoveredAdditions("p", changes, "A new `Brand` type, and Old.", noWaivers)
	if len(got) != 1 || got[0] != "p Old.Extra" {
		t.Errorf("owed = %v, want only p Old.Extra (Brand.Run rides with the new type Brand; Old is not new)", got)
	}
}

func TestWaivers_needAReason(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "good.txt")
	if err := os.WriteFile(good, []byte("# comment\naudio Foo.Bar a test hook, not release-note material\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	w, err := loadAPIWaivers(good)
	if err != nil || w["audio Foo.Bar"] == "" {
		t.Fatalf("waivers = %v, %v", w, err)
	}
	bad := filepath.Join(dir, "bad.txt")
	if err := os.WriteFile(bad, []byte("audio Foo.Bar\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadAPIWaivers(bad); err == nil {
		t.Error("a waiver with no reason was accepted")
	}
	if w, err := loadAPIWaivers(filepath.Join(dir, "none.txt")); err != nil || w != nil {
		t.Errorf("a missing waiver file is none, not an error: %v %v", w, err)
	}
	changes := parseAPIDiff("Compatible changes:\n- Foo.Bar: added\n")
	waived := func(pkg, sym string) bool { _, ok := w[pkg+" "+sym]; return ok }
	if got := uncoveredAdditions("audio", changes, "nothing", waived); len(got) != 0 {
		t.Errorf("a waived symbol is still owed: %v", got)
	}
}

// A name ending in ForTest is a test seam and is not owed a line (v1.62.0's VoxtralDefectCountForTest was the one real release the check flagged).
func TestUncoveredAdditions_testSeamsAreExempt(t *testing.T) {
	changes := parseAPIDiff("Compatible changes:\n- VoxtralDefectCountForTest: added\n- (*Enc).SetDefectForTest: added\n- Real: added\n")
	got := uncoveredAdditions("audio", changes, "nothing", noWaivers)
	if len(got) != 1 || got[0] != "audio Real" {
		t.Errorf("owed = %v, want only audio Real", got)
	}
}
