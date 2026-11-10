package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestKenUses(t *testing.T) {
	dir := t.TempDir()
	write := func(rel, body string) {
		p := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("a/a.go", "package a\nimport (\n\t\"github.com/townsendmerino/aikit/ann\"\n\t\"github.com/townsendmerino/aikit/embed\"\n)\nvar _ = ann.New\nvar m *embed.StaticModel\nvar _ = embed.LoadGGUF\n")
	write("a/a_test.go", "package a\nimport \"github.com/townsendmerino/aikit/linalg\"\nvar _ = linalg.Dot\n")
	write("b/b.go", "package b\nimport \"embed\"\nvar _ embed.FS\n")                       // the stdlib's embed: not ours
	write(".cache/x.go", "package x\nimport \"github.com/townsendmerino/aikit/vision\"\n") // skipped
	write("docs/readme.md", "import \"github.com/townsendmerino/aikit/audio\"\n")          // not Go
	pkgs, syms, err := kenUses(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{"ann", "embed", "linalg"} {
		if !pkgs[p] {
			t.Errorf("missed the import of %s", p)
		}
	}
	if len(pkgs) != 3 {
		t.Errorf("packages = %v, want ann, embed, linalg only", pkgs)
	}
	if !syms["embed"]["StaticModel"] || !syms["embed"]["LoadGGUF"] || syms["embed"]["FS"] {
		t.Errorf("embed symbols = %v, want StaticModel and LoadGGUF, not FS", syms["embed"])
	}
	if !syms["linalg"]["Dot"] {
		t.Errorf("linalg symbols = %v, want Dot (from a test file)", syms["linalg"])
	}
}
