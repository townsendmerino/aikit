package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/tools/gate"
)

func TestInDeeperModule(t *testing.T) {
	dirs := map[string]string{rootPath + "/gpu": "gpu", rootPath + "/gpu/anncuda": "gpu/anncuda", rootPath + "/chunk/treesitter": "chunk/treesitter"}
	for _, tc := range []struct {
		file, dir string
		want      bool
	}{
		{"gpu/metal.go", "gpu", false},
		{"gpu/anncuda/backend.go", "gpu", true},          // another module's file
		{"gpu/anncuda/backend.go", "gpu/anncuda", false}, // its own
		{"gpu/anncudax/x.go", "gpu", false},              // a prefix of a name is not a parent
		{"chunk/treesitter/x.go", "chunk/treesitter", false},
	} {
		if got := inDeeperModule(tc.file, tc.dir, dirs); got != tc.want {
			t.Errorf("inDeeperModule(%q, %q) = %v, want %v", tc.file, tc.dir, got, tc.want)
		}
	}
}

func TestInRepo(t *testing.T) {
	var mf modFile
	mf.Require = []struct{ Path, Version string }{
		{rootPath, "v1.75.0"}, {rootPath + "/gpu", "v0.35.0"}, {"github.com/ebitengine/purego", "v0.9.0"},
		{"github.com/townsendmerino/aikitx", "v1.0.0"}, // not this repo
	}
	paths, label := inRepo(mf)
	if len(paths) != 2 || label != "root@v1.75.0,gpu@v0.35.0" {
		t.Errorf("inRepo = %v, %q; want the two in-repo requires, labelled root@v1.75.0,gpu@v0.35.0", paths, label)
	}
}

func TestHeadTestable(t *testing.T) {
	for dir, want := range map[string]bool{"chunk/treesitter": true, "gpu": false, "gpu/annmetal": false, "gpux": true} {
		if got := headTestable(dir); got != want {
			t.Errorf("headTestable(%q) = %v, want %v", dir, got, want)
		}
	}
}

// A retired module is n/a without touching the network: there is no future tag to protect.
func TestCheckHead_retiredIsNA(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "gpu", "old")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	mod := rootPath + "/gpu/old"
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("// Deprecated: retired.\nmodule "+mod+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	c := checkHead(config{proxy: "off"}, root, mod, map[string]string{mod: "gpu/old"})
	if c.Outcome != gate.NA || c.Field("retired") != "n/a" {
		t.Errorf("retired module: %+v, want n/a", c)
	}
}

// Every published module has a directory, or --head would skip it without saying so.
func TestModuleDirs_coverPublished(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	dirs, err := moduleDirs(root)
	if err != nil {
		t.Fatal(err)
	}
	for m := range published {
		if _, ok := dirs[m]; !ok {
			t.Errorf("published module %s has no go.mod in the tree", m)
		}
	}
}
