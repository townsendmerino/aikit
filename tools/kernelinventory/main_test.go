package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/tools/gpumod"
)

// TestInventory_upToDate fails when docs/internal/kernel-inventory.md differs from a fresh run: a kernel added,
// removed, re-dispatched or given a new oracle without regenerating it (go run -C tools ./kernelinventory --write).
func TestInventory_upToDate(t *testing.T) {
	root, err := gpumod.RepoRoot()
	if err != nil {
		t.Fatal(err)
	}
	rows, err := gather(root)
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(root, InventoryPath))
	if err != nil {
		t.Fatal(err)
	}
	if want := render(rows); !bytes.Equal(got, want) {
		t.Errorf("%s is stale: run go run -C tools ./kernelinventory --write", InventoryPath)
	}
}

// TestGather_readsRowsAndTextSymbols pins the reader on a planted package: literal and constant fields, a + of
// literals, a row per arch, and a row whose symbol no .s file defines (an error, not a silent gap).
func TestGather_readsRowsAndTextSymbols(t *testing.T) {
	dir := t.TempDir()
	write := func(name, src string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("k_test.go", `package k
import "github.com/townsendmerino/aikit/internal/kernreg"
const why = "a measurement " + "arm"
var rows = []kernreg.Row{
	{Symbol: "sumAVX2", Arch: "amd64", Dispatch: "Sum", Oracle: "TestSum"},
	{Symbol: "sumNEON", Arch: "arm64", Undispatched: why, Oracle: "TestSumNEON"},
}
`)
	write("k_amd64.s", "TEXT ·sumAVX2(SB), NOSPLIT, $0-8\n")
	write("k_arm64.s", "TEXT ·sumNEON(SB), NOSPLIT, $0-8\n")
	rows, err := registryRows(dir, "k")
	if err != nil || len(rows) != 2 {
		t.Fatalf("registryRows: %v, %d rows", err, len(rows))
	}
	if rows[1].Undispatched != "a measurement arm" {
		t.Errorf("a constant field read as %q", rows[1].Undispatched)
	}
	text, err := textSymbols(dir)
	if err != nil || text["amd64/sumAVX2"] != "k_amd64.s" || text["arm64/sumNEON"] != "k_arm64.s" {
		t.Fatalf("textSymbols: %v %v", text, err)
	}
	out := string(render(append(rows[:0:0], rows...)))
	for _, want := range []string{"`sumAVX2`", "none: a measurement arm"} {
		if !strings.Contains(out, want) {
			t.Errorf("render is missing %q", want)
		}
	}
}
