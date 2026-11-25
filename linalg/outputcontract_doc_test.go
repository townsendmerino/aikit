package linalg

import (
	"go/ast"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg/internal/outputcontract"
)

// R-11 (goinfer docs/tasks/task-recompute-audit.md §5), the documentation half: every exported function that writes into a caller's buffer says, in its doc comment,
// which of outputcontract.ContractForm's four forms it is (overwrites, zeroes internally, accumulates, rewrites in place). The probe in outputcontract_test.go
// proves the statements are TRUE; this test proves they are PRESENT, and that a new writer cannot be added without one. outputcontract.ParseDecls reads
// source, so this covers every arch-tagged file on any host.

// longer exists, or no longer carries a contract line, fails.
var nonDstWriters = map[string]bool{
	"QuantizeActivationsGroupedInto": true, "QuantizeRowInt8": true, "QuantizeGroupInt4Row": true,
	"RepackInt4InPlace": true,
	"MatmulBTW8A8Batch": true, "MatmulBTW4A8Batch": true, // outputs are each op's Dst field
}

func TestOutputContract_everyWriterIsDocumented(t *testing.T) {
	seenNonDst := map[string]bool{}
	for _, dir := range []string{".", "../embed", "../encoder", "../gpu/encmetal", "../gpu/enccuda"} {
		for _, d := range outputcontract.ParseDecls(t, dir) {
			// unexported helpers carry no contract obligation, and neither do exported methods of unexported types
			unexportedRecv := strings.Contains(d.Key, "(") && !ast.IsExported(strings.TrimLeft(d.Key[1:strings.Index(d.Key, ")")], "*"))
			isWriter := d.Exported && !unexportedRecv && (d.OutParam || (dir == "." && nonDstWriters[d.Key]))
			if dir == "." && nonDstWriters[d.Key] {
				seenNonDst[d.Key] = true
			}
			if d.Exported && strings.Contains(d.Doc, "Output contract:") && !outputcontract.ContractForm.MatchString(d.Doc) {
				t.Errorf("%s/%s %s: 'Output contract:' line is not one of the recognised forms (overwrites …; do not pre-zero | zeroes … internally | accumulates into …; caller zeroes | rewrites … in place)", dir, d.File, d.Key)
			}
			// cpuBackend is unexported but is the CPU implementation of an exported interface; it is documented too
			if d.Key == "(*cpuBackend).MatmulBT" {
				isWriter = true
			}
			if isWriter && !outputcontract.ContractForm.MatchString(d.Doc) {
				t.Errorf("%s/%s: %s writes into a caller's buffer but its doc comment has no 'Output contract:' line (R-11) — say which of overwrites / zeroes internally / accumulates it is", dir, d.File, d.Key)
			}
		}
	}
	for n := range nonDstWriters {
		if !seenNonDst[n] {
			t.Errorf("nonDstWriters lists %s, which no longer exists in linalg", n)
		}
	}
	// the three embed writers the name scan cannot see (dst is a parameter of a returned closure, or named differently)
	for _, want := range []string{"DequantMXFP4Blocks", "DequantMXFP4Split", "L2Normalize", "(*GGUFFile).RowDequantizer"} {
		found := false
		for _, d := range outputcontract.ParseDecls(t, "../embed") {
			if d.Key == want {
				found = true
				if !outputcontract.ContractForm.MatchString(d.Doc) {
					t.Errorf("embed %s has no 'Output contract:' line", want)
				}
			}
		}
		if !found {
			t.Errorf("embed %s no longer exists; update this test", want)
		}
	}
	if _, err := os.Stat("../embed"); err != nil {
		t.Fatal(err)
	}
}
