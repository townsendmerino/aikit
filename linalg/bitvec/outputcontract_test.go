package bitvec

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg/internal/outputcontract"
)

// The output-contract gate over bitvec's writers (the harness and its rules are linalg/internal/outputcontract's package
// doc). Run it per build: the host's default kernels, GOARCH=386 for the generic ones, -tags aikit_nopopcnt for the
// portable amd64 path.
func TestOutputContract_overwriteOrZeroInternally(t *testing.T) {
	for _, c := range outputContractCases() {
		t.Run(c.Name, func(t *testing.T) { outputcontract.Run(t, c) })
	}
}

func outputContractCases() []outputcontract.Case {
	var cs []outputcontract.Case
	add := func(name string, setup func() (func(), []*outputcontract.OutBuf)) {
		cs = append(cs, outputcontract.Case{Name: name, Setup: setup})
	}
	for _, dim := range []int{37, 64, 130} {
		words := (dim + 63) / 64
		add(fmt.Sprintf("PackSignBitsRow/dim=%d", dim), func() (func(), []*outputcontract.OutBuf) {
			v := randF32(60, dim)
			dst, o := outputcontract.MkOut[uint64]("dst", words)
			return func() { PackSignBitsRow(dst, v) }, []*outputcontract.OutBuf{o}
		})
		add(fmt.Sprintf("PackSignBits/dim=%d", dim), func() (func(), []*outputcontract.OutBuf) {
			const n = 5
			src := randF32(61, n*dim)
			dst, o := outputcontract.MkOut[uint64]("dst", n*words)
			return func() { PackSignBits(dst, src, n, dim) }, []*outputcontract.OutBuf{o}
		})
		add(fmt.Sprintf("HammingRows/dim=%d", dim), func() (func(), []*outputcontract.OutBuf) {
			const n = 7
			q := make([]uint64, words)
			PackSignBitsRow(q, randF32(62, dim))
			codes := make([]uint64, n*words)
			PackSignBits(codes, randF32(63, n*dim), n, dim)
			dst, o := outputcontract.MkOut[uint16]("dst", n)
			return func() { HammingRows(q, codes, words, n, dst) }, []*outputcontract.OutBuf{o}
		})
	}
	return cs
}

// TestOutputContract_everyWriterIsDocumented: every exported bitvec writer carries an "Output contract:" line in one of
// outputcontract.ContractForm's forms.
func TestOutputContract_everyWriterIsDocumented(t *testing.T) {
	n := 0
	for _, d := range outputcontract.ParseDecls(t, ".") {
		if d.Exported && strings.Contains(d.Doc, "Output contract:") && !outputcontract.ContractForm.MatchString(d.Doc) {
			t.Errorf("%s %s: 'Output contract:' line is not one of the recognised forms", d.File, d.Key)
		}
		if d.Exported && d.OutParam {
			n++
			if !outputcontract.ContractForm.MatchString(d.Doc) {
				t.Errorf("%s: %s writes into a caller's buffer but its doc comment has no 'Output contract:' line", d.File, d.Key)
			}
		}
	}
	if n == 0 {
		t.Fatal("no writers found: the scan read nothing")
	}
}

func randF32(seed int64, n int) []float32 {
	r := rand.New(rand.NewSource(seed))
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}
