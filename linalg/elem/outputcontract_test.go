package elem

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg/internal/outputcontract"
)

// The output-contract gate over elem's writers (the harness and its rules are linalg/internal/outputcontract's package
// doc). Run it per build: the host's default kernels, GOARCH=386 for the generic ones, -tags aikit_noavx2 for the
// narrower amd64 tier.
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

	// ---- elementwise and row kernels: dst <- f(src); they may alias, which is one more reason a pre-zero of dst is harmful.
	type ew struct {
		name string
		f    func(dst, src []float32)
	}
	for _, k := range []ew{
		{"ExpF32Into", expF32Into}, {"SoftmaxRowInto", softmaxRowInto}, {"GELUInto", geluInto}, {"SiLUInto", siluInto}, {"GELUTanhInto", GELUTanhInto}, {"TanhInto", TanhInto},
		{"ExpContractInto", ExpContractInto}, {"GELUContractInto", GELUContractInto}, {"GELUTanhContractInto", GELUTanhContractInto}, {"SiLUContractInto", SiLUContractInto}, {"SoftmaxRowContractInto", SoftmaxRowContractInto},
		{"SoftmaxRowScaledInto", func(d, s []float32) { softmaxRowScaledInto(d, s, 0.35) }}, {"SoftmaxRowScaledContractInto", func(d, s []float32) { SoftmaxRowScaledContractInto(d, s, 0.35) }},
	} {
		for _, n := range []int{1, 5, 37, 130} {
			k, n := k, n
			add(fmt.Sprintf("%s/n=%d", k.name, n), func() (func(), []*outputcontract.OutBuf) {
				src := randF32(int64(n), n)
				dst, o := outputcontract.MkOut[float32]("dst", n)
				return func() { k.f(dst, src) }, []*outputcontract.OutBuf{o}
			})
		}
	}
	return cs
}

// TestOutputContract_everyWriterIsDocumented: every exported elem function with a dst parameter carries an
// "Output contract:" line in one of outputcontract.ContractForm's forms.
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
