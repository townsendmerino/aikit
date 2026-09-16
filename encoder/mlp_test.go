package encoder

import (
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestConfig_geluTanh pins the activation-name → erf-vs-tanh split. "gelu_new",
// "gelu_fast", and "gelu_pytorch_tanh" are three HF names for the identical tanh
// formula (linalg.GELUTanhF32) — a different function from plain "gelu"'s erf
// form, not a safe collapse. All four still route to the dense (non-gated) MLP
// via gatedMLP(); geluTanh() picks which GELU that dense MLP applies.
func TestConfig_geluTanh(t *testing.T) {
	cases := []struct {
		act  string
		tanh bool
	}{
		{"gelu", false},
		{"gelu_new", true},
		{"gelu_fast", true},
		{"gelu_pytorch_tanh", true},
		// swiglu and unknown/empty activations stay on the gated path per
		// gatedMLP()'s doc comment; geluTanh() is meaningless there but must
		// still report false rather than true.
		{"swiglu", false},
		{"", false},
	}
	for _, c := range cases {
		cfg := &Config{ActivationFunction: c.act}
		if got := cfg.geluTanh(); got != c.tanh {
			t.Errorf("activation_function=%q: geluTanh() = %v, want %v", c.act, got, c.tanh)
		}
	}
	// The four names geluTanh() distinguishes must all still select the dense
	// (non-gated) MLP path — geluTanh() only decides which GELU that path uses.
	for _, act := range []string{"gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh"} {
		if (&Config{ActivationFunction: act}).gatedMLP() {
			t.Errorf("activation_function=%q: gatedMLP() = true, want false (dense path)", act)
		}
	}
}

// TestGeluMLP_tanhDispatch guards against geluMLP silently collapsing the tanh
// and erf branches back into one — the bug this pair of functions fixes. A
// dense-GELU checkpoint whose activation_function is "gelu_pytorch_tanh" (e.g.
// any Gemma-family MLP) must produce numerically different output than "gelu"
// would, and must match linalg.GELUTanhF32 applied by hand.
func TestGeluMLP_tanhDispatch(t *testing.T) {
	const D, inter, L = 8, 16, 3
	rng := []float32{
		0.5, -0.5, 1.2, -1.2, 0.0, 2.5, -2.5, 0.1,
		-0.1, 0.9, -0.9, 1.7, -1.7, 3.0, -3.0, 0.3,
		-0.3, 0.7, -0.7, 1.1, -1.1, 2.1, -2.1, 0.4,
	}
	h := make([]float32, L*D)
	fc1 := make([]float32, inter*D)
	fc2 := make([]float32, D*inter)
	for i := range fc1 {
		fc1[i] = rng[i%len(rng)] * 0.1
	}
	for i := range fc2 {
		fc2[i] = rng[(i+7)%len(rng)] * 0.1
	}

	run := func(tanh bool) []float32 {
		hh := append([]float32(nil), h...)
		for i := range hh {
			hh[i] = rng[i%len(rng)]
		}
		s := getScratch()
		defer putScratch(s)
		s.ensureLayer(L, D, inter, 1, D, L)
		geluMLP(hh, fc1, nil, fc2, nil, D, inter, L, tanh, s)
		return hh
	}

	erf := run(false)
	tanh := run(true)
	diff := false
	for i := range erf {
		if erf[i] != tanh[i] {
			diff = true
			break
		}
	}
	if !diff {
		t.Fatal("geluMLP(tanh=true) produced bit-identical output to geluMLP(tanh=false) — the erf/tanh dispatch collapsed back to one branch")
	}

	// geluTanh (the elementwise function geluMLP dispatches to) must match
	// linalg.GELUTanhF32 exactly — it's a thin chunked wrapper, not a new formula.
	x := append([]float32(nil), rng...)
	want := make([]float32, len(x))
	for i, v := range x {
		want[i] = linalg.GELUTanhF32(v)
	}
	geluTanh(x)
	for i := range x {
		if x[i] != want[i] {
			t.Fatalf("geluTanh[%d] = %v, want linalg.GELUTanhF32 = %v", i, x[i], want[i])
		}
	}
}

func BenchmarkAddRowBias(b *testing.B) {
	for _, tc := range []struct {
		name string
		M, N int
	}{
		{"M512_N768", 512, 768},
		{"M512_N3072", 512, 3072},
	} {
		dst := make([]float32, tc.M*tc.N)
		bias := make([]float32, tc.N)
		for i := range bias {
			bias[i] = float32(i%100) * 0.01
		}
		b.Run(tc.name, func(b *testing.B) {
			b.SetBytes(int64(tc.M * tc.N * 4))
			b.ReportAllocs()
			for b.Loop() {
				addRowBias(dst, bias, tc.M, tc.N)
			}
		})
	}
}
