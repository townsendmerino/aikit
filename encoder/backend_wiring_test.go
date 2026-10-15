package encoder

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// backend_wiring_test.go gates the Phase-4 seam wiring: that Backend.MatmulBT is
// ACTUALLY CALLED by the forward.
//
// This is the specific thing that was broken. Backend, NewBackend and RegisterBackend
// all existed and goinfer/gpu registered "webgpu", but nothing in the forward ever
// invoked MatmulBT — so a caller could ask for a backend, receive one, and have it do
// nothing at all, silently. No test caught that, because every existing test asserts
// numeric output and the pure-Go path produced correct numbers whether or not a
// backend was attached.
//
// So the assertions here are deliberately about DISPATCH, not just numerics:
//   - a spy backend must observe calls (it was observing none),
//   - a backend that computes the same thing must not change the output,
//   - a backend that computes something WRONG must change it — otherwise the seam is
//     decorative and a passing "backend works" claim would be vacuous.
//
// These run against the forward helpers with synthetic weights rather than a loaded
// checkpoint, so they are not asset-gated: a seam test that skips in CI is exactly the
// gap that let the dangling seam survive this long.

// spyBackend records every matmul it is asked for and delegates to the pure-Go path.
type spyBackend struct {
	calls  int
	shapes [][3]int // (M, K, N) per call
}

func (s *spyBackend) Name() string { return "spy" }
func (s *spyBackend) MatmulBT(a, b, dst []float32, M, K, N int) {
	s.calls++
	s.shapes = append(s.shapes, [3]int{M, K, N})
	matmulBTInto(a, b, dst, M, K, N)
}
func (s *spyBackend) Close() error { return nil }

// wrongBackend computes a deliberately different result, so the break-it-first check
// can prove the seam is load-bearing.
type wrongBackend struct{}

func (wrongBackend) Name() string { return "wrong" }
func (wrongBackend) MatmulBT(a, b, dst []float32, M, K, N int) {
	matmulBTInto(a, b, dst, M, K, N)
	for i := range dst {
		dst[i] *= 2 // any deviation will do; the point is that it must SHOW
	}
}
func (wrongBackend) Close() error { return nil }

// attnFixture builds a small self-attention problem with deterministic weights.
type attnFixture struct {
	h, wqkv, wqkvb, outProj, outProjB []float32
	heads, headDim, D, L              int
	rope                              *ropeTable
}

func newAttnFixture() *attnFixture {
	const heads, headDim, L = 2, 8, 6
	D := heads * headDim
	rng := rand.New(rand.NewSource(42))
	rf := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(rng.NormFloat64()) * 0.1
		}
		return out
	}
	return &attnFixture{
		h: rf(L * D), wqkv: rf(3 * D * D), wqkvb: rf(3 * D),
		outProj: rf(D * D), outProjB: rf(D),
		heads: heads, headDim: headDim, D: D, L: L,
		rope: newRopeTable(L, headDim, 10000),
	}
}

// run executes selfAttention with the given backend and returns the output.
func (f *attnFixture) run(be Backend) []float32 {
	s := getScratch()
	defer putScratch(s)
	s.be = be
	s.ensureLayer(scratchLayout{packedQKV: true}, f.L, f.D, 4*f.D, f.heads, f.headDim, f.L)
	h := append([]float32(nil), f.h...)
	return selfAttention(h, f.wqkv, f.wqkvb, f.outProj, f.outProjB, f.heads, f.headDim, f.D, f.L, f.L, f.rope, s)
}

// TestBackend_isActuallyCalled is the regression gate for the dangling seam: a
// registered backend must SEE the forward's matmuls.
func TestBackend_isActuallyCalled(t *testing.T) {
	f := newAttnFixture()
	spy := &spyBackend{}
	f.run(spy)

	if spy.calls == 0 {
		t.Fatal("backend saw ZERO matmuls — the seam is dangling again (this is exactly the Phase-4 bug)")
	}
	// selfAttention does the QKV projection, per-head QKᵀ and scores·V, and the
	// output projection: 2 + 2*heads for this fixture.
	want := 2 + 2*f.heads
	if spy.calls != want {
		t.Errorf("backend saw %d matmuls, want %d (shapes: %v)", spy.calls, want, spy.shapes)
	}
	// The big projections must be among them — a seam that only caught the tiny
	// per-head matmuls would be nearly useless.
	sawQKV := false
	for _, sh := range spy.shapes {
		if sh[0] == f.L && sh[1] == f.D && sh[2] == 3*f.D {
			sawQKV = true
		}
	}
	if !sawQKV {
		t.Errorf("backend never saw the QKV projection [%d,%d]x[%d] — shapes: %v", f.L, f.D, 3*f.D, spy.shapes)
	}
	t.Logf("backend observed %d matmuls: %v", spy.calls, spy.shapes)
}

// TestBackend_nilIsUnchanged pins the default: no backend must produce EXACTLY what a
// backend delegating to the pure-Go path produces. Bit-identical, not close — the
// wiring is supposed to be a dispatch, not a reimplementation.
func TestBackend_nilIsUnchanged(t *testing.T) {
	f := newAttnFixture()
	base := f.run(nil)
	viaCPU := f.run(&spyBackend{})
	for i := range base {
		if base[i] != viaCPU[i] {
			t.Fatalf("nil-backend and delegating-backend diverged at %d: %v != %v", i, base[i], viaCPU[i])
		}
	}
	// The registered "cpu" backend must agree too — it is what NewBackend("") hands
	// callers, so a divergence there would be a real behaviour change.
	cpu, err := NewBackend("cpu")
	if err != nil {
		t.Fatalf("NewBackend(cpu): %v", err)
	}
	defer func() { _ = cpu.Close() }()
	viaRegistered := f.run(cpu)
	for i := range base {
		if base[i] != viaRegistered[i] {
			t.Fatalf(`nil and NewBackend("cpu") diverged at %d: %v != %v`, i, base[i], viaRegistered[i])
		}
	}
	t.Log("nil ≡ delegating backend ≡ NewBackend(\"cpu\"), bit-identical")
}

// TestBackend_isLoadBearing is the break-it-first: a backend that computes something
// different MUST change the output. Without this, the two tests above could both pass
// on a seam that silently ignored the backend for the numerics.
func TestBackend_isLoadBearing(t *testing.T) {
	f := newAttnFixture()
	base := f.run(nil)
	bad := f.run(wrongBackend{})
	worst := 0.0
	for i := range base {
		if d := math.Abs(float64(base[i]) - float64(bad[i])); d > worst {
			worst = d
		}
	}
	if worst == 0 {
		t.Fatal("a deliberately-wrong backend changed nothing — the seam is decorative")
	}
	t.Logf("wrong backend moves the output by %.3g — the seam is load-bearing", worst)
}

// TestBackend_mlpIsRouted covers the other half of the hot path: the MLP projections
// are the largest matmuls in a forward, so a seam that missed them would leave most of
// the FLOPs on the CPU.
func TestBackend_mlpIsRouted(t *testing.T) {
	const D, inter, L = 16, 64, 5
	rng := rand.New(rand.NewSource(7))
	rf := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(rng.NormFloat64()) * 0.1
		}
		return out
	}
	h, fc1, fc1b, fc2, fc2b := rf(L*D), rf(inter*D), rf(inter), rf(D*inter), rf(D)

	s := getScratch()
	defer putScratch(s)
	spy := &spyBackend{}
	s.be = spy
	s.ensureLayer(scratchLayout{}, L, D, inter, 2, D/2, L)
	geluMLP(append([]float32(nil), h...), fc1, fc1b, fc2, fc2b, D, inter, L, false, s)

	if spy.calls == 0 {
		t.Fatal("backend saw no MLP matmuls — the MLP is not routed through the seam")
	}
	t.Logf("MLP routed: %d matmuls %v", spy.calls, spy.shapes)
}

// TestBackend_scratchDoesNotLeakBackend guards the pooling hazard: scratches are
// pooled and shared, so a backend left on one would silently attach itself to the next
// forward — including one on a model that never opted in.
func TestBackend_scratchDoesNotLeakBackend(t *testing.T) {
	s := getScratch()
	s.be = &spyBackend{}
	putScratch(s)
	for range 20 {
		got := getScratch()
		if got.be != nil {
			putScratch(got)
			t.Fatal("a pooled scratch carried a backend into a new forward")
		}
		putScratch(got)
	}
	t.Log("putScratch clears the backend; no cross-forward leak")
}

// --- int8 (Q8Backend) dispatch ---

// spyQ8 is a Backend that ALSO implements Q8Backend, recording the int8 matmuls and
// delegating to the CPU path so numerics are unchanged.
type spyQ8 struct {
	spyBackend
	q8calls  int
	q8shapes [][3]int
	handle   bool // whether to claim the call
	deq      []float32
}

func (s *spyQ8) MatmulBTQ8(dst, a []float32, wq []int8, wscales []float32, M, K, N int) bool {
	s.q8calls++
	s.q8shapes = append(s.q8shapes, [3]int{M, K, N})
	if !s.handle {
		return false
	}
	if cap(s.deq) < N*K {
		s.deq = make([]float32, N*K)
	}
	matmulBTQ8Into(dst, a, wq, wscales, M, K, N, s.deq[:N*K])
	return true
}

type wrongQ8 struct{ spyBackend }

func (wrongQ8) MatmulBTQ8(dst, a []float32, wq []int8, wscales []float32, M, K, N int) bool {
	for i := range dst[:M*N] {
		dst[i] = 1
	}
	return true
}

// runQ8MLP drives swigluMLPQ8 — the int8 MLP — with synthetic weight-only Q8 weights.
// linalg.QuantizeInt8(..., w8a8=false) is what the encoder's loader uses: weight-only,
// NOT full W8A8. That flag is the same distinction Q8Backend's contract turns on.
func runQ8MLP(be Backend) []float32 {
	const D, inter, L = 16, 64, 5
	rng := rand.New(rand.NewSource(21))
	rf := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(rng.NormFloat64()) * 0.1
		}
		return out
	}
	qm := func(rows, cols int) *linalg.WeightMat {
		m := linalg.QuantizeInt8(rf(rows*cols), rows, cols, false)
		return &m
	}
	h := rf(L * D)
	fc11, fc12, fc2 := qm(inter, D), qm(inter, D), qm(D, inter)
	s := getScratch()
	defer putScratch(s)
	s.be = be
	s.ensureLayer(q8ScratchLayout, L, D, inter, 2, D/2, L)
	s.ensureDeqW(D, inter)
	swigluMLPQ8(h, fc11, fc12, fc2, D, inter, L, s)
	return h
}

// TestQ8Backend_isActuallyCalled is the int8 twin of TestBackend_isActuallyCalled. The
// int8 forward called matmulBTQ8Into DIRECTLY before this seam existed, so an attached
// backend accelerated the f32 path and silently did nothing for int8 models.
func TestQ8Backend_isActuallyCalled(t *testing.T) {
	spy := &spyQ8{handle: true}
	runQ8MLP(spy)
	if spy.q8calls == 0 {
		t.Fatal("Q8Backend saw ZERO int8 matmuls — the int8 seam is dangling")
	}
	if spy.q8calls != 3 { // fc11, fc12, fc2
		t.Errorf("Q8Backend saw %d int8 matmuls, want 3 (shapes %v)", spy.q8calls, spy.q8shapes)
	}
	t.Logf("Q8Backend observed %d int8 matmuls: %v", spy.q8calls, spy.q8shapes)
}

// TestQ8Backend_declineFallsBackExactly pins the decline contract: a backend returning
// false must leave the result BIT-IDENTICAL to having no backend at all. That is what
// makes "decline on anything unusual" a safe default for implementations.
func TestQ8Backend_declineFallsBackExactly(t *testing.T) {
	base := runQ8MLP(nil)
	declined := runQ8MLP(&spyQ8{handle: false})
	for i := range base {
		if base[i] != declined[i] {
			t.Fatalf("declining backend changed the result at %d: %v != %v", i, declined[i], base[i])
		}
	}
	// A backend with NO Q8 capability must behave the same — this is the WebGPU case.
	plain := runQ8MLP(&spyBackend{})
	for i := range base {
		if base[i] != plain[i] {
			t.Fatalf("f32-only backend changed the int8 result at %d", i)
		}
	}
	t.Log("decline ≡ no backend ≡ f32-only backend, bit-identical")
}

// TestQ8Backend_isLoadBearing: a Q8 backend computing something different must change
// the output, or the int8 seam is decorative.
func TestQ8Backend_isLoadBearing(t *testing.T) {
	base := runQ8MLP(nil)
	bad := runQ8MLP(&wrongQ8{})
	diff := false
	for i := range base {
		if base[i] != bad[i] {
			diff = true
			break
		}
	}
	if !diff {
		t.Fatal("a deliberately-wrong Q8 backend changed nothing — the int8 seam is decorative")
	}
	t.Log("wrong Q8 backend moves the output — the int8 seam is load-bearing")
}
