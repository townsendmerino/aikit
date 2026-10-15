package encoder

import (
	"math/rand"
	"os"
	"sync"
	"testing"
)

func loadTestGTE(t *testing.T) *GTE {
	t.Helper()
	const dir = "../testdata/arctic2-m"
	if _, err := os.Stat(dir); err != nil {
		t.Skip("testdata/arctic2-m/ not present; see scripts/README.md")
	}
	g, err := LoadGTE(dir)
	if err != nil {
		t.Fatalf("LoadGTE: %v", err)
	}
	return g
}

// TestRopeCache_viewIsBitIdentical pins the claim ropeCache rests on: a view
// over the first seqLen rows of a longer table equals a table built at exactly
// seqLen, bit for bit. If row m's value depended on the table's length in any
// way, the cache would silently change every rotary model's output.
func TestRopeCache_viewIsBitIdentical(t *testing.T) {
	for _, headDim := range []int{32, 64, 128} {
		for _, base := range []float64{1000, 10000, 160000} {
			full := newRopeTable(1024, headDim, base)
			for _, seqLen := range []int{1, 2, 7, 63, 64, 512, 1023, 1024} {
				want := newRopeTable(seqLen, headDim, base)
				got := full.view(seqLen)
				if got.seqLen != want.seqLen || got.headDim != want.headDim || got.halfDim != want.halfDim {
					t.Fatalf("headDim=%d base=%g seqLen=%d: shape mismatch", headDim, base, seqLen)
				}
				if len(got.cos) != len(want.cos) || len(got.sin) != len(want.sin) {
					t.Fatalf("headDim=%d base=%g seqLen=%d: cos/sin length %d/%d want %d/%d",
						headDim, base, seqLen, len(got.cos), len(got.sin), len(want.cos), len(want.sin))
				}
				for i := range want.cos {
					if got.cos[i] != want.cos[i] || got.sin[i] != want.sin[i] {
						t.Fatalf("headDim=%d base=%g seqLen=%d: entry %d cos %v/%v sin %v/%v — "+
							"a view is supposed to be the same arithmetic",
							headDim, base, seqLen, i, got.cos[i], want.cos[i], got.sin[i], want.sin[i])
					}
				}
			}
		}
	}
}

// TestRopeCache_getMatchesFresh drives the cache the way a forward does —
// growing, shrinking, and switching (headDim, base) — and checks every result
// against a freshly built table. The shrink case is the one that matters: a
// short request after a long one must return a NARROWED view, not the long
// table, or apply() would rotate past the end of the activations.
func TestRopeCache_getMatchesFresh(t *testing.T) {
	var c ropeCache
	type req struct {
		seqLen, headDim int
		base            float64
	}
	// Deliberately out of order: grow, shrink, exact-hit, then change the key.
	for _, r := range []req{
		{16, 64, 10000}, {512, 64, 10000}, {8, 64, 10000}, {512, 64, 10000},
		{64, 32, 1000}, {128, 32, 1000}, {4, 32, 1000},
		{256, 64, 160000}, {16, 64, 10000},
	} {
		got := c.get(r.seqLen, r.headDim, r.base)
		want := newRopeTable(r.seqLen, r.headDim, r.base)
		if got.seqLen != want.seqLen {
			t.Fatalf("%+v: seqLen %d want %d — a shrink must narrow the view", r, got.seqLen, want.seqLen)
		}
		if len(got.cos) != len(want.cos) {
			t.Fatalf("%+v: len(cos) %d want %d", r, len(got.cos), len(want.cos))
		}
		for i := range want.cos {
			if got.cos[i] != want.cos[i] || got.sin[i] != want.sin[i] {
				t.Fatalf("%+v: entry %d differs from a fresh table", r, i)
			}
		}
	}
}

// TestRopeCache_concurrent is a race-detector target: concurrent growth must
// publish a new table rather than resize one another goroutine is reading.
func TestRopeCache_concurrent(t *testing.T) {
	var c ropeCache
	var wg sync.WaitGroup
	for w := range 8 {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(w)))
			for range 200 {
				n := 1 + rng.Intn(600)
				tab := c.get(n, 64, 10000)
				if tab.seqLen != n {
					t.Errorf("seqLen %d want %d", tab.seqLen, n)
					return
				}
				// Read every entry: with an unsafe resize this races or tears.
				var acc float32
				for i := range tab.cos {
					acc += tab.cos[i] + tab.sin[i]
				}
				_ = acc
			}
		}(w)
	}
	wg.Wait()
}

// TestScratchUpGate_fullyOverwritten guards the hazard item 8 introduces: the
// fused up/gate buffer used to be a fresh (zeroed) allocation and is now pooled,
// so it arrives holding a previous forward's values. Poisoning it and checking
// the encode is unchanged proves the fused matmul writes every element it reads.
func TestScratchUpGate_fullyOverwritten(t *testing.T) {
	g := loadTestGTE(t)
	const text = "how do i parse json in go"

	clean, err := g.Encode(text)
	if err != nil {
		t.Fatal(err)
	}

	// Poison every pooled scratch this process can hand out, then re-encode.
	poisoned := make([]*scratch, 0, 16)
	for range 16 {
		s := getScratch()
		s.upGate = ensureF32(s.upGate, 4096*2*3072)
		for i := range s.upGate {
			s.upGate[i] = 7777.0
		}
		poisoned = append(poisoned, s)
	}
	for _, s := range poisoned {
		putScratch(s)
	}

	dirty, err := g.Encode(text)
	if err != nil {
		t.Fatal(err)
	}
	if len(clean) != len(dirty) {
		t.Fatalf("length changed: %d vs %d", len(clean), len(dirty))
	}
	for i := range clean {
		if clean[i] != dirty[i] {
			t.Fatalf("element %d: %v with a clean arena, %v with a poisoned one — "+
				"the pooled upGate buffer is being read before it is written", i, clean[i], dirty[i])
		}
	}
}

// TestGTEEncode_loadIndependent is the end-to-end version of the per-kernel
// inertness gates: a forward now chooses among parallel and serial
// implementations for its matmuls, its attention softmax AND its GeGLU
// activation, and every one of those choices is made by reading the in-flight
// counter. If any split were not numerically inert, an encoder's output would
// depend on how busy the machine is — a golden that passes in CI and drifts in
// production under a concurrent batch.
//
// Raising the counter forces every gate to decline, so this compares the
// fully-parallel forward against the fully-serial one, bit for bit.
func TestGTEEncode_loadIndependent(t *testing.T) {
	g := loadTestGTE(t)
	// Long enough that every threshold is cleared: softmax is O(L²) and the
	// GeGLU pass is L*intermediate.
	text := gteBenchText(256)

	parallel, err := g.Encode(text)
	if err != nil {
		t.Fatal(err)
	}

	enterForward()
	enterForward() // a batch owns the cores: every gate must decline
	serial, err := g.Encode(text)
	leaveForward()
	leaveForward()
	if err != nil {
		t.Fatal(err)
	}

	if len(parallel) != len(serial) {
		t.Fatalf("length %d vs %d", len(parallel), len(serial))
	}
	var diff int
	for i := range parallel {
		if parallel[i] != serial[i] {
			diff++
		}
	}
	if diff != 0 {
		t.Fatalf("%d/%d embedding components differ between the parallel and serial forwards — "+
			"encoder output must not depend on machine load", diff, len(parallel))
	}
	if inflightForwards.Load() != 0 {
		t.Fatalf("in-flight counter leaked: %d", inflightForwards.Load())
	}
}
