package encoder

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sync"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// The phase-boundary poison gate (A-2 of docs/internal/archive/task-encoder-scratch-aliasing.md).
//
// Every forward calls s.phaseBoundary() before each attention and before each MLP, and
// claims that every phase-scoped scratch buffer is dead there: nothing written before the
// boundary is read after it. Aliasing the two phases over one slab (A-3) is only safe if
// that claim holds, so this gate checks it. At each boundary it overwrites every
// phase-scoped buffer with garbage, and requires each forward's output to be
// BIT-IDENTICAL to a run on a fresh, unpoisoned scratch.
//
// It poisons BOTH phases at every boundary, not only the phase being entered. On today's
// separate-buffer layout that is what aliasing looks like from the inside: entering the
// MLP, the attention buffers are about to be overwritten, and the MLP buffers hold the
// other phase's leftovers. Poisoning only the phase being entered would leave a stale
// cross-phase read (the MLP reading s.out) seeing valid data, and the gate would pass it.
//
// Comparing against a fresh-scratch reference, never against a second poisoned run, is
// deliberate: stale-buffer reads are deterministic, so two poisoned runs would agree with
// each other and both be wrong (see vision's TestQwenScratch_poisonedArenaIsInert).
//
// The models are synthetic, so the gate runs everywhere, CI included, with no checkpoint
// on disk. Real checkpoints go through exactly the same code.

// poisonF32s are the fills. NaN propagates through almost all arithmetic, but a
// comparison (top-k routing, a max) silently drops it, so a large finite value runs too.
var poisonF32s = []float32{float32(math.NaN()), 1e4}

// poisonI32 makes a stale read of an index table (moeExpert, moeTokens) panic on the
// bounds check instead of quietly routing to a real row.
const poisonI32 = 1 << 30

// phaseExempt are the scratch fields that are deliberately NOT phase-scoped.
var phaseExempt = map[string]string{
	"be":   "the compute backend, not a buffer",
	"deqW": "used by every q8 matmul in both phases; each one rewrites it in full first",
}

// poisonScratch overwrites every phase-scoped buffer of s, to its full capacity, with
// f (float32) or poisonI32 (int32). It walks the struct by reflection so that a buffer
// added later is covered without anyone remembering to list it here, and panics on a
// field type it does not know, which forces a decision about the new field's phase.
func poisonScratch(s *scratch, f float32) {
	poisonStruct(reflect.ValueOf(s).Elem(), f, true)
}

func poisonStruct(v reflect.Value, f float32, top bool) {
	t := v.Type()
	for i := range v.NumField() {
		name := t.Field(i).Name
		if _, ok := phaseExempt[name]; ok && top {
			continue
		}
		fv := v.Field(i)
		// Unexported fields: reach them through their address.
		fv = reflect.NewAt(fv.Type(), unsafe.Pointer(fv.UnsafeAddr())).Elem()
		switch b := fv.Addr().Interface().(type) {
		case *[]float32:
			full := (*b)[:cap(*b)]
			for j := range full {
				full[j] = f
			}
		case *[]int32:
			full := (*b)[:cap(*b)]
			for j := range full {
				full[j] = poisonI32
			}
		case *[]headScratch:
			for j := range *b {
				poisonStruct(reflect.ValueOf(&(*b)[j]).Elem(), f, false)
			}
		default:
			panic(fmt.Sprintf("poisonScratch: field %s.%s has type %s, which the gate does not know; "+
				"decide whether it is phase-scoped (poison it) or not (add it to phaseExempt)",
				t.Name(), name, fv.Type()))
		}
	}
}

// checkPhaseInert runs one forward three ways: fresh scratch (the reference), then twice
// per poison value with the boundary hook installed. The second poisoned run matters for
// buffers sized lazily inside a phase (the MoE tables): on the first run they are empty
// when the boundary fires, on the second they hold the first run's values.
//
// headParallel says whether this shape must take the head-parallel attention path. It is
// asserted, not assumed: a "parallel" case that silently ran serially would leave the
// workerScratch sets ungated.
func checkPhaseInert(t *testing.T, headParallel bool, run func() []float32) {
	t.Helper()
	if headParallel && numCPU < 2 {
		t.Skip("one CPU: attention never fans out across heads")
	}
	t.Cleanup(func() { phaseBoundaryHook, scratchPutHook = nil, nil })

	// Reference on a fresh pool, recording whether the head loop fanned out.
	scratchPool = sync.Pool{New: func() any { return &scratch{} }}
	var mu sync.Mutex
	workers := 0
	scratchPutHook = func(s *scratch) {
		mu.Lock()
		workers = max(workers, len(s.workerScratch))
		mu.Unlock()
	}
	ref := run()
	scratchPutHook = nil
	if got := workers > 0; got != headParallel {
		t.Fatalf("head-parallel attention engaged = %v (workers %d), this case needs %v", got, workers, headParallel)
	}

	for _, pv := range poisonF32s {
		phaseBoundaryHook = func(s *scratch) { poisonScratch(s, pv) }
		for rep := range 2 {
			got := run()
			if len(got) != len(ref) {
				t.Fatalf("poison %v run %d: %d outputs, reference %d", pv, rep, len(got), len(ref))
			}
			for i := range ref {
				if math.Float32bits(got[i]) != math.Float32bits(ref[i]) {
					t.Fatalf("poison %v run %d: output[%d] = %v, fresh-scratch reference %v; "+
						"a phase reads scratch written before the boundary",
						pv, rep, i, got[i], ref[i])
				}
			}
		}
		phaseBoundaryHook = nil
	}
}

// Synthetic dimensions. headDim 64 so a lone L=160 forward fans the head loop out
// (mOut·L·hd = 1.6M ≥ attnHeadParallelThreshold) while the per-head matmul stays serial.
const (
	synD, synHeads, synInter, synVocab = 128, 2, 256, 97
	synShort, synLong                  = 17, 160
)

type synth struct{ r *rand.Rand }

func newSynth(seed int64) *synth { return &synth{rand.New(rand.NewSource(seed))} }

func (g *synth) f(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(g.r.NormFloat64()) * 0.08
	}
	return v
}

// ln is a LayerNorm weight: near 1, so the stack stays well-scaled.
func (g *synth) ln(n int) []float32 {
	v := g.f(n)
	for i := range v {
		v[i] += 1
	}
	return v
}

func (g *synth) ids(n int) []int32 {
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32(g.r.Intn(synVocab))
	}
	return ids
}

// synthWeights builds a nomic-family model whose layers take the given applyMLP branches:
// "swiglu", "moe" or "gelu". qkvBias puts a bias on the packed projection of every layer.
func synthWeights(seed int64, kinds []string, qkvBias bool) *Weights {
	g := newSynth(seed)
	const D, I, E = synD, synInter, 4
	act := "swiglu"
	for _, k := range kinds {
		if k == "gelu" {
			act = "gelu" // geluMLP and moeMLP read the activation; SwiGLU layers ignore it
		}
	}
	w := &Weights{
		Cfg: Config{
			VocabSize: synVocab, HiddenDim: D, NumLayers: len(kinds), NumHeads: synHeads,
			IntermediateDim: I, TypeVocabSize: 1, RoPEBase: 10000, LayerNormEpsilon: 1e-5,
			ActivationFunction: act, QKVProjBias: qkvBias, NumExperts: E, MoETopK: 2,
		},
		WordEmb: g.f(synVocab * D), TokenTypeEmb: g.f(D),
		EmbLN_W: g.ln(D), EmbLN_B: g.f(D),
	}
	for _, k := range kinds {
		l := LayerWeights{
			Wqkv: g.f(3 * D * D), OutProj: g.f(D * D), OutProjB: g.f(D),
			Norm1W: g.ln(D), Norm1B: g.f(D), Norm2W: g.ln(D), Norm2B: g.f(D),
		}
		if qkvBias {
			l.WqkvB = g.f(3 * D)
		}
		switch k {
		case "swiglu":
			l.Fc11, l.Fc12, l.Fc2 = g.f(I*D), g.f(I*D), g.f(D*I)
		case "moe":
			l.IsMoE = true
			l.Router, l.ExpW1, l.ExpW2, l.ExpBias = g.f(E*D), g.f(E*I*D), g.f(E*I*D), g.f(D)
		case "gelu":
			l.Fc1, l.Fc1B, l.Fc2, l.Fc2B = g.f(I*D), g.f(I), g.f(D*I), g.f(D)
		default:
			panic("synthWeights: unknown layer kind " + k)
		}
		w.Layers = append(w.Layers, l)
	}
	return w
}

// synthQ8 quantizes a SwiGLU synthWeights the way LoadWeightsQ8 lays it out.
func synthQ8(w *Weights) *WeightsQ8 {
	D, I := w.Cfg.HiddenDim, w.Cfg.IntermediateDim
	q := &WeightsQ8{Cfg: w.Cfg, WordEmb: w.WordEmb, TokenTypeEmb: w.TokenTypeEmb, EmbLN_W: w.EmbLN_W, EmbLN_B: w.EmbLN_B}
	for _, l := range w.Layers {
		q.Layers = append(q.Layers, LayerWeightsQ8{
			Wqkv:    linalg.QuantizeInt8(l.Wqkv, 3*D, D, false),
			OutProj: linalg.QuantizeInt8(l.OutProj, D, D, false),
			Fc11:    linalg.QuantizeInt8(l.Fc11, I, D, false),
			Fc12:    linalg.QuantizeInt8(l.Fc12, I, D, false),
			Fc2:     linalg.QuantizeInt8(l.Fc2, D, I, false),
			Norm1W:  l.Norm1W, Norm1B: l.Norm1B, Norm2W: l.Norm2W, Norm2B: l.Norm2B,
		})
	}
	return q
}

func synthBERT(seed int64) *BERT {
	g := newSynth(seed)
	const D, I, maxPos = synD, synInter, 256
	b := &BERT{
		cfg: bertConfig{VocabSize: synVocab, Hidden: D, Layers: 2, Heads: synHeads, Intermediate: I,
			MaxPos: maxPos, TypeVocab: 2, LNEps: 1e-12, Act: "gelu", ModelType: "bert"},
		wordEmb: g.f(synVocab * D), posEmb: g.f(maxPos * D), typeEmb: g.f(2 * D),
		embLNW: g.ln(D), embLNB: g.f(D), maxSeq: maxPos,
	}
	for range b.cfg.Layers {
		b.layers = append(b.layers, bertLayer{
			Wq: g.f(D * D), Wk: g.f(D * D), Wv: g.f(D * D), Bq: g.f(D), Bk: g.f(D), Bv: g.f(D),
			Wo: g.f(D * D), Bo: g.f(D), AttnLNW: g.ln(D), AttnLNB: g.f(D),
			Wi: g.f(I * D), Bi: g.f(I), Wd: g.f(D * I), Bd: g.f(D), OutLNW: g.ln(D), OutLNB: g.f(D),
		})
	}
	return b
}

func synthGTE(seed int64) *GTE {
	g := newSynth(seed)
	const D, I = synD, synInter
	m := &GTE{
		cfg: gteConfig{VocabSize: synVocab, Hidden: D, Layers: 2, Heads: synHeads, Intermediate: I,
			LNEps: 1e-12, RopeTheta: 10000},
		wordEmb: g.f(synVocab * D), typeEmb: g.f(D), embLNW: g.ln(D), embLNB: g.f(D), maxSeq: 256,
	}
	for range m.cfg.Layers {
		m.layers = append(m.layers, gteLayer{
			Wqkv: g.f(3 * D * D), WqkvB: g.f(3 * D), Wo: g.f(D * D), WoB: g.f(D),
			AttnLNW: g.ln(D), AttnLNB: g.f(D), UpGate: g.f(2 * I * D), Down: g.f(D * I), DownB: g.f(D),
			MLPLNW: g.ln(D), MLPLNB: g.f(D),
		})
	}
	return m
}

func flatten(rows [][]float32) []float32 {
	var out []float32
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

// TestScratchPhases_poisonedBoundaryIsInert is the A-2 gate: every forward, serial and
// head-parallel, single and batched, bit-identical under phase-boundary poisoning.
func TestScratchPhases_poisonedBoundaryIsInert(t *testing.T) {
	g := newSynth(1)
	short, long := g.ids(synShort), g.ids(synLong)
	// Batched: three sequences of different lengths, so padding rows exist.
	batch := [][]int32{g.ids(23), g.ids(9), g.ids(31)}

	mixed := synthWeights(2, []string{"swiglu", "moe", "gelu"}, true)
	swiglu := synthWeights(3, []string{"swiglu", "swiglu"}, false)
	q8 := synthQ8(swiglu)
	bert := synthBERT(4)
	gte := synthGTE(5)

	type tc struct {
		name string
		run  func(ids []int32) []float32 // lone forward
		// fansOut: the long input takes the head-parallel path. selfAttentionQ8 has
		// its own serial head loop with no fan-out, so q8's long input stays serial
		// (asserted, so a future q8 fan-out shows up here and gets gated).
		fansOut bool
	}
	lone := []tc{
		{"nomic/mixed", mixed.forward, true},
		{"nomic/swiglu", swiglu.forward, true},
		{"nomic/tokens", mixed.forwardTokens, true},
		{"q8", q8.forward, false},
		{"bert", func(ids []int32) []float32 { return bert.forward(ids, nil, false) }, true},
		{"bert/clsOnly", func(ids []int32) []float32 { return bert.forward(ids, nil, true) }, true},
		{"gte", func(ids []int32) []float32 { return gte.forward(ids, false) }, true},
		{"gte/clsOnly", func(ids []int32) []float32 { return gte.forward(ids, true) }, true},
	}
	for _, c := range lone {
		t.Run(c.name+"/short", func(t *testing.T) {
			checkPhaseInert(t, false, func() []float32 { return c.run(short) })
		})
		t.Run(c.name+"/long", func(t *testing.T) {
			checkPhaseInert(t, c.fansOut, func() []float32 { return c.run(long) })
		})
	}

	// Batched forwards never fan the head loop out (inflightForwards > 1 under
	// EncodeBatch, and the batched attention has no head-parallel path).
	t.Run("nomic/batched", func(t *testing.T) {
		checkPhaseInert(t, false, func() []float32 { return flatten(swiglu.forwardBatch(batch)) })
	})
	t.Run("q8/batched", func(t *testing.T) {
		checkPhaseInert(t, false, func() []float32 { return flatten(q8.forwardBatch(batch)) })
	})
}

// TestScratchPhases_poisonReachesEveryBuffer pins the poisoner itself: after it runs on a
// fully sized scratch, no phase-scoped buffer may hold a clean value anywhere up to its
// capacity. Without this, a poisoner that skipped a field would make the gate vacuous for
// exactly that field.
func TestScratchPhases_poisonReachesEveryBuffer(t *testing.T) {
	s := &scratch{}
	s.ensureLayer(scratchLayout{packedQKV: true, gate: true, upGate: true}, 8, 16, 32, 2, 8, 8)
	s.ensureWorkerScratch(2, 8, 8, 8)
	s.moeAllScores, s.moeWeight = ensureF32(nil, 8), ensureF32(nil, 8)
	s.moeGathered, s.moeAcc = ensureF32(nil, 8), ensureF32(nil, 8)
	s.moeExpert, s.moeTokens = ensureI32(nil, 8), ensureI32(nil, 8)
	s.ensureDeqW(16, 32)
	poisonScratch(s, 1e4)

	v := reflect.ValueOf(s).Elem()
	var check func(v reflect.Value, path string, top bool)
	check = func(v reflect.Value, path string, top bool) {
		for i := range v.NumField() {
			name := v.Type().Field(i).Name
			fv := reflect.NewAt(v.Field(i).Type(), unsafe.Pointer(v.Field(i).UnsafeAddr())).Elem()
			switch b := fv.Addr().Interface().(type) {
			case *[]float32:
				full := (*b)[:cap(*b)]
				_, exempt := phaseExempt[name]
				if exempt && top {
					if len(full) > 0 && full[0] == 1e4 {
						t.Errorf("%s%s is exempt but was poisoned", path, name)
					}
					continue
				}
				if len(full) == 0 {
					t.Errorf("%s%s was never sized here, so this test cannot check it; size it above", path, name)
				}
				for j, x := range full {
					if x != 1e4 {
						t.Fatalf("%s%s[%d] = %v after poisoning", path, name, j, x)
					}
				}
			case *[]int32:
				for j, x := range (*b)[:cap(*b)] {
					if x != poisonI32 {
						t.Fatalf("%s%s[%d] = %v after poisoning", path, name, j, x)
					}
				}
			case *[]headScratch:
				for j := range *b {
					check(reflect.ValueOf(&(*b)[j]).Elem(), fmt.Sprintf("%s%s[%d].", path, name, j), false)
				}
			}
		}
	}
	check(v, "", true)
}
