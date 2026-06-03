package decoder

import (
	"context"
	"fmt"
	"math"
)

// Model is a loaded Gemma 3 checkpoint plus the compute backend. Goroutine
// safety follows encoder.Model: Weights are immutable after Load; per-
// sequence state (the KV cache) is owned by each Generate call, so distinct
// sequences can run concurrently, but a single KVCache is not shared.
type Model struct {
	w  *Weights
	be Backend
}

// Options configures Load.
type Options struct {
	Backend string // "cpu" (default) or "webgpu"
}

// Load reads a Gemma 3 snapshot (config.json + model.safetensors) from dir
// and selects a backend.
//
// SCAFFOLD: LoadWeights is unimplemented (M1), so this surfaces that error
// while still constructing the backend so the wiring is exercised.
func Load(dir string, opts Options) (*Model, error) {
	be, beErr := NewBackend(opts.Backend)
	// beErr is non-nil for the not-yet-implemented webgpu fallback; keep the
	// (cpu) backend and surface the note rather than abort.
	w, err := LoadWeights(dir)
	if err != nil {
		if be != nil {
			_ = be.Close()
		}
		return nil, err
	}
	if beErr != nil {
		// webgpu requested but fell back — not fatal.
		fmt.Println(beErr)
	}
	return &Model{w: w, be: be}, nil
}

// Config exposes the loaded architecture config.
func (m *Model) Config() *Config { return &m.w.Cfg }

// NewCache allocates a KV cache sized for this model. capHint pre-sizes for
// a known max length (0 = grow on demand).
func (m *Model) NewCache(capHint int) *KVCache {
	c := &m.w.Cfg
	return NewKVCache(c.NumLayers, c.NumKVHeads, c.HeadDim, c.SlidingWindow, capHint)
}

// forward runs one decode step for token id at position cache.Pos() and
// returns the logit vector ([VocabSize]).
//
// SCAFFOLD: spells out the exact Gemma 3 block order so M3 is a fill-in.
// Returns errNotImplemented from the first unimplemented sub-op.
//
//	1. h = Embed[id] * sqrt(HiddenDim)        // embedding scale (invariant)
//	2. for each layer l:
//	     a. n  = rmsNorm(h, PreAttnNorm)
//	     b. a  = causalAttention(l, n, ...)    // GQA + RoPE + QK-norm + cache
//	     c. a  = rmsNorm(a, PostAttnNorm)
//	     d. h  = h + a                          // residual
//	     e. n2 = rmsNorm(h, PreMLPNorm)
//	     f. g  = geGLU(n2, ...)
//	     g. g  = rmsNorm(g, PostMLPNorm)
//	     h. h  = h + g                          // residual
//	3. h = rmsNorm(h, FinalNorm)
//	4. logits = h · Embedᵀ                      // tied LM head (invariant)
func (m *Model) forward(id int, cache *KVCache) ([]float32, error) {
	c := &m.w.Cfg
	h := make([]float32, c.HiddenDim)
	scale := float32(math.Sqrt(float64(c.HiddenDim)))
	emb := m.w.Embed
	if len(emb) == 0 {
		return nil, fmt.Errorf("decoder.forward: weights not loaded %w [M1]", errNotImplemented)
	}
	copy(h, emb[id*c.HiddenDim:(id+1)*c.HiddenDim])
	for i := range h {
		h[i] *= scale
	}
	for l := 0; l < c.NumLayers; l++ {
		lw := &m.w.Layers[l]
		n := append([]float32(nil), h...)
		rmsNorm(n, lw.PreAttnNorm, 1, c.HiddenDim, c.RMSNormEps)
		if _, err := causalAttention(l, n, lw, c, cache, m.be); err != nil {
			return nil, err
		}
		// ... residual + MLP wiring (M3) ...
	}
	return nil, fmt.Errorf("decoder.forward: layer stack %w [M3]", errNotImplemented)
}

// Generate streams generated token ids over the returned channel until EOS,
// a stop id, maxTokens, or ctx cancellation. prompt is already-tokenized
// ids (the demo runs the tokenizer). The channel closes when generation
// ends; check Err after the range loop for a terminal error.
//
// SCAFFOLD: prefills the cache over the prompt then decodes — both via
// forward(), which is unimplemented, so the stream yields nothing and Err
// returns the M-pointer. The control flow (prefill → decode loop → sample →
// EOS/stop/max) is the real M4/M6 shape.
func (m *Model) Generate(ctx context.Context, prompt []int, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	go func() {
		defer close(out)
		cache := m.NewCache(len(prompt) + maxTokens)
		sampler := NewSampler(sp)
		// Prefill: run every prompt token through the cache. The last
		// token's logits seed the first generated token.
		var logits []float32
		for _, id := range prompt {
			l, err := m.forward(id, cache)
			if err != nil {
				g.err = err
				return
			}
			logits = l
		}
		// Decode loop.
		for n := 0; n < maxTokens; n++ {
			select {
			case <-ctx.Done():
				g.err = ctx.Err()
				return
			default:
			}
			next, err := sampler.Sample(logits)
			if err != nil {
				g.err = err
				return
			}
			if m.isStop(next, sp) {
				return
			}
			out <- next
			if logits, err = m.forward(next, cache); err != nil {
				g.err = err
				return
			}
		}
	}()
	return out, g
}

// isStop reports whether id ends generation (EOS or a configured stop).
// SCAFFOLD: EOS id wiring comes from the tokenizer at M2/M6.
func (m *Model) isStop(id int, sp SamplingParams) bool {
	_ = id
	_ = sp
	return false
}

// Generation carries the terminal status of a Generate stream.
type Generation struct{ err error }

// Err returns the error that ended the stream, or nil if it ended cleanly
// (EOS / stop / maxTokens). Read it after the channel closes.
func (g *Generation) Err() error { return g.err }
