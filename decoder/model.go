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
	w      *Weights
	be     Backend
	eosIDs []int // end-of-sequence ids from config (generation stops on these)
}

// Options configures Load.
type Options struct {
	Backend string // "cpu" (default) or "webgpu"
}

// Load reads a Gemma 3 snapshot (config.json + model.safetensors) from dir
// and selects a backend. The forward pass (M3) is implemented; the CPU
// backend is the default and the only one wired (webgpu falls back to CPU).
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
	return &Model{w: w, be: be, eosIDs: w.Cfg.EOSIDs()}, nil
}

// Config exposes the loaded architecture config.
func (m *Model) Config() *Config { return &m.w.Cfg }

// NewCache allocates a KV cache sized for this model. capHint pre-sizes for
// a known max length (0 = grow on demand).
func (m *Model) NewCache(capHint int) *KVCache {
	c := &m.w.Cfg
	return NewKVCache(c.NumLayers, c.NumKVHeads, c.HeadDim, c.SlidingWindow, capHint)
}

// runLayers advances one decode step for token id at position cache.Pos():
// it embeds the token, runs the full Gemma 3 block stack (appending this
// position's K/V to the cache), and returns the residual-stream hidden state
// after the final layer — BEFORE the final norm and LM head. Splitting it out
// lets prefill skip the (vocab-sized) LM head on every token but the last.
//
//  1. h = Embed[id] * sqrt(HiddenDim)        // embedding scale (invariant)
//  2. for each layer l:                       // Gemma "sandwich" norms
//     a. n  = rmsNorm(h, PreAttnNorm)
//     b. a  = causalAttention(l, n, ...)    // GQA + RoPE + QK-norm + cache
//     c. a  = rmsNorm(a, PostAttnNorm)
//     d. h  = h + a                          // residual
//     e. n2 = rmsNorm(h, PreMLPNorm)
//     f. g  = geGLU(n2, ...)
//     g. g  = rmsNorm(g, PostMLPNorm)
//     h. h  = h + g                          // residual
func (m *Model) runLayers(id int, cache *KVCache) ([]float32, error) {
	c := &m.w.Cfg
	if len(m.w.Embed) == 0 {
		return nil, fmt.Errorf("decoder.forward: weights not loaded %w [M1]", errNotImplemented)
	}
	h := make([]float32, c.HiddenDim)
	copy(h, m.w.Embed[id*c.HiddenDim:(id+1)*c.HiddenDim])
	// Embedding scale. NOTE: HF computes this normalizer as sqrt(hidden) cast
	// to the model's dtype — bf16 for a bf16 checkpoint (≈25.25 here) — then
	// multiplies. We use the f32 value (≈25.2982). It matches our parity gate
	// because the very next op (PreAttnNorm RMSNorm) divides out a global
	// scalar, so the difference only survives in the residual and stays well
	// under the ≥1−1e-4 cosine bar. If that bar is ever tightened past ~1e-5,
	// round `scale` to bf16 first to match HF exactly. See M3-forward.md.
	scale := float32(math.Sqrt(float64(c.HiddenDim)))
	for i := range h {
		h[i] *= scale
	}
	for l := 0; l < c.NumLayers; l++ {
		lw := &m.w.Layers[l]
		n := append([]float32(nil), h...)
		rmsNorm(n, lw.PreAttnNorm, 1, c.HiddenDim, c.RMSNormEps)
		a, err := causalAttention(l, n, lw, c, cache, m.be)
		if err != nil {
			return nil, err
		}
		rmsNorm(a, lw.PostAttnNorm, 1, c.HiddenDim, c.RMSNormEps)
		for i := range h {
			h[i] += a[i]
		}
		n2 := append([]float32(nil), h...)
		rmsNorm(n2, lw.PreMLPNorm, 1, c.HiddenDim, c.RMSNormEps)
		g, err := geGLU(n2, lw, c, m.be)
		if err != nil {
			return nil, err
		}
		rmsNorm(g, lw.PostMLPNorm, 1, c.HiddenDim, c.RMSNormEps)
		for i := range h {
			h[i] += g[i]
		}
	}
	return h, nil
}

// forward runs runLayers then the final norm + tied LM head, returning the
// logit vector ([VocabSize]) for the next token. logits = rmsNorm(h)·Embedᵀ —
// the embedding table doubles as the output projection (tied weights), and
// Gemma 3 has no final logit soft-capping.
func (m *Model) forward(id int, cache *KVCache) ([]float32, error) {
	c := &m.w.Cfg
	h, err := m.runLayers(id, cache)
	if err != nil {
		return nil, err
	}
	rmsNorm(h, m.w.FinalNorm, 1, c.HiddenDim, c.RMSNormEps)
	logits := make([]float32, c.VocabSize)
	m.be.MatmulBT(h, m.w.Embed, logits, 1, c.HiddenDim, c.VocabSize)
	return logits, nil
}

// Generate streams generated token ids over the returned channel until EOS,
// a stop id, maxTokens, or ctx cancellation. prompt is already-tokenized
// ids (the demo runs the tokenizer). The channel closes when generation
// ends; check Err after the range loop for a terminal error.
//
// The forward pass (prefill + per-step decode) is implemented (M3). Remaining
// for M4/M6: non-greedy sampling (Sampler.Sample still stubs temp/top-k/top-p)
// and isStop's EOS/stop-id wiring — so today this greedy-decodes to maxTokens.
func (m *Model) Generate(ctx context.Context, prompt []int, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	go func() {
		defer close(out)
		if len(prompt) == 0 {
			g.err = fmt.Errorf("decoder.Generate: empty prompt")
			return
		}
		cache := m.NewCache(len(prompt) + maxTokens)
		sampler := NewSampler(sp)
		// Prefill: run every prompt token through the cache. Only the last
		// token needs logits (to seed the first generated token), so the rest
		// skip the vocab-sized LM head via runLayers.
		for _, id := range prompt[:len(prompt)-1] {
			if _, err := m.runLayers(id, cache); err != nil {
				g.err = err
				return
			}
		}
		logits, err := m.forward(prompt[len(prompt)-1], cache)
		if err != nil {
			g.err = err
			return
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

// isStop reports whether id ends generation: a checkpoint EOS id (from
// config) or a caller-supplied stop id (SamplingParams.StopIDs, e.g.
// <end_of_turn> for chat).
func (m *Model) isStop(id int, sp SamplingParams) bool {
	for _, e := range m.eosIDs {
		if id == e {
			return true
		}
	}
	for _, s := range sp.StopIDs {
		if id == s {
			return true
		}
	}
	return false
}

// Generation carries the terminal status of a Generate stream.
type Generation struct{ err error }

// Err returns the error that ended the stream, or nil if it ended cleanly
// (EOS / stop / maxTokens). Read it after the channel closes.
func (g *Generation) Err() error { return g.err }
