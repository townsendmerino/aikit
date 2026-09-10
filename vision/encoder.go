package vision

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
)

// SigLIP / ViT vision encoder (the Gemma 3 vision tower) as a pure-Go forward —
// the P2 piece of goinfer's multimodal.md. It maps preprocessed pixel_values to a
// last_hidden_state, the sequence of patch embeddings the projector turns into
// image tokens. The attention/FFN projections run f32 or int8 W8A8 (LoadEncoder's
// quant flag; the patch-embed conv stays f32); parity is cosine vs the HF
// SiglipVisionModel golden (scripts/oracle/pin_siglip_vision.py) — 1.0 for f32, ~0.9999
// for int8 — the standard the rest of the f32-SIMD attention path meets.
//
// Structure (all reused from the text side's primitives): Conv2d patch embedding
// (as im2col + matmul), learned position embeddings, N pre-LN transformer blocks
// (BIDIRECTIONAL multi-head attention — no causal mask, this is an image — plus a
// gelu-tanh MLP), and a final post-layernorm.

// EncoderConfig mirrors the SiglipVisionConfig fields the forward needs.
type EncoderConfig struct {
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumChannels       int     `json:"num_channels"`
	ImageSize         int     `json:"image_size"`
	PatchSize         int     `json:"patch_size"`
	LayerNormEps      float64 `json:"layer_norm_eps"`
}

// validate rejects a config whose dimensions would divide-by-zero or
// mis-partition at load/Forward (H8): there is no vision equivalent of the text
// encoder's ValidateAssumptions, so without this an absent patch_size ÷0s at
// e.grid, an absent num_attention_heads ÷0s at headDim, and an odd
// hidden/heads split silently leaves output columns zero. Called after config
// parse + defaults, before any dimension is used.
func (c EncoderConfig) validate() error {
	switch {
	case c.HiddenSize <= 0:
		return fmt.Errorf("hidden_size must be > 0, got %d", c.HiddenSize)
	case c.IntermediateSize <= 0:
		return fmt.Errorf("intermediate_size must be > 0, got %d", c.IntermediateSize)
	case c.NumHiddenLayers < 0:
		return fmt.Errorf("num_hidden_layers must be >= 0, got %d", c.NumHiddenLayers)
	case c.NumAttentionHeads <= 0:
		return fmt.Errorf("num_attention_heads must be > 0, got %d", c.NumAttentionHeads)
	case c.HiddenSize%c.NumAttentionHeads != 0:
		return fmt.Errorf("hidden_size %d not divisible by num_attention_heads %d", c.HiddenSize, c.NumAttentionHeads)
	case c.NumChannels <= 0:
		return fmt.Errorf("num_channels must be > 0, got %d", c.NumChannels)
	case c.PatchSize <= 0:
		return fmt.Errorf("patch_size must be > 0, got %d", c.PatchSize)
	case c.ImageSize <= 0:
		return fmt.Errorf("image_size must be > 0, got %d", c.ImageSize)
	case c.ImageSize%c.PatchSize != 0:
		return fmt.Errorf("image_size %d not divisible by patch_size %d", c.ImageSize, c.PatchSize)
	}
	return nil
}

type encLayer struct {
	ln1w, ln1b     []float32
	qw, kw, vw, ow linalg.WeightMat // [hidden,hidden] matmul weights (f32 or int8)
	qb, kb, vb, ob []float32        // biases stay f32
	ln2w, ln2b     []float32
	fc1w, fc2w     linalg.WeightMat // [inter,hidden] / [hidden,inter] matmul weights
	fc1b, fc2b     []float32
}

// Encoder is a loaded SigLIP vision tower.
type Encoder struct {
	Cfg              EncoderConfig
	grid, numPatches int
	patchW           []float32 // [hidden, C*P*P] (Conv2d weight, kept f32 — input embedding)
	patchB           []float32 // [hidden]
	posEmb           []float32 // [numPatches, hidden]
	layers           []encLayer
	postLNw, postLNb []float32
	resident         ResidentEncoder // device-resident GPU forward (EnableResident); nil = CPU path
}

// LoadEncoder reads a SigLIP vision checkpoint (config.json + model.safetensors)
// and returns a ready Encoder. Weights are copied out, so the safetensors file is
// closed before return (no retained mmap).
func LoadEncoder(dir string, quant bool) (*Encoder, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("vision: read config: %w", err)
	}
	// The tiny pinned tower's config.json IS the SigLIP EncoderConfig (flat); a
	// real HF VL checkpoint nests it under "vision_config". Prefer the nested one.
	var wrap struct {
		EncoderConfig
		VisionConfig *EncoderConfig `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil {
		return nil, fmt.Errorf("vision: parse config: %w", err)
	}
	cfg := wrap.EncoderConfig
	if wrap.VisionConfig != nil {
		cfg = *wrap.VisionConfig
	}
	if cfg.LayerNormEps == 0 {
		cfg.LayerNormEps = 1e-6
	}
	if cfg.NumChannels == 0 {
		cfg.NumChannels = 3 // SigLIP is RGB; real vision_config omits num_channels
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("vision: %w", err)
	}
	st, err := openWeights(dir)
	if err != nil {
		return nil, fmt.Errorf("vision: open safetensors: %w", err)
	}
	defer st.Close()

	e := &Encoder{Cfg: cfg}
	e.grid = cfg.ImageSize / cfg.PatchSize
	e.numPatches = e.grid * e.grid
	// "" for the tiny stripped tower, "vision_tower.vision_model." inside a real
	// gemma-3-4b-it (where the SigLIP tower lives in the model shards).
	pfx := tensorPrefix(st, "embeddings.patch_embedding.weight", "vision_tower.vision_model.")
	// get reads a tensor and, when want dims are given, shape-checks it (H7):
	// without this a mismatched/hostile checkpoint panics deep in QuantizeRowsInt8
	// or MatmulBT at load/Forward instead of returning a clean error. Shapes
	// follow HF SiglipVisionModel (Linear weights [out,in], the Conv2d
	// patch-embed [hidden,C,P,P], position_embedding [numPatches,hidden], 1-D
	// biases/LayerNorms) — the parity test (testdata/siglip-tiny) is the gate.
	get := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		var v []float32
		v, err = st.TensorF32(pfx+name, want...)
		if err != nil {
			return nil
		}
		return append([]float32(nil), v...) // copy out so st can close
	}
	hidden, inter := cfg.HiddenSize, cfg.IntermediateSize
	// qm wraps a matmul weight as f32 or int8 (W8A8). Attention/FFN projections
	// quantize under -vision-quant; the patch-embed conv stays f32 (input
	// embedding — quant error there propagates through every layer).
	qm := func(name string, rows, cols int) linalg.WeightMat {
		w := get(name, rows, cols)
		if err != nil {
			return linalg.WeightMat{}
		}
		return newQMat(w, rows, cols, quant)
	}
	numChan := cfg.NumChannels
	e.patchW = get("embeddings.patch_embedding.weight", hidden, numChan, cfg.PatchSize, cfg.PatchSize) // Conv2d
	e.patchB = get("embeddings.patch_embedding.bias", hidden)
	e.posEmb = get("embeddings.position_embedding.weight", e.numPatches, hidden)
	e.layers = make([]encLayer, cfg.NumHiddenLayers)
	for l := range e.layers {
		p := fmt.Sprintf("encoder.layers.%d.", l)
		lw := &e.layers[l]
		lw.ln1w, lw.ln1b = get(p+"layer_norm1.weight", hidden), get(p+"layer_norm1.bias", hidden)
		lw.qw, lw.qb = qm(p+"self_attn.q_proj.weight", hidden, hidden), get(p+"self_attn.q_proj.bias", hidden)
		lw.kw, lw.kb = qm(p+"self_attn.k_proj.weight", hidden, hidden), get(p+"self_attn.k_proj.bias", hidden)
		lw.vw, lw.vb = qm(p+"self_attn.v_proj.weight", hidden, hidden), get(p+"self_attn.v_proj.bias", hidden)
		lw.ow, lw.ob = qm(p+"self_attn.out_proj.weight", hidden, hidden), get(p+"self_attn.out_proj.bias", hidden)
		lw.ln2w, lw.ln2b = get(p+"layer_norm2.weight", hidden), get(p+"layer_norm2.bias", hidden)
		lw.fc1w, lw.fc1b = qm(p+"mlp.fc1.weight", inter, hidden), get(p+"mlp.fc1.bias", inter)
		lw.fc2w, lw.fc2b = qm(p+"mlp.fc2.weight", hidden, inter), get(p+"mlp.fc2.bias", hidden)
	}
	e.postLNw, e.postLNb = get("post_layernorm.weight", hidden), get("post_layernorm.bias", hidden)
	if err != nil {
		return nil, fmt.Errorf("vision: load weights: %w", err)
	}
	return e, nil
}

// Forward runs the encoder on pixel_values [NumChannels*ImageSize*ImageSize]
// (a single image, CHW order — the preprocess output) and returns last_hidden_state
// [numPatches * HiddenSize], row-major over patches in (row, col) grid order.
//
// Concurrent Forward calls are safe, but NOT concurrent with EnableResident or
// Close: those write e.resident without synchronization, so a Forward racing
// one may read a torn pointer. Enable/close the resident backend before sharing
// the Encoder across goroutines.
func (e *Encoder) Forward(pixels []float32) ([]float32, error) {
	c := e.Cfg
	want := c.NumChannels * c.ImageSize * c.ImageSize
	if len(pixels) != want {
		return nil, fmt.Errorf("vision: pixels len %d, want %d (%d×%d×%d)", len(pixels), want, c.NumChannels, c.ImageSize, c.ImageSize)
	}
	// Device-resident GPU path (EnableResident): im2col on the host, the whole
	// transformer on the device. Same numerics (W8A8), ~9× faster on a real model.
	if e.resident != nil {
		patches, err := e.GridPatches(pixels)
		if err != nil {
			return nil, err
		}
		return e.resident.ForwardPatches(patches)
	}
	hidden, np := c.HiddenSize, e.numPatches
	cpp := c.NumChannels * c.PatchSize * c.PatchSize

	// 1. im2col patch extraction in the Conv2d weight's (c,kh,kw) order, patches in
	// (gh,gw) row-major — matching HF's embeddings.flatten(2).transpose. Shared
	// with the resident path via GridPatches (was duplicated inline here).
	patches, err := e.GridPatches(pixels)
	if err != nil {
		return nil, err
	}
	// patch embed: h[np,hidden] = patches[np,cpp] · patchW[hidden,cpp]ᵀ + bias, + posEmb
	h := make([]float32, np*hidden)
	linalg.MatmulBT(patches, e.patchW, h, np, cpp, hidden)
	addBias(h, e.patchB, np, hidden)
	for i := range h {
		h[i] += e.posEmb[i]
	}

	// Allocate every per-layer scratch buffer ONCE (all layers share one shape) and
	// reuse across the layer loop. The old code re-make'd n1/att/o/n2/mid/mlp plus
	// attention's q/k/v/qh/kh/vt/scores/oh every layer — at SigLIP-so400m
	// (np=4096, 27 layers) ≈290 MB allocated and discarded PER LAYER, ~7.9 GB per
	// image, all trivially reusable (audit #5). Bit-identical: same ops, distinct
	// buffers. One scratch per Forward preserves the documented concurrent-Forward
	// safety (each call has its own).
	inter := c.IntermediateSize
	hd := hidden / c.NumAttentionHeads
	s := newEncScratch(np, hidden, inter, hd, c.NumAttentionHeads)
	for l := range e.layers {
		lw := &e.layers[l]
		// attention block (pre-LN, residual)
		layerNormInto(s.n1, h, lw.ln1w, lw.ln1b, np, hidden, c.LayerNormEps)
		if err := e.attentionInto(s.att, s.n1, lw, np, s); err != nil {
			return nil, err
		}
		lw.ow.MatmulBTInto(&s.ws, s.att, s.o, np)
		addBias(s.o, lw.ob, np, hidden)
		for i := range h {
			h[i] += s.o[i]
		}
		// MLP block (pre-LN, residual): fc2(geluTanh(fc1(x)))
		layerNormInto(s.n2, h, lw.ln2w, lw.ln2b, np, hidden, c.LayerNormEps)
		lw.fc1w.MatmulBTInto(&s.ws, s.n2, s.mid, np)
		addBias(s.mid, lw.fc1b, np, inter)
		geluTanh(s.mid)
		lw.fc2w.MatmulBTInto(&s.ws, s.mid, s.mlp, np)
		addBias(s.mlp, lw.fc2b, np, hidden)
		for i := range h {
			h[i] += s.mlp[i]
		}
	}
	return layerNorm(h, e.postLNw, e.postLNb, np, hidden, c.LayerNormEps), nil
}

// encScratch holds the SigLIP encoder's per-layer working buffers, allocated once
// per Forward and reused across layers (audit #5). All layers share one shape.
type encScratch struct {
	n1, att, o, n2, mid, mlp []float32        // block buffers
	q, k, v                  []float32        // attention projections [np,hidden]
	ws                       linalg.Workspace // reused across the WeightMat projections (audit #12)
	headPool                 []encHeadScratch // P6: fused-attention worker slots, len = fan-out width
	loFull, hiFull           []int            // shared across every head/worker: bidirectional, so
	// every row's key range is the whole tile — [0,np-1] — computed once per Forward, not per head.
}

// encHeadScratch is one fused-attention worker's private scratch: gather buffers sized to the
// caller's tile width kt (SigLIP: np, one segment per image; Qwen: maxSeg, the longest
// cu_seqlens run — AttendTileFused's buffers are prefix slices, so sizing to the max and using a
// shorter prefix per call is exactly the reuse contract FusedAttnScratch documents), so
// AttendTileFused's Fits(kt, hd) check always succeeds for the calls each tower makes (see
// attentionInto's doc comment on why a decline is provably unreachable there) — and a private
// serial matmul Workspace, so this worker's head-level fan-out doesn't nest inside MatmulBT's own
// column-level fan-out (the identical reasoning goinfer's decoder/scratch.go documents for its
// own per-worker Workspace). hi is private, not shared, because Qwen recomputes it per segment
// (segment length varies within one Forward) — a shared buffer would make concurrent workers
// write it in the same call, a data race even though every writer would agree on the value.
type encHeadScratch struct {
	qh, kh, vBlk, ch []float32
	hi               []int
	fused            *linalg.FusedAttnScratch
	mmWS             *linalg.Workspace
}

func newEncHeadScratch(kt, hd int) encHeadScratch {
	ws := &linalg.Workspace{}
	ws.SetThreshold(1 << 62) // never fan out internally — see the type doc comment
	return encHeadScratch{
		qh: make([]float32, kt*hd), kh: make([]float32, kt*hd), vBlk: make([]float32, kt*hd),
		ch:    make([]float32, kt*hd),
		hi:    make([]int, kt),
		fused: linalg.NewFusedAttnScratch(kt, hd),
		mmWS:  ws,
	}
}

func newEncScratch(np, hidden, inter, hd, nH int) *encScratch {
	workers := max(min(runtime.GOMAXPROCS(0), nH), 1)
	pool := make([]encHeadScratch, workers)
	for i := range pool {
		pool[i] = newEncHeadScratch(np, hd)
	}
	loFull, hiFull := make([]int, np), make([]int, np)
	for i := range hiFull {
		hiFull[i] = np - 1 // loFull stays zero-valued: every row attends to every patch
	}
	return &encScratch{
		n1: make([]float32, np*hidden), att: make([]float32, np*hidden),
		o: make([]float32, np*hidden), n2: make([]float32, np*hidden),
		mid: make([]float32, np*inter), mlp: make([]float32, np*hidden),
		q: make([]float32, np*hidden), k: make([]float32, np*hidden),
		v:        make([]float32, np*hidden),
		headPool: pool, loFull: loFull, hiFull: hiFull,
	}
}

// attentionInto runs bidirectional multi-head self-attention (no causal mask) over the np
// patches, via linalg's fused (FlashAttention-style) schedule (P6 — aikit/linalg/fusedattn.go),
// the SAME primitive goinfer's own text-decoder prefill already uses, fanned out across heads
// with a private Workspace per worker (mirroring goinfer's decoder/forwardn.go
// attendBatchedHeads exactly, down to the contiguous-head-run split and the serial-Workspace
// reasoning — see encHeadScratch's doc comment). Replaces the old per-head materialized
// QKᵀ→softmax→scores·V loop; f32 throughout remains the right precision here (HF runs SigLIP in
// bf16/f16, far less precise than this).
//
// AttendTileFused declining (returning false) is possible in general — see its own doc comment —
// but provably UNREACHABLE at this call site: every encHeadScratch's FusedAttnScratch is sized
// via NewFusedAttnScratch(np, hd) for EXACTLY the (np, hd) this Forward call uses, and mm is
// always non-nil, so Fits(np, hd) always holds. A decline here means the scratch-sizing
// invariant above was violated, not a normal runtime condition, so it is reported as an error
// (parity gate: TestSiglipEncoder_parity) rather than silently falling back to a different
// numeric path.
func (e *Encoder) attentionInto(att, x []float32, lw *encLayer, np int, s *encScratch) error {
	hidden, nH := e.Cfg.HiddenSize, e.Cfg.NumAttentionHeads
	hd := hidden / nH
	scale := 1.0 / math.Sqrt(float64(hd))
	lw.qw.MatmulBTInto(&s.ws, x, s.q, np)
	addBias(s.q, lw.qb, np, hidden)
	lw.kw.MatmulBTInto(&s.ws, x, s.k, np)
	addBias(s.k, lw.kb, np, hidden)
	lw.vw.MatmulBTInto(&s.ws, x, s.v, np)
	addBias(s.v, lw.vb, np, hidden)

	attendHead := func(ws *encHeadScratch, mm func(a, b, dst []float32, M, K, N int), head int) error {
		off := head * hd
		linalg.GatherVBlockMajor(ws.kh, ws.vBlk, s.k, s.v, head, hd, hidden, np)
		for i := range np { // Q gather: not covered by GatherVBlockMajor (K/V only)
			copy(ws.qh[i*hd:(i+1)*hd], s.q[i*hidden+off:i*hidden+off+hd])
		}
		if !linalg.AttendTileFusedContractExp(mm, ws.qh, ws.kh, ws.vBlk, ws.ch, ws.fused, np, hd, np, scale, s.loFull, s.hiFull) {
			return fmt.Errorf("vision: AttendTileFused declined for np=%d hd=%d — scratch sizing invariant violated", np, hd)
		}
		for i := range np {
			copy(att[i*hidden+off:i*hidden+off+hd], ws.ch[i*hd:(i+1)*hd])
		}
		return nil
	}

	workers := len(s.headPool)
	if workers <= 1 {
		ws := &s.headPool[0]
		for head := range nH {
			if err := attendHead(ws, linalg.MatmulBT, head); err != nil {
				return err
			}
		}
		return nil
	}
	var wg sync.WaitGroup
	errs := make([]error, workers)
	headsPer := (nH + workers - 1) / workers
	for w := range workers {
		h0, h1 := w*headsPer, min((w+1)*headsPer, nH)
		if h0 >= h1 {
			continue
		}
		wg.Add(1)
		go func(w, h0, h1 int) {
			defer wg.Done()
			ws := &s.headPool[w]
			for head := h0; head < h1; head++ {
				if err := attendHead(ws, ws.mmWS.MatmulBT, head); err != nil {
					errs[w] = err
					return
				}
			}
		}(w, h0, h1)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// --- small f32 helpers (LayerNorm is standard — mean/var — not RMS) ---

// layerNormInto writes the normalized rows into dst (reused across layers, audit
// #5). layerNorm is the allocating wrapper for one-off callers (the final post-LN).
func layerNorm(x, w, b []float32, rows, dim int, eps float64) []float32 {
	out := make([]float32, rows*dim)
	layerNormInto(out, x, w, b, rows, dim, eps)
	return out
}

// Row-split across cores (audit M-10). Bit-identical: each row's f64 mean and
// variance folds run in the same order they did serially, and rows never
// interact — the split re-associates nothing.
func layerNormInto(out, x, w, b []float32, rows, dim int, eps float64) {
	parallelRows(rows, rows*dim, func(start, end int) {
		layerNormRows(out, x, w, b, start, end, dim, eps)
	})
}

func layerNormRows(out, x, w, b []float32, start, end, dim int, eps float64) {
	for r := start; r < end; r++ {
		xr := x[r*dim : r*dim+dim]
		var mean float64
		for _, val := range xr {
			mean += float64(val)
		}
		mean /= float64(dim)
		var variance float64
		for _, val := range xr {
			d := float64(val) - mean
			variance += d * d
		}
		variance /= float64(dim)
		inv := 1.0 / math.Sqrt(variance+eps)
		dst := out[r*dim : r*dim+dim]
		for d := range dim {
			dst[d] = float32((float64(xr[d])-mean)*inv)*w[d] + b[d]
		}
	}
}

// geluTanh applies SigLIP's gelu_pytorch_tanh activation in place.
//
// linalg's float32 kernel, not f64 math.Tanh (perf-campaign item 13). SigLIP is
// the heaviest transcendental consumer in the kit — a so400m tower at 4096
// patches issues billions of these per image — and math.Tanh is a branchy scalar
// f64 routine. Not bit-identical; contract is absolute error ≤1e-06, gated by
// linalg's TestGELUTanhF32_accuracy and end-to-end by TestSiglipEncoder_parity.
func geluTanh(x []float32) {
	// Chunk-split across cores (audit M-10): elementwise and in place, so the
	// split is numerically inert. This is the heaviest transcendental pass in
	// the kit — 476M elements per so400m image — and it ran on one goroutine.
	parallelChunks(len(x), func(lo, hi int) {
		linalg.GELUTanhContractInto(x[lo:hi], x[lo:hi])
	})
}

func addBias(x, bias []float32, rows, dim int) {
	for r := range rows {
		dst := x[r*dim : r*dim+dim]
		for d := range dim {
			dst[d] += bias[d]
		}
	}
}
