package vision

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// Qwen2.5-VL vision tower — aikit's second ViT family (after SigLIP / encoder.go),
// the Qwen2.5-VL `.visual` submodule as a pure-Go fp32 forward. Where SigLIP is
// fixed-resolution (896×896 → 256 tokens, learned absolute pos, LayerNorm,
// gelu-tanh MLP), this is DYNAMIC-resolution: pre-flattened patches + grid_thw,
// 2D rotary, RMSNorm, windowed + full attention, a gated SiLU MLP, and a
// spatial-merge patch merger. Parity is cosine vs the HF
// Qwen2_5_VisionTransformerPretrainedModel golden (scripts/pin_qwen25vl_vision.py),
// gated in two stages: the ViT pre-merge hidden and the merged image features.
//
// Forward takes pre-patchified input (the goinfer P5.3 preprocessor does
// image→pixel_values+grid_thw via smart-resize upstream), not a CHW image — so the
// encoder is fed pixel_values [n_patches, patch_dim] + per-image (t,h,w) grids.
//
// Added ADDITIVELY: SigLIP's Encoder/LoadEncoder are untouched. The qmat W8A8
// wrapper is reused for the projections (the patch-embed matmul stays f32); the
// resident-GPU seam is a follow-on (the fp32 CPU path is the v1 deliverable).

// QwenEncoderConfig mirrors the HF Qwen2_5_VLVisionConfig fields the forward needs.
type QwenEncoderConfig struct {
	Depth               int    `json:"depth"`
	HiddenSize          int    `json:"hidden_size"`
	IntermediateSize    int    `json:"intermediate_size"`
	NumHeads            int    `json:"num_heads"`
	InChans             int    `json:"in_chans"`
	PatchSize           int    `json:"patch_size"`
	SpatialMergeSize    int    `json:"spatial_merge_size"`
	TemporalPatchSize   int    `json:"temporal_patch_size"`
	OutHiddenSize       int    `json:"out_hidden_size"`
	WindowSize          int    `json:"window_size"`
	FullattBlockIndexes []int  `json:"fullatt_block_indexes"`
	HiddenAct           string `json:"hidden_act"`
}

// validate rejects a config whose dimensions would divide-by-zero or
// mis-partition at load/Forward (H8): head_dim = hidden/num_heads, the
// merge-unit groups = n_patches/spatial_merge_size², and the window grid
// vmws = window_size/spatial_merge_size/patch_size all ÷0 on an absent field.
// Called after config parse + defaults, before any dimension is used.
func (c QwenEncoderConfig) validate() error {
	switch {
	case c.HiddenSize <= 0:
		return fmt.Errorf("hidden_size must be > 0, got %d", c.HiddenSize)
	case c.IntermediateSize <= 0:
		return fmt.Errorf("intermediate_size must be > 0, got %d", c.IntermediateSize)
	case c.Depth < 0:
		return fmt.Errorf("depth must be >= 0, got %d", c.Depth)
	case c.NumHeads <= 0:
		return fmt.Errorf("num_heads must be > 0, got %d", c.NumHeads)
	case c.HiddenSize%c.NumHeads != 0:
		return fmt.Errorf("hidden_size %d not divisible by num_heads %d", c.HiddenSize, c.NumHeads)
	case (c.HiddenSize/c.NumHeads)%4 != 0:
		// The rotary path derives rdim = head_dim/2 and len(inv_freq) =
		// head_dim/4; a head_dim not divisible by 4 makes rdim/inv_freq degenerate
		// and forwardViT divides by zero at first Forward. (num_heads > 0 and
		// hidden%num_heads == 0 are guaranteed above.)
		return fmt.Errorf("head_dim %d (hidden_size/num_heads) must be divisible by 4 for rotary", c.HiddenSize/c.NumHeads)
	case c.InChans <= 0:
		return fmt.Errorf("in_chans must be > 0, got %d", c.InChans)
	case c.PatchSize <= 0:
		return fmt.Errorf("patch_size must be > 0, got %d", c.PatchSize)
	case c.TemporalPatchSize <= 0:
		// patch_dim = in_chans·temporal·patch² feeds the patch-embed matmul; a 0
		// makes it zero-wide → an all-zero embedding, silently.
		return fmt.Errorf("temporal_patch_size must be > 0, got %d", c.TemporalPatchSize)
	case c.SpatialMergeSize <= 0:
		return fmt.Errorf("spatial_merge_size must be > 0, got %d", c.SpatialMergeSize)
	case c.OutHiddenSize <= 0:
		return fmt.Errorf("out_hidden_size must be > 0, got %d", c.OutHiddenSize)
	case c.WindowSize < c.SpatialMergeSize*c.PatchSize:
		return fmt.Errorf("window_size %d must be >= spatial_merge_size*patch_size = %d (else the window grid divides by zero)",
			c.WindowSize, c.SpatialMergeSize*c.PatchSize)
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		// mlp() hardcodes SiLU; a checkpoint declaring gelu/quick_gelu would
		// silently run the wrong activation (the H8 "known activation" check).
		return fmt.Errorf("hidden_act %q unsupported (silu only)", c.HiddenAct)
	}
	return nil
}

type qwenBlock struct {
	norm1w     []float32 // RMSNorm (weight only)
	qkvw       qmat      // [3*hidden, hidden] fused
	qkvb       []float32 // [3*hidden]
	projw      qmat      // [hidden, hidden]
	projb      []float32
	norm2w     []float32 // RMSNorm
	gatew, upw qmat      // [inter, hidden]
	gateb, upb []float32
	downw      qmat // [hidden, inter]
	downb      []float32
}

// QwenVisionEncoder is a loaded Qwen2.5-VL vision tower (dynamic resolution).
type QwenVisionEncoder struct {
	Cfg        QwenEncoderConfig
	patchW     []float32 // [hidden, patch_dim] (Conv3d weight flattened, kept f32)
	blocks     []qwenBlock
	mergerLNw  []float32 // merger.ln_q RMSNorm weight [hidden]
	merger0w   qmat      // merger.mlp.0 [hidden*merge², hidden*merge²]
	merger0b   []float32
	merger2w   qmat // merger.mlp.2 [out_hidden, hidden*merge²]
	merger2b   []float32
	rotInvFreq []float32 // head_dim/4 rotary frequencies
}

const qwenRotaryTheta = 10000.0 // Qwen2_5_VisionRotaryEmbedding default theta

// LoadQwenVisionEncoder reads a Qwen2.5-VL checkpoint (config.json vision_config +
// safetensors) and returns a ready encoder. Weights are copied out, so the
// safetensors file is closed before return (no retained mmap). quant wraps the
// projections as int8 W8A8 (the patch-embed matmul stays f32); the fp32 parity gate
// runs quant=false.
func LoadQwenVisionEncoder(dir string, quant bool) (*QwenVisionEncoder, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("vision: read config: %w", err)
	}
	// The vision config is nested under "vision_config" in a real VL checkpoint; a
	// stripped tower could carry it flat — prefer the nested one when present.
	var wrap struct {
		QwenEncoderConfig
		VisionConfig *QwenEncoderConfig `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil {
		return nil, fmt.Errorf("vision: parse config: %w", err)
	}
	cfg := wrap.QwenEncoderConfig
	if wrap.VisionConfig != nil {
		cfg = *wrap.VisionConfig
	}
	if cfg.InChans == 0 {
		cfg.InChans = 3
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("vision: %w", err)
	}
	st, err := openWeights(dir)
	if err != nil {
		return nil, fmt.Errorf("vision: open safetensors: %w", err)
	}
	defer st.Close()

	e := &QwenVisionEncoder{Cfg: cfg}
	// "visual." in the tiny checkpoint, "model.visual." inside a full HF VL
	// checkpoint (where the tower lives alongside the language model shards).
	pfx := qwenTensorPrefix(st)
	// get reads a tensor and, when want dims are given, shape-checks it (H7):
	// otherwise a mismatched/hostile checkpoint panics deep in QuantizeRowsInt8 or
	// MatmulBT at load/Forward instead of returning an error. Shapes follow HF
	// Qwen2.5-VL (fused-QKV/Linear weights [out,in], RMSNorms [hidden], 1-D
	// biases); patch_embed.proj is a 5-D Conv3d and is left unchecked. The parity
	// test (testdata/qwen25vl-vision-tiny) is the gate.
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
	qm := func(name string, rows, cols int) qmat {
		w := get(name, rows, cols)
		if err != nil {
			return qmat{}
		}
		return newQMat(w, rows, cols, quant)
	}
	// Conv3d weight [hidden, in_chans, temporal, patch, patch] — shape-checked
	// like every other tensor (H7), so a mismatch is a clean load-time error, not
	// a later matmul panic / silent prefix.
	e.patchW = get("patch_embed.proj.weight", hidden, cfg.InChans, cfg.TemporalPatchSize, cfg.PatchSize, cfg.PatchSize)
	e.blocks = make([]qwenBlock, cfg.Depth)
	for i := range e.blocks {
		p := fmt.Sprintf("blocks.%d.", i)
		b := &e.blocks[i]
		b.norm1w = get(p+"norm1.weight", hidden)
		b.qkvw, b.qkvb = qm(p+"attn.qkv.weight", 3*hidden, hidden), get(p+"attn.qkv.bias", 3*hidden)
		b.projw, b.projb = qm(p+"attn.proj.weight", hidden, hidden), get(p+"attn.proj.bias", hidden)
		b.norm2w = get(p+"norm2.weight", hidden)
		b.gatew, b.gateb = qm(p+"mlp.gate_proj.weight", inter, hidden), get(p+"mlp.gate_proj.bias", inter)
		b.upw, b.upb = qm(p+"mlp.up_proj.weight", inter, hidden), get(p+"mlp.up_proj.bias", inter)
		b.downw, b.downb = qm(p+"mlp.down_proj.weight", hidden, inter), get(p+"mlp.down_proj.bias", hidden)
	}
	mh := hidden * cfg.SpatialMergeSize * cfg.SpatialMergeSize
	e.mergerLNw = get("merger.ln_q.weight", hidden)
	e.merger0w, e.merger0b = qm("merger.mlp.0.weight", mh, mh), get("merger.mlp.0.bias", mh)
	e.merger2w, e.merger2b = qm("merger.mlp.2.weight", cfg.OutHiddenSize, mh), get("merger.mlp.2.bias", cfg.OutHiddenSize)
	if err != nil {
		return nil, fmt.Errorf("vision: load weights: %w", err)
	}

	// Rotary inv_freq over head_dim/2 (the Qwen2_5_VisionRotaryEmbedding dim) —
	// inv_freq = 1/theta^(arange(0,dim,2)/dim), i.e. head_dim/4 frequencies.
	headDim := hidden / cfg.NumHeads
	rdim := headDim / 2
	e.rotInvFreq = make([]float32, rdim/2)
	for i := range e.rotInvFreq {
		e.rotInvFreq[i] = float32(1.0 / math.Pow(qwenRotaryTheta, float64(2*i)/float64(rdim)))
	}
	return e, nil
}

// qwenTensorPrefix reports the namespace the tower is nested under: "visual." for
// the tiny saved checkpoint, "model.visual." inside a full HF VL safetensors.
func qwenTensorPrefix(st *embed.SafetensorsFile) string {
	for _, pfx := range []string{"visual.", "model.visual."} {
		if _, err := st.Tensor(pfx + "patch_embed.proj.weight"); err == nil {
			return pfx
		}
	}
	return "visual."
}

// Forward runs the ViT + merger on pre-patchified pixel_values [n_patches, patch_dim]
// (patch_dim = in_chans*temporal*patch*patch) with per-image grids (t,h,w in patch
// units, h/w multiples of spatial_merge_size). It returns the merged image
// embeddings [n_merged, out_hidden_size] in ORIGINAL patch order — the embeddings
// that replace the decoder's <image> placeholders. n_merged = Σ t*h*w / merge².
func (e *QwenVisionEncoder) Forward(pixelValues []float32, gridTHW [][3]int) ([]float32, error) {
	hid, err := e.forwardViT(pixelValues, gridTHW)
	if err != nil {
		return nil, err
	}
	return e.merge(hid, gridTHW), nil
}

// ForwardViT runs only the transformer blocks (no merger), returning the pre-merge
// hidden state [n_patches, hidden] in ORIGINAL patch order — the stage the parity
// gate checks against HF's last_hidden_state. (HF returns last_hidden_state in
// WINDOW order; for a self-contained gate we de-window here so the caller sees
// original order. The encoder_test compares against an order-matched golden.)
func (e *QwenVisionEncoder) forwardViT(pixelValues []float32, gridTHW [][3]int) ([]float32, error) {
	c := e.Cfg
	merge := c.SpatialMergeSize
	mergeUnit := merge * merge
	hidden := c.HiddenSize
	patchDim := c.InChans * c.TemporalPatchSize * c.PatchSize * c.PatchSize

	nPatches := 0
	for _, g := range gridTHW {
		// Per-image validation: h and w must each be divisible by
		// spatial_merge_size. The global nPatches%merge² check below is not
		// sufficient — e.g. {1,3,4} with merge 2 passes it (12%4==0) but h=3
		// isn't divisible, so windowIndex/rotaryFreqs emit fewer than `groups`
		// entries and winIdx[g] indexes OOB.
		if g[0] < 0 || g[1] < 0 || g[2] < 0 {
			return nil, fmt.Errorf("vision: negative grid dim in %v", g)
		}
		if g[1]%merge != 0 || g[2]%merge != 0 {
			return nil, fmt.Errorf("vision: grid %v h/w not divisible by spatial_merge_size %d", g, merge)
		}
		nPatches += g[0] * g[1] * g[2]
	}
	if len(pixelValues) != nPatches*patchDim {
		return nil, fmt.Errorf("vision: pixel_values len %d, want %d (%d patches × %d)", len(pixelValues), nPatches*patchDim, nPatches, patchDim)
	}
	if nPatches%mergeUnit != 0 {
		return nil, fmt.Errorf("vision: n_patches %d not a multiple of merge² %d", nPatches, mergeUnit)
	}

	// 1. patch embed: h[n,hidden] = pixel_values[n,patch_dim] · patchWᵀ (no bias).
	h := make([]float32, nPatches*hidden)
	linalg.MatmulBT(pixelValues, e.patchW, h, nPatches, patchDim, hidden)

	// 2. rotary freqs per patch (head_dim/2 each) from the (h_idx,w_idx) grid coords.
	freqs := e.rotaryFreqs(gridTHW) // [nPatches][rdim], original patch order

	// 3. window reorder (at merge-unit granularity) + the two cu_seqlens.
	winIdx, cuWin := e.windowIndex(gridTHW)
	cuFull := cuSeqlensFull(gridTHW)
	groups := nPatches / mergeUnit

	// reorder hidden + freqs into window order, grouping merge_unit patches.
	rdim := len(freqs) / nPatches
	hWin := make([]float32, nPatches*hidden)
	fWin := make([]float32, nPatches*rdim)
	for g := range groups {
		src := winIdx[g]
		for u := range mergeUnit {
			dp, sp := (g*mergeUnit+u)*hidden, (src*mergeUnit+u)*hidden
			copy(hWin[dp:dp+hidden], h[sp:sp+hidden])
			df, sf := (g*mergeUnit+u)*rdim, (src*mergeUnit+u)*rdim
			copy(fWin[df:df+rdim], freqs[sf:sf+rdim])
		}
	}

	// precompute cos/sin per patch over the full head_dim (emb = cat(freqs,freqs)).
	headDim := hidden / c.NumHeads
	cos := make([]float32, nPatches*headDim)
	sin := make([]float32, nPatches*headDim)
	for i := 0; i < nPatches; i++ {
		fr := fWin[i*rdim : i*rdim+rdim]
		for d := range headDim {
			f := float64(fr[d%rdim]) // emb[d]=freqs[d] (d<rdim), freqs[d-rdim] (d≥rdim)
			cos[i*headDim+d] = float32(math.Cos(f))
			sin[i*headDim+d] = float32(math.Sin(f))
		}
	}

	// 4. blocks (pre-norm residual). fullatt blocks attend per-image; others per-window.
	for li := range e.blocks {
		cu := cuWin
		if e.isFullAtt(li) {
			cu = cuFull
		}
		b := &e.blocks[li]
		n1 := rmsNorm(hWin, b.norm1w, nPatches, hidden)
		att := e.attention(n1, b, nPatches, cos, sin, cu)
		o := make([]float32, nPatches*hidden)
		b.projw.matmul(att, o, nPatches)
		addBias(o, b.projb, nPatches, hidden)
		for i := range hWin {
			hWin[i] += o[i]
		}
		n2 := rmsNorm(hWin, b.norm2w, nPatches, hidden)
		mlp := e.mlp(n2, b, nPatches)
		for i := range hWin {
			hWin[i] += mlp[i]
		}
	}

	// de-window back to original patch order (merge-unit granularity).
	out := make([]float32, nPatches*hidden)
	for g := range groups {
		dst := winIdx[g]
		for u := range mergeUnit {
			dp, sp := (dst*mergeUnit+u)*hidden, (g*mergeUnit+u)*hidden
			copy(out[dp:dp+hidden], hWin[sp:sp+hidden])
		}
	}
	return out, nil
}

// ForwardViT exposes the pre-merge hidden state for stage-isolated parity tests.
func (e *QwenVisionEncoder) ForwardViT(pixelValues []float32, gridTHW [][3]int) ([]float32, error) {
	return e.forwardViT(pixelValues, gridTHW)
}

// merge runs the patch merger on the ViT hidden (original patch order):
// RMSNorm(ln_q) → reshape merge² patches into one hidden*merge² vector → mlp.0 →
// GELU(erf) → mlp.2. Output [n_merged, out_hidden], one row per merge-unit group.
func (e *QwenVisionEncoder) merge(hidden []float32, gridTHW [][3]int) []float32 {
	c := e.Cfg
	H := c.HiddenSize
	mergeUnit := c.SpatialMergeSize * c.SpatialMergeSize
	mh := H * mergeUnit
	nPatches := len(hidden) / H
	groups := nPatches / mergeUnit

	// ln_q over hidden, then the [groups, mh] view falls out for free (contiguous).
	nrm := rmsNorm(hidden, e.mergerLNw, nPatches, H)
	mid := make([]float32, groups*mh)
	e.merger0w.matmul(nrm, mid, groups)
	addBias(mid, e.merger0b, groups, mh)
	geluErf(mid)
	out := make([]float32, groups*c.OutHiddenSize)
	e.merger2w.matmul(mid, out, groups)
	addBias(out, e.merger2b, groups, c.OutHiddenSize)
	return out
}

// attention runs bidirectional MHA within each cu_seqlens segment (window or full
// image). qkv is fused (reshape seq,3,heads,head_dim); 2D rotary is applied to q,k
// before attending. Per-head QKᵀ / scores·V run on the f32 SIMD A·Bᵀ kernel.
func (e *QwenVisionEncoder) attention(x []float32, b *qwenBlock, seq int, cos, sin []float32, cu []int) []float32 {
	hidden, nH := e.Cfg.HiddenSize, e.Cfg.NumHeads
	hd := hidden / nH
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	qkv := make([]float32, seq*3*hidden)
	b.qkvw.matmul(x, qkv, seq)
	addBias(qkv, b.qkvb, seq, 3*hidden)
	// split: row layout is [3, nH, hd], so q/k/v are the three contiguous halves.
	q := make([]float32, seq*hidden)
	k := make([]float32, seq*hidden)
	v := make([]float32, seq*hidden)
	for i := range seq {
		base := i * 3 * hidden
		copy(q[i*hidden:(i+1)*hidden], qkv[base:base+hidden])
		copy(k[i*hidden:(i+1)*hidden], qkv[base+hidden:base+2*hidden])
		copy(v[i*hidden:(i+1)*hidden], qkv[base+2*hidden:base+3*hidden])
	}
	// 2D rotary on q,k (NeoX rotate_half over the full head_dim).
	for i := range seq {
		co, si := cos[i*hd:i*hd+hd], sin[i*hd:i*hd+hd]
		for head := range nH {
			off := i*hidden + head*hd
			applyRotaryVision(q[off:off+hd], co, si)
			applyRotaryVision(k[off:off+hd], co, si)
		}
	}

	out := make([]float32, seq*hidden)
	maxSeg := 0
	for s := 1; s < len(cu); s++ {
		if l := cu[s] - cu[s-1]; l > maxSeg {
			maxSeg = l
		}
	}
	qh := make([]float32, maxSeg*hd)
	kh := make([]float32, maxSeg*hd)
	vt := make([]float32, hd*maxSeg)
	scores := make([]float32, maxSeg*maxSeg)
	oh := make([]float32, maxSeg*hd)
	for head := range nH {
		off := head * hd
		for s := 1; s < len(cu); s++ {
			start, n := cu[s-1], cu[s]-cu[s-1]
			for ii := range n {
				gi := start + ii
				copy(qh[ii*hd:(ii+1)*hd], q[gi*hidden+off:gi*hidden+off+hd])
				copy(kh[ii*hd:(ii+1)*hd], k[gi*hidden+off:gi*hidden+off+hd])
				vrow := v[gi*hidden+off : gi*hidden+off+hd]
				for d := range hd {
					vt[d*n+ii] = vrow[d]
				}
			}
			linalg.MatmulBT(qh, kh, scores[:n*n], n, hd, n)
			for i := range n {
				row := scores[i*n : (i+1)*n]
				for j := range row {
					row[j] *= scale
				}
				softmaxRow(row)
			}
			linalg.MatmulBT(scores[:n*n], vt[:hd*n], oh[:n*hd], n, n, hd)
			for ii := range n {
				copy(out[(start+ii)*hidden+off:(start+ii)*hidden+off+hd], oh[ii*hd:(ii+1)*hd])
			}
		}
	}
	return out
}

// mlp runs the gated SiLU MLP: down(silu(gate(x)) * up(x)).
func (e *QwenVisionEncoder) mlp(x []float32, b *qwenBlock, seq int) []float32 {
	hidden, inter := e.Cfg.HiddenSize, e.Cfg.IntermediateSize
	gate := make([]float32, seq*inter)
	b.gatew.matmul(x, gate, seq)
	addBias(gate, b.gateb, seq, inter)
	up := make([]float32, seq*inter)
	b.upw.matmul(x, up, seq)
	addBias(up, b.upb, seq, inter)
	silu(gate)
	for i := range gate {
		gate[i] *= up[i]
	}
	down := make([]float32, seq*hidden)
	b.downw.matmul(gate, down, seq)
	addBias(down, b.downb, seq, hidden)
	return down
}

func (e *QwenVisionEncoder) isFullAtt(layer int) bool {
	return slices.Contains(e.Cfg.FullattBlockIndexes, layer)
}

// rotaryFreqs builds per-patch rotary frequencies [nPatches][head_dim/2] in
// original patch order: per patch, [h_idx*inv_freq..., w_idx*inv_freq...]. The
// (h_idx,w_idx) coords follow HF get_vision_position_ids — the spatial_merge
// interleave so each merge-unit group's patches are consecutive.
func (e *QwenVisionEncoder) rotaryFreqs(gridTHW [][3]int) []float32 {
	merge := e.Cfg.SpatialMergeSize
	nf := len(e.rotInvFreq) // head_dim/4
	rdim := 2 * nf          // head_dim/2
	var out []float32
	for _, g := range gridTHW {
		t, h, w := g[0], g[1], g[2]
		block := make([]float32, 0, h*w*rdim)
		for a := 0; a < h/merge; a++ {
			for c := 0; c < w/merge; c++ {
				for b := range merge {
					for d := range merge {
						hpos, wpos := float32(a*merge+b), float32(c*merge+d)
						for _, f := range e.rotInvFreq {
							block = append(block, hpos*f)
						}
						for _, f := range e.rotInvFreq {
							block = append(block, wpos*f)
						}
					}
				}
			}
		}
		for range t {
			out = append(out, block...)
		}
	}
	return out
}

// windowIndex ports HF get_vision_window_index: groups merge-units into windows of
// (window_size/patch_size/merge)² merged-units, returning the per-group reorder
// indices (length n_patches/merge²) and the cumulative window seqlens (in patch
// units) for the windowed attention blocks.
func (e *QwenVisionEncoder) windowIndex(gridTHW [][3]int) (winIdx, cuWin []int) {
	merge := e.Cfg.SpatialMergeSize
	vmws := e.Cfg.WindowSize / merge / e.Cfg.PatchSize
	mergeUnit := merge * merge
	cuWin = []int{0}
	idOffset := 0
	for _, g := range gridTHW {
		t, h, w := g[0], g[1], g[2]
		llmH, llmW := h/merge, w/merge
		// HF pads up to a window multiple; when already divisible it adds a full
		// (all-pad) window that contributes nothing — replicated via count>0 below.
		padH := vmws - llmH%vmws
		padW := vmws - llmW%vmws
		numWinH := (llmH + padH) / vmws
		numWinW := (llmW + padW) / vmws
		for ti := range t {
			for wh := range numWinH {
				for ww := range numWinW {
					count := 0
					for bi := range vmws {
						for bj := range vmws {
							i, j := wh*vmws+bi, ww*vmws+bj
							if i < llmH && j < llmW {
								winIdx = append(winIdx, idOffset+ti*llmH*llmW+i*llmW+j)
								count++
							}
						}
					}
					if count > 0 { // skip all-pad windows (HF unique_consecutive)
						cuWin = append(cuWin, cuWin[len(cuWin)-1]+count*mergeUnit)
					}
				}
			}
		}
		idOffset += t * llmH * llmW
	}
	return winIdx, cuWin
}

// cuSeqlensFull builds the full-attention boundaries: per image, t segments of h*w
// patches each (HF repeat_interleave(h*w, t).cumsum, padded with a leading 0).
func cuSeqlensFull(gridTHW [][3]int) []int {
	cu := []int{0}
	acc := 0
	for _, g := range gridTHW {
		t, h, w := g[0], g[1], g[2]
		for range t {
			acc += h * w
			cu = append(cu, acc)
		}
	}
	return cu
}

// --- small f32 helpers specific to the Qwen tower (RMSNorm, SiLU, erf-GELU) ---

// rmsNorm is the weight-only RMSNorm (eps 1e-6) HF uses for the Qwen ViT — variance
// over the last dim, no mean subtraction, no bias. Computed in f64 then cast.
func rmsNorm(x, w []float32, rows, dim int) []float32 {
	const eps = 1e-6
	out := make([]float32, rows*dim)
	for r := range rows {
		xr := x[r*dim : r*dim+dim]
		var ss float64
		for _, v := range xr {
			ss += float64(v) * float64(v)
		}
		inv := 1.0 / math.Sqrt(ss/float64(dim)+eps)
		dst := out[r*dim : r*dim+dim]
		for d := range dim {
			dst[d] = float32(float64(xr[d])*inv) * w[d]
		}
	}
	return out
}

// applyRotaryVision applies NeoX rotate_half rotary to one head_dim vector in place,
// given precomputed cos/sin of length head_dim.
func applyRotaryVision(vec, cos, sin []float32) {
	// In-place pairwise: each (vec[d], vec[d+half]) rotates into itself, so no
	// per-call scratch is needed — this was ~8M tiny allocs on a realistic image
	// (~8k patches × 16 heads × 32 blocks). Reads x,y before overwriting either,
	// and is bit-identical to the tmp version (a+(-b)·s == a-b·s in IEEE).
	half := len(vec) / 2
	for d := range half {
		x, y := vec[d], vec[d+half]
		vec[d] = x*cos[d] - y*sin[d]
		vec[d+half] = y*cos[d+half] + x*sin[d+half]
	}
}

func silu(x []float32) {
	for i, v := range x {
		vv := float64(v)
		x[i] = float32(vv / (1.0 + math.Exp(-vv)))
	}
}

// geluErf is the exact (erf) GELU — nn.GELU() default, what the patch merger uses
// (distinct from SigLIP's gelu-tanh).
func geluErf(x []float32) {
	for i, v := range x {
		vv := float64(v)
		x[i] = float32(0.5 * vv * (1.0 + math.Erf(vv/math.Sqrt2)))
	}
}
